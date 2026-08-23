"""
evaluate.py
-----------
Offline evaluation harness. Uses a leave-k-out per-user split: for each user
with >=10 ratings, hide their top-rated movies (rating >= 4.0) as the "held
out" relevant set, retrain/predict on the rest, and check whether the
recommender surfaces them.

Metrics implemented:
- Precision@K, Recall@K
- NDCG@K (rank-aware — a hit at position 1 counts more than position 10)
- Coverage (fraction of the catalogue the system is capable of recommending
  across all users, i.e. does it always recommend the same 10 popular movies?)
- Latency (ms per recommendation call)
"""
import time
import numpy as np
import pandas as pd


def precision_recall_at_k(recommended_ids, relevant_ids, k):
    recommended_k = recommended_ids[:k]
    hits = len(set(recommended_k) & set(relevant_ids))
    precision = hits / k if k > 0 else 0
    recall = hits / len(relevant_ids) if relevant_ids else 0
    return precision, recall


def ndcg_at_k(recommended_ids, relevant_ids, k):
    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for i, mid in enumerate(recommended_k):
        if mid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommended_ids, catalogue_size):
    unique_recommended = len(set(all_recommended_ids))
    return unique_recommended / catalogue_size


def train_test_split_ratings(ratings_df, test_frac=0.3, min_ratings=10, relevance_threshold=4.0, seed=7):
    """
    Per-user leave-k-out split: for users with >= min_ratings, move a
    fraction of their rating>=threshold ("relevant") items into a held-out
    test set and remove them from the training set entirely, so the model
    genuinely never sees them during fit. This gives an honest (not in-sample)
    Precision/Recall/NDCG measurement.
    """
    rng = np.random.default_rng(seed)
    user_counts = ratings_df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= min_ratings].index.tolist()

    test_rows = []
    train_df = ratings_df.copy()

    for user_id in eligible_users:
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        relevant = user_ratings[user_ratings["rating"] >= relevance_threshold]
        if len(relevant) < 2:
            continue
        n_holdout = max(1, int(len(relevant) * test_frac))
        holdout = relevant.sample(n=n_holdout, random_state=int(rng.integers(0, 1e6)))
        test_rows.append(holdout)
        train_df = train_df.drop(holdout.index)

    test_df = pd.concat(test_rows) if test_rows else pd.DataFrame(columns=ratings_df.columns)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), eligible_users


def evaluate_collaborative(collab_model_cls, ratings_df, movies_df, k=10, min_ratings=10,
                            relevance_threshold=4.0, test_frac=0.3):
    """
    Proper held-out evaluation: fits a fresh model on a train split with each
    eligible user's top-rated movies withheld, then checks whether those
    withheld ("relevant") movies get recommended from the remaining catalogue.
    """
    train_df, test_df, eligible_users = train_test_split_ratings(
        ratings_df, test_frac=test_frac, min_ratings=min_ratings, relevance_threshold=relevance_threshold
    )
    model = collab_model_cls(train_df, movies_df)

    results = {"precision": [], "recall": [], "ndcg": [], "latency_ms": []}
    all_recs = []

    for user_id in test_df["user_id"].unique():
        relevant_ids = set(test_df[test_df["user_id"] == user_id]["movie_id"])
        if not relevant_ids or user_id not in model.user_to_idx:
            continue

        start = time.time()
        recs, err = model.recommend(user_id, top_n=k)
        latency = (time.time() - start) * 1000
        if err:
            continue

        recommended_ids = [r["movie_id"] for r in recs]
        all_recs.extend(recommended_ids)

        p, r_ = precision_recall_at_k(recommended_ids, relevant_ids, k)
        ndcg = ndcg_at_k(recommended_ids, relevant_ids, k)

        results["precision"].append(p)
        results["recall"].append(r_)
        results["ndcg"].append(ndcg)
        results["latency_ms"].append(latency)

    summary = {
        "n_users_evaluated": len(results["precision"]),
        f"precision@{k}": float(np.mean(results["precision"])) if results["precision"] else 0,
        f"recall@{k}": float(np.mean(results["recall"])) if results["recall"] else 0,
        f"ndcg@{k}": float(np.mean(results["ndcg"])) if results["ndcg"] else 0,
        "coverage": coverage(all_recs, len(movies_df)),
        "avg_latency_ms": float(np.mean(results["latency_ms"])) if results["latency_ms"] else 0,
    }
    return summary


def evaluate_content(content_model, movies_df, sample_titles, k=10):
    """
    Content-based sanity check: for a sample of titles, verify recommended
    titles share at least one genre with the query title (a weak but useful
    'is this even topically relevant' check, since there's no ground-truth
    relevance label for content similarity).
    """
    genre_match_rates = []
    latencies = []
    for title in sample_titles:
        start = time.time()
        recs, err = content_model.recommend(title, top_n=k)
        latencies.append((time.time() - start) * 1000)
        if err or not recs:
            continue
        query_genres = set(movies_df[movies_df["title"] == title].iloc[0]["genres"].split("|"))
        matches = 0
        for r in recs:
            rec_genres = set(r["genres"].split("|"))
            if query_genres & rec_genres:
                matches += 1
        genre_match_rates.append(matches / len(recs))

    return {
        "n_titles_evaluated": len(genre_match_rates),
        "avg_genre_overlap_rate": float(np.mean(genre_match_rates)) if genre_match_rates else 0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_movies, load_ratings
    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeRecommender

    movies_df = load_movies()
    ratings_df = load_ratings()

    print("=== Content-Based Evaluation ===")
    cb = ContentBasedRecommender(movies_df)
    sample = movies_df["title"].sample(15, random_state=1).tolist()
    print(evaluate_content(cb, movies_df, sample))

    print("\n=== Collaborative Filtering Evaluation (held-out split) ===")
    print(evaluate_collaborative(CollaborativeRecommender, ratings_df, movies_df))
