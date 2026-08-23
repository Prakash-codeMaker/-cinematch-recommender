"""
test_recommender.py
--------------------
Run with: pytest tests/ -v   (or: python3 tests/test_recommender.py)

Covers the "Successful Scenarios" and "Failure Scenarios" required by the
assignment. Each test doubles as living documentation of expected behaviour.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_movies, load_ratings
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

movies_df = load_movies()
ratings_df = load_ratings()
content_model = ContentBasedRecommender(movies_df)
collab_model = CollaborativeRecommender(ratings_df, movies_df)
hybrid_model = HybridRecommender(content_model, collab_model, movies_df, ratings_df)


# ----------------------------------------------------------- successful ---
def test_content_based_returns_genre_overlapping_results():
    """SUCCESS: recommending from a clear-genre anchor (Inception, Sci-Fi/Action/Thriller)
    should return movies sharing at least one of those genres for most results."""
    recs, err = content_model.recommend("Inception", top_n=10)
    assert err is None
    assert len(recs) == 10
    anchor_genres = set(movies_df[movies_df["title"] == "Inception"].iloc[0]["genres"].split("|"))
    overlap_count = sum(1 for r in recs if anchor_genres & set(r["genres"].split("|")))
    assert overlap_count >= 6, f"expected strong genre overlap, got {overlap_count}/10"


def test_content_based_same_director_ranks_high():
    """SUCCESS: two Christopher Nolan films should surface each other,
    since director is part of the similarity 'soup'."""
    recs, err = content_model.recommend("Interstellar", top_n=15)
    titles = [r["title"] for r in recs]
    nolan_films = {"Inception", "The Dark Knight", "The Prestige", "Dunkirk"}
    assert len(nolan_films & set(titles)) >= 1


def test_collaborative_recommends_unseen_movies_only():
    """SUCCESS: collaborative recommendations must never include a movie
    the user has already rated."""
    user_id = ratings_df["user_id"].iloc[0]
    already_rated = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"])
    recs, err = collab_model.recommend(user_id, top_n=10)
    assert err is None
    recommended_ids = {r["movie_id"] for r in recs}
    assert recommended_ids.isdisjoint(already_rated)


def test_hybrid_blends_and_returns_results():
    """SUCCESS: hybrid recommender should return ranked results with both
    content and collaborative reasoning present."""
    user_id = ratings_df["user_id"].iloc[0]
    recs, err, alpha = hybrid_model.recommend_for_user(user_id, top_n=5)
    assert err is None
    assert len(recs) == 5
    assert 0.0 <= alpha <= 1.0
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True), "results must be ranked descending by score"


def test_cold_start_user_leans_on_content_weight():
    """SUCCESS: a sparsely-rated user should get a LOWER alpha (collaborative
    weight) than a heavily-rated user -- i.e. the system correctly detects
    and compensates for cold start."""
    counts = ratings_df.groupby("user_id").size().sort_values()
    sparse_user = counts.index[0]
    dense_user = counts.index[-1]
    _, _, alpha_sparse = hybrid_model.recommend_for_user(sparse_user, top_n=5)
    _, _, alpha_dense = hybrid_model.recommend_for_user(dense_user, top_n=5)
    assert alpha_sparse <= alpha_dense


# -------------------------------------------------------------- failure ---
def test_content_based_unknown_movie_fails_gracefully():
    """FAILURE CASE: an unknown/misspelled movie title should return an
    empty list and a clear error message, never a crash."""
    recs, err = content_model.recommend("This Movie Does Not Exist 9999", top_n=5)
    assert recs == []
    assert err is not None
    assert "not found" in err.lower()


def test_collaborative_unknown_user_fails_gracefully():
    """FAILURE CASE: a user_id outside the training data should fail
    gracefully -- this is the classic 'new user cold start' limitation of
    pure collaborative filtering, documented in README.md."""
    recs, err = collab_model.recommend(user_id=999999, top_n=5)
    assert recs == []
    assert err is not None


def test_content_based_niche_movie_yields_weak_matches():
    """FAILURE / DEGRADED CASE: a movie whose genre tag is rare in the
    catalogue (e.g. a lone Documentary) should still return top_n results,
    but similarity scores should be visibly weaker (near zero) -- this is
    an honest limitation of TF-IDF genre overlap for thin genres, and the
    UI surfaces the low score rather than hiding it."""
    recs, err = content_model.recommend("Free Solo", top_n=10)
    assert err is None
    assert len(recs) == 10
    top_score = recs[0]["score"]
    assert top_score < 0.5, f"expected weak similarity for a niche genre, got {top_score}"


def test_hybrid_unknown_user_fails_gracefully():
    """FAILURE CASE: hybrid recommender should surface the same clear error
    as the underlying collaborative model rather than raising an exception."""
    recs, err, alpha = hybrid_model.recommend_for_user(999999, top_n=5)
    assert recs == []
    assert err is not None


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
