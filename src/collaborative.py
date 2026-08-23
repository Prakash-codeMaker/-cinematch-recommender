"""
collaborative.py
-----------------
Collaborative filtering via matrix factorization (truncated SVD on the
mean-centered user-item ratings matrix). Recommends movies for a *given
user_id* based on patterns across all users, without needing item metadata.

Why SVD over user-user / item-item kNN:
- Scales better as the catalogue grows (dense low-rank factors vs O(n^2)
  similarity matrices).
- Captures latent taste dimensions that raw genre tags miss (e.g. "prefers
  slow-burn direction" cutting across multiple genres).
- Standard, well-understood baseline (this is the same idea behind the
  2009 Netflix Prize-winning approaches, minus the bias terms/regularized
  SGD training they used).

Trade-off (documented as a limitation): pure SVD here has no built-in cold
start handling for brand-new users/items, and latent factors aren't
directly human-interpretable the way genre overlap is -- hence the hybrid
module blends this with content-based explainability.
"""
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class CollaborativeRecommender:
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n_factors: int = 20):
        self.movies_df = movies_df.reset_index(drop=True)
        self.ratings_df = ratings_df
        self.n_factors = n_factors
        self._build_matrix()
        self._fit()

    def _build_matrix(self):
        self.user_ids = sorted(self.ratings_df["user_id"].unique())
        self.movie_ids = sorted(self.movies_df["movie_id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

        mat = np.zeros((len(self.user_ids), len(self.movie_ids)))
        for row in self.ratings_df.itertuples():
            mat[self.user_to_idx[row.user_id], self.movie_to_idx[row.movie_id]] = row.rating
        self.raw_matrix = mat
        self.rated_mask = mat > 0

    def _fit(self):
        mat = self.raw_matrix
        counts = (mat != 0).sum(1)
        with np.errstate(invalid="ignore"):
            self.user_means = np.where(counts > 0, mat.sum(1) / np.where(counts > 0, counts, 1), 3.0)
        centered = mat.copy()
        for i in range(mat.shape[0]):
            centered[i, self.rated_mask[i]] -= self.user_means[i]

        k = min(self.n_factors, min(mat.shape) - 1)
        U, sigma, Vt = svds(centered, k=k)
        sigma = np.diag(sigma)
        self.predicted = U @ sigma @ Vt + self.user_means.reshape(-1, 1)

    def recommend(self, user_id: int, top_n: int = 10):
        if user_id not in self.user_to_idx:
            return [], f"user_id {user_id} not found."

        u_idx = self.user_to_idx[user_id]
        preds = self.predicted[u_idx].copy()
        already_rated = self.rated_mask[u_idx]
        preds[already_rated] = -np.inf  # never recommend what they've already rated

        top_idx = np.argsort(preds)[::-1][:top_n]
        results = []
        for m_idx in top_idx:
            if preds[m_idx] == -np.inf:
                continue
            movie_id = self.idx_to_movie[m_idx]
            row = self.movies_df[self.movies_df["movie_id"] == movie_id].iloc[0]
            results.append({
                "movie_id": int(movie_id),
                "title": row["title"],
                "year": int(row["year"]),
                "genres": row["genres"],
                "score": round(float(preds[m_idx]), 3),
                "reason": "predicted from rating patterns of users with similar taste",
            })
        return results, None

    def user_rated_titles(self, user_id: int):
        sub = self.ratings_df[self.ratings_df["user_id"] == user_id].sort_values("rating", ascending=False)
        merged = sub.merge(self.movies_df, on="movie_id")
        return merged[["title", "rating", "genres"]]
