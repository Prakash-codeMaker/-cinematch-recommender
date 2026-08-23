"""
hybrid.py
---------
Combines content-based and collaborative signals.

Strategy: weighted score blending.
  final_score = alpha * normalized_collaborative_score
              + (1 - alpha) * normalized_content_score(vs. user's top-rated movie)

Why weighted hybrid over switching/cascade hybrids:
- Handles the collaborative cold-start problem gracefully: if a user is new
  or sparsely rated (few overlapping ratings with others), collaborative
  scores are unreliable, so alpha is auto-lowered in favor of content
  similarity to that user's known-liked movie(s).
- Keeps both signals' explanations intact, so the UI can show *why* -- genre
  overlap from content-based, "taste-pattern match" from collaborative --
  rather than a single opaque score.
"""
import numpy as np
import pandas as pd


class HybridRecommender:
    def __init__(self, content_model, collab_model, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.content_model = content_model
        self.collab_model = collab_model
        self.movies_df = movies_df
        self.ratings_df = ratings_df

    def _min_max_norm(self, scores: dict):
        if not scores:
            return scores
        vals = np.array(list(scores.values()))
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-9:
            return {k: 0.5 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def recommend_for_user(self, user_id: int, top_n: int = 10, alpha: float = None):
        n_user_ratings = len(self.ratings_df[self.ratings_df["user_id"] == user_id])

        # auto cold-start handling: fewer ratings -> lean more on content-based
        if alpha is None:
            alpha = float(np.clip(n_user_ratings / 40.0, 0.15, 0.85))

        collab_results, err = self.collab_model.recommend(user_id, top_n=40)
        if err:
            return [], err, alpha

        collab_scores = {r["movie_id"]: r["score"] for r in collab_results}
        collab_norm = self._min_max_norm(collab_scores)

        # content signal: similarity to the user's single highest-rated movie
        top_rated = self.collab_model.user_rated_titles(user_id)
        content_norm = {}
        anchor_title = None
        if len(top_rated) > 0:
            anchor_title = top_rated.iloc[0]["title"]
            content_results, _ = self.content_model.recommend(anchor_title, top_n=60)
            content_scores = {r["movie_id"]: r["score"] for r in content_results}
            content_norm = self._min_max_norm(content_scores)

        blended = []
        for r in collab_results:
            mid = r["movie_id"]
            c_score = collab_norm.get(mid, 0.0)
            ct_score = content_norm.get(mid, 0.0)
            final = alpha * c_score + (1 - alpha) * ct_score
            reason = f"collaborative: taste-pattern match"
            if ct_score > 0 and anchor_title:
                reason += f"; content: similar to your top-rated '{anchor_title}'"
            blended.append({**r, "score": round(float(final), 4), "reason": reason})

        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:top_n], None, alpha

    def recommend_similar(self, title: str, top_n: int = 10):
        """Pure content-based path for the 'because you're viewing X' use case."""
        return self.content_model.recommend(title, top_n=top_n)
