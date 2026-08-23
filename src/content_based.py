"""
content_based.py
-----------------
Content-based filtering: recommends movies similar to a *given movie*
using TF-IDF over a combined "soup" of genres + director + plot, then
cosine similarity.

Why TF-IDF + cosine similarity:
- Zero cold-start cost for new items (only needs metadata, no ratings).
- Fully explainable: we can show exactly which shared genres/plot terms
  drove a recommendation.
- Cheap to compute and easy to reason about for an assignment of this scope,
  vs. an embedding model that would add an opaque dependency.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        self._build_soup()
        self._fit()

    def _build_soup(self):
        def soup(row):
            genres = row["genres"].replace("|", " ")
            # repeat genres 2x so they weigh more heavily than single plot words
            return f"{genres} {genres} {row['director'].replace(' ', '')} {row['plot']}"

        self.movies_df["soup"] = self.movies_df.apply(soup, axis=1)

    def _fit(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["soup"])
        self.sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.title_to_idx = {t: i for i, t in enumerate(self.movies_df["title"])}

    def recommend(self, title: str, top_n: int = 10):
        if title not in self.title_to_idx:
            return [], f"'{title}' not found in catalogue."

        idx = self.title_to_idx[title]
        sims = list(enumerate(self.sim_matrix[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        sims = [s for s in sims if s[0] != idx][:top_n]

        results = []
        for i, score in sims:
            row = self.movies_df.iloc[i]
            reason = self._explain(idx, i)
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "year": int(row["year"]),
                "genres": row["genres"],
                "score": round(float(score), 4),
                "reason": reason,
            })
        return results, None

    def _explain(self, idx_a, idx_b):
        genres_a = set(self.movies_df.iloc[idx_a]["genres"].split("|"))
        genres_b = set(self.movies_df.iloc[idx_b]["genres"].split("|"))
        shared = genres_a & genres_b
        director_a = self.movies_df.iloc[idx_a]["director"]
        director_b = self.movies_df.iloc[idx_b]["director"]
        parts = []
        if shared:
            parts.append(f"shares genre(s): {', '.join(sorted(shared))}")
        if director_a == director_b:
            parts.append(f"same director ({director_a})")
        if not parts:
            parts.append("similar plot themes (TF-IDF match)")
        return "; ".join(parts)

    def similar_movies_matrix(self):
        return self.sim_matrix
