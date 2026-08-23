"""
data.py - loading utilities for movies/ratings.

Includes a drop-in load_movielens() stub: if you have internet access,
point it at a MovieLens ml-latest-small export and it will reshape it into
the same movies.csv / ratings.csv schema this project expects, so every
other module (content_based, collaborative, hybrid, evaluate) works
unmodified on real data.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_movies() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))


def load_users() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "users.csv"))


def load_movielens(movies_csv_path: str, ratings_csv_path: str):
    """
    Optional real-data loader. Given MovieLens' raw movies.csv (movieId,title,genres)
    and ratings.csv (userId,movieId,rating,timestamp), reshape + save into this
    project's schema so the rest of the codebase needs zero changes.
    """
    ml_movies = pd.read_csv(movies_csv_path)
    ml_ratings = pd.read_csv(ratings_csv_path)

    movies_df = pd.DataFrame({
        "movie_id": ml_movies["movieId"],
        "title": ml_movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True),
        "year": ml_movies["title"].str.extract(r"\((\d{4})\)$").fillna(0).astype(int),
        "genres": ml_movies["genres"].str.replace("|", "|", regex=False),
        "director": "Unknown",
        "plot": "",
    })
    ratings_df = ml_ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})[
        ["user_id", "movie_id", "rating"]
    ]

    movies_df.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)
    ratings_df.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
    return movies_df, ratings_df
