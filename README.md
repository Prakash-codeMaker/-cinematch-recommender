# 🎞️ CineMatch — Hybrid Movie Recommendation Engine

A from-scratch recommendation system built for the **Technical Assignment – Recommendation System**. CineMatch recommends movies using a **hybrid of content-based filtering and collaborative filtering** to deliver personalized, explainable recommendations.

> **Live demo:** [https://prakash-codemaker--cinematch-recommender-app-akrtfv.streamlit.app/](https://prakash-codemaker--cinematch-recommender-app-akrtfv.streamlit.app/)
> **GitHub repo:** [https://github.com/Prakash-codeMaker/-cinematch-recommender](https://github.com/Prakash-codeMaker/-cinematch-recommender)

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Use Case & Motivation](#2-use-case--motivation)
3. [Approach](#3-approach)
4. [System Architecture](#4-system-architecture)
5. [Recommendation Methodology](#5-recommendation-methodology)
6. [Dataset](#6-dataset)
7. [Technologies Used](#7-technologies-used)
8. [Assumptions](#8-assumptions)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Setup & How to Run](#10-setup--how-to-run)
11. [Deployment](#11-deployment)
12. [Evaluation Methodology & Metrics](#12-evaluation-methodology--metrics)
13. [Test Cases](#13-test-cases)
14. [Known Limitations](#14-known-limitations)
15. [Future Improvements](#15-future-improvements)
16. [Bonus: Comparison with Netflix](#16-bonus-comparison-with-netflix)

---

## 1. Problem Statement

Given a catalogue of movies and a history of user ratings, recommend movies that a specific user (or a specific movie's viewers) is likely to enjoy — while being able to explain *why* each recommendation was made. This assignment tests the ability to build a **production-ready, explainable recommendation system** from scratch.

## 2. Use Case & Motivation

**Movie recommendations** were chosen because the domain:
- Has rich, well-understood metadata (genre, director, plot) that supports a genuinely explainable content-based model — not just a black box.
- Naturally supports a **hybrid** design, which is more representative of production recommender systems (Netflix, Prime Video) than either technique alone, and lets this assignment demonstrate handling of cold-start users.
- Is intuitive to evaluate manually — a reviewer can sanity-check "Inception → similar sci-fi/thriller films" without domain expertise.

## 3. Approach

Three progressively combined strategies:

1. **Content-Based Filtering** — "because you watched X, here's similar Y" — using TF-IDF over genre/director/plot text and cosine similarity. Works with zero rating history (no cold start for new users).
2. **Collaborative Filtering** — "users like you also enjoyed" — using matrix factorization (truncated SVD) over the user–item ratings matrix. Captures latent taste patterns that plain genre-matching can't.
3. **Hybrid Blending** — combines both with a weighted score, where the weight **automatically shifts toward content-based for sparsely-rated (cold-start) users** and toward collaborative for well-established users.

## 4. System Architecture

```
                         ┌─────────────────────────┐
                         │        data/             │
                         │  movies.csv  ratings.csv │
                         │      users.csv           │
                         └────────────┬─────────────┘
                                      │
                          src/data.py │ (loaders)
                                      ▼
        ┌─────────────────────┐            ┌──────────────────────────┐
        │ content_based.py     │            │  collaborative.py         │
        │ TF-IDF + cosine sim   │            │  User-Item matrix + SVD   │
        │ (item-item)           │            │  (matrix factorization)   │
        └──────────┬───────────┘            └────────────┬─────────────┘
                   │                                       │
                   └──────────────────┬────────────────────┘
                                      ▼
                           ┌──────────────────────┐
                           │     hybrid.py          │
                           │  weighted score blend  │
                           │  + auto cold-start α    │
                           └──────────┬─────────────┘
                                      ▼
                           ┌──────────────────────┐
                           │      app.py            │
                           │  Streamlit UI (4 modes) │
                           │  content / collab /     │
                           │  hybrid / evaluation     │
                           └──────────────────────┘

                 (src/evaluate.py sits alongside all three
                  models and runs offline metrics on demand)
```

**Data flow for a single request (Hybrid mode):**
`user_id` → `hybrid.py` reads the user's rating count → picks blend weight α → calls `collaborative.py` for top-40 SVD-predicted movies → calls `content_based.py` for similarity to the user's highest-rated movie → returns ranked, blended list.

## 5. Recommendation Methodology

| Technique | Input | Core Algorithm | Output |
|---|---|---|---|
| Content-Based | one movie title | TF-IDF vectorization of `genres + director + plot` → cosine similarity matrix | top-N most similar movies, with explicit shared-genre/director explanation |
| Collaborative Filtering | one `user_id` | Mean-center the user–item ratings matrix → truncated SVD (`k=20` latent factors) → reconstruct predicted ratings → rank unrated movies | top-N predicted-rating-ranked movies (no explicit reasons, but latent factors inferred) |
| Hybrid | one `user_id` | `score = α · norm(collab_score) + (1-α) · norm(content_score vs. user's top-rated movie)`, where `α = clip(n_ratings / 40, 0.15, 0.85)` | top-N blended ranking with combined reasoning ("Collab: similar taste" + "Content: shares X genre with Y") |

## 6. Dataset

**147 real, well-known movies** (title, year, genre(s), director, one-line plot) spanning Hollywood, Bollywood, anime, and international cinema — curated by hand rather than scraped, so the catalogue is manageable and vetted.

**300 synthetic users, ~9,700 ratings** — generated from **8 latent "taste archetypes"** (e.g. *Action-Sci-Fi Fan*, *Prestige Awards Fan*, *Bollywood Fan*; see `data/generate_data.py`). Each synthetic user's ratings follow a probability distribution linked to their archetype and a random bias term, creating realistic-looking rating patterns.

**Why synthetic ratings instead of MovieLens?** This project was built in a sandboxed environment without general internet egress, so public datasets (MovieLens, TMDB, etc.) could not be downloaded at build time. The synthetic data is deterministic, reproducible, and small enough to ship as flat CSVs in the repo.

Run `python3 data/generate_data.py` to regenerate the dataset (or swap in real data).

## 7. Technologies Used

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Standard for ML prototyping, rich ecosystem |
| Data | pandas, NumPy | Standard tabular manipulation |
| Content-based model | scikit-learn (`TfidfVectorizer`, `cosine_similarity`) | Battle-tested, interpretable, no GPU needed |
| Collaborative model | SciPy (`svds`) | Lightweight matrix factorization without a full deep-learning dependency |
| UI | Streamlit | Fastest path to a genuinely interactive, deployable testing interface for a time-boxed assignment |
| Testing | pytest | Standard Python testing, also runnable stand-alone without pytest installed |
| Deployment target | Streamlit Community Cloud | Free, zero-config, directly from a GitHub repo |

## 8. Assumptions

- Evaluators care more about **correct, explainable, testable recommendation logic** than about visual polish or a production-grade database — so ratings/movies are flat CSVs, not a real database, and the UI is minimal.
- A rating **≥ 4.0** (out of 5) is treated as "relevant" / "liked" for evaluation purposes (precision/recall/NDCG).
- Users are anonymous/numeric IDs (no auth) since this is a demo, not a production login system.
- The synthetic ratings' latent taste structure is a reasonable proxy for real user behavior for the purposes of *demonstrating* the collaborative algorithm — it is **not** a claim that real audiences actually follow an 8-archetype model.

## 9. Key Design Decisions

- **TF-IDF over embeddings for content-based**: keeps the system dependency-light, fast, and fully explainable (you can point to the exact shared genre/director that drove a match) — appropriate for a time-boxed assignment where interpretability is as important as accuracy.
- **SVD over user/item k-NN for collaborative**: scales sub-quadratically as the catalogue grows and captures latent taste dimensions that raw genre tags can't (e.g., "prefers slow-burn direction" across thriller and drama genres).
- **Auto-adjusting hybrid weight (α)** instead of a fixed 50/50 blend: this is the single most important design decision in the system — it means a brand-new user with 3 ratings doesn't get unreliable collaborative predictions, but a heavy rater (40+) gradually trusts the collaborative model more. Formula: `α = clip(n_ratings / 40, 0.15, 0.85)` — the `clip` ensures we never go pure collaborative (min 0.15 content-based) even for superusers.
- **Explanation strings are first-class, not an afterthought**: every recommendation object carries a `reason` field surfaced directly in the UI, satisfying the assignment's "understand why or how recommendations are made."
- **Errors return `(results, error_message)` tuples, never exceptions**, so the UI and test suite can assert on graceful failure (unknown movie, unknown user) instead of crashing.

## 10. Setup & How to Run

```bash
# 1. Clone
git clone https://github.com/Prakash-codeMaker/-cinematch-recommender
cd cinematch-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the dataset — a copy is already committed in data/
python3 data/generate_data.py

# 4. Run the app
streamlit run app.py
# → open http://localhost:8501

# 5. Run tests
python3 -m pytest tests/ -v
# or, without pytest installed:
python3 tests/test_recommender.py

# 6. Run the evaluation script directly (prints metrics to console)
python3 src/evaluate.py
```

## 11. Deployment

Deployed via **Streamlit Community Cloud** (free tier):

1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Select the repo, branch `main`, and entry point `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically and gives you a public `*.streamlit.app` URL.
5. Paste that URL at the top of this README and in your submission.

(Any other host — Render, Railway, Hugging Face Spaces — works identically since the app has no external dependencies beyond `requirements.txt` and the bundled CSVs.)

## 12. Evaluation Methodology & Metrics

Implemented in `src/evaluate.py`, runnable standalone or from the UI's **"System evaluation"** tab.

**Collaborative filtering** — proper held-out evaluation (not in-sample):
- For every user with ≥10 ratings, **30% of their rating≥4.0 ("relevant") movies are removed from the training matrix entirely** before the SVD model is fit.
- The model is then asked to recommend from the remaining catalogue, and we measure whether it surfaces the held-out relevant movies.
- **Precision@10** — fraction of the top-10 recommendations that were actually relevant.
- **Recall@10** — fraction of the user's held-out relevant movies that appeared in the top-10.
- **NDCG@10** — rank-aware version of the above (a hit at position 1 counts more than position 10).
- **Coverage** — fraction of the entire catalogue the system is capable of recommending across all evaluated users (checks the system isn't just repeatedly recommending the same 10 popular movies).
- **Latency** — wall-clock ms per recommendation call.

**Content-based filtering** — since there's no ground-truth "relevance" label for similarity, we use a genre-overlap sanity check: for a random sample of query movies, what fraction of the top-10 recommendations share at least one genre with the query? (Expect ~90%+ if the model is working.)

**Latest run (see `src/evaluate.py` output):**

| Metric | Value |
|---|---|
| Precision@10 (collaborative) | ~0.12 |
| Recall@10 (collaborative) | ~0.36 |
| NDCG@10 (collaborative) | ~0.28 |
| Catalogue coverage | ~95% |
| Avg latency (collaborative) | ~4 ms |
| Avg genre overlap (content-based) | ~93% |
| Avg latency (content-based) | ~3 ms |

*(Exact numbers vary slightly run-to-run because the held-out split re-samples; re-run `python3 src/evaluate.py` to reproduce.)*

**Why these numbers, not "0.9 precision"**: with only 147 movies and a 30% per-user holdout, random chance alone would put Precision@10 around 0.03–0.05 (holding out ~3-6 movies out of 147 and hoping all 10 recs are hits is unlikely). Our ~0.12 shows the model is learning real patterns, not just overfitting to popularity. Real-world systems on MovieLens-100K see Precision@10 ≈ 0.4–0.6, but that's with 100K ratings, 6K movies, and years of tuning.

## 13. Test Cases

Full automated suite in `tests/test_recommender.py` (9 tests, run via `pytest tests/ -v`).

### ✅ Successful Scenarios
| Test | What it verifies |
|---|---|
| `test_content_based_returns_genre_overlapping_results` | Recommending from "Inception" returns ≥6/10 genre-overlapping movies |
| `test_content_based_same_director_ranks_high` | Nolan films surface each other via the director signal |
| `test_collaborative_recommends_unseen_movies_only` | Collaborative filtering never re-recommends an already-rated movie |
| `test_hybrid_blends_and_returns_results` | Hybrid output is correctly ranked descending by blended score |
| `test_cold_start_user_leans_on_content_weight` | A sparse-rating user gets a **lower** collaborative weight (α) than a dense-rating user — proves the cold-start adaptation actually works, not just a claim |

### ❌ Failure Scenarios (the system's honesty about its own limits)
| Test | What it verifies |
|---|---|
| `test_content_based_unknown_movie_fails_gracefully` | Misspelled/unknown title → clean error message, not a crash |
| `test_collaborative_unknown_user_fails_gracefully` | Unknown `user_id` (classic new-user cold start) → clean error, not a crash |
| `test_content_based_niche_movie_yields_weak_matches` | A movie in a thin genre (e.g. *Free Solo*, Documentary) still returns 10 results, but with visibly weak similarity scores (<0.5) — the UI shows this, not hidden |
| `test_hybrid_unknown_user_fails_gracefully` | Hybrid layer correctly propagates the underlying error instead of raising an exception |

## 14. Known Limitations

- **Synthetic ratings, not real user behavior.** The collaborative model's usefulness is demonstrated on data generated from known taste archetypes; real-world ratings are noisier and less structured.
- **No true user cold start in the UI.** The hybrid model adapts its *weighting* based on rating count, but a genuinely brand-new user (0 ratings, not in the matrix at all) currently falls back to a content-based-only recommendation. A production system would add an onboarding flow.
- **Small catalogue (147 movies).** Coverage and diversity metrics look strong partly because the candidate pool is small; behavior at Netflix-scale (thousands to millions of items) would need approximation algorithms (e.g., locality-sensitive hashing).
- **SVD has no explicit regularization or bias terms** (unlike, e.g., the regularized-SGD approaches used in the Netflix Prize), so predicted ratings can occasionally fall outside the 1–5 range at the extremes.
- **Content similarity is genre/director/plot-keyword based**, not semantic — two movies with similar *themes* but no shared genre tag or plot keywords will not be matched (e.g., a horror movie and a thriller with the same psychological premise).
- **Streamlit's `@st.cache_resource`** means the models are fit once per server process; a genuinely new movie/user added after startup wouldn't appear until the app restarts or cache is cleared.

## 15. Future Improvements

1. **Swap in real MovieLens/TMDB data** via `load_movielens()` for a production-grade evaluation.
2. **New-user onboarding flow** ("rate 5 movies to get started") to solve the true cold-start case, not just the sparse-rating case.
3. **Approximate nearest-neighbor search (FAISS/Annoy)** for the content-based similarity matrix to scale past a few thousand items.
4. **Learned hybrid weighting** (e.g., a small logistic regression over [α, user activity, recency] trained to predict click-through) instead of the current hand-tuned linear formula.
5. **Diversity/serendipity re-ranking** (e.g., Maximal Marginal Relevance) so top-10 lists aren't dominated by near-duplicate sequels/genres.
6. **Session-based signals** (what the user just clicked in this session) blended in alongside the static rating history.
7. **A/B evaluation harness** to compare hybrid-α strategies against each other with simulated user feedback, not just static held-out metrics.

## 16. Bonus: Comparison with Netflix

CineMatch's UI deliberately borrows Netflix's dark theme, card-based browsing, and "Because you watched X" framing.

| | Netflix | CineMatch |
|---|---|---|
| **Similarities** | Dark UI, card-based rows, "Because you watched" content-based row, personalized "For You" row | Same visual language and same two-row mental model (content-based / collaborative), same goal of *explaining* the recommendation |
| **Differences** | Recommendations blend dozens of signals (watch time, time-of-day, device, thumbnail A/B tests, social signals) and are served by a real-time, multi-stage ranking pipeline at massive scale (millions of users, thousands of titles) | Single signal per strategy (content or collaborative) and batch-offline models |
| **Current limitations (CineMatch)** | — | No implicit signals (watch time, skips, replays) — only explicit 1–5 ratings; no session/context awareness; no diversity re-ranking, so a top-10 list can be homogeneous |
| **Areas for improvement** | — | Diversity re-ranking, session signals, real dataset, learned hybrid weight (see [§15](#15-future-improvements)) |
| **What we'd build next with more time** | — | A/B testing harness for α strategies, a lightweight implicit-feedback signal (e.g. "watched to completion" simulated from session data), and approximation algorithms to handle Netflix-scale catalogues |

---

## Repository Structure

```
movie-recommender/
├── app.py                      # Streamlit UI (4 modes: content / collaborative / hybrid / evaluation)
├── requirements.txt
├── README.md                   # this file
├── LICENSE
├── .streamlit/config.toml      # dark theme config for deployment
├── data/
│   ├── generate_data.py        # builds movies.csv / ratings.csv / users.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── users.csv
├── src/
│   ├── data.py                 # loaders + optional real-MovieLens importer
│   ├── content_based.py        # TF-IDF + cosine similarity
│   ├── collaborative.py        # SVD matrix factorization
│   ├── hybrid.py                # weighted blend + cold-start handling
│   └── evaluate.py             # Precision/Recall/NDCG/Coverage/Latency
└── tests/
    └── test_recommender.py     # 9 automated tests (success + failure cases)
```
