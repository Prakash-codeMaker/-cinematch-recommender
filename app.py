"""
CineMatch — Hybrid Movie Recommendation Engine
Streamlit testing interface for evaluators.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from data import load_movies, load_ratings, load_users
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

st.set_page_config(page_title="CineMatch — Recommendation Engine", page_icon="🎞️", layout="wide")

# ---------------------------------------------------------------- styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0C0D10;
  --bg-panel: #16171C;
  --accent: #C8102E;
  --accent-soft: #E8455F;
  --gold: #D4AF37;
  --text: #ECECEC;
  --text-dim: #9A9AA2;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.03em; }

.cine-hero {
  padding: 1.6rem 0 1rem 0;
  border-bottom: 1px solid #232429;
  margin-bottom: 1.4rem;
}
.cine-hero h1 {
  font-size: 3rem;
  color: var(--text);
  margin: 0;
  line-height: 1;
}
.cine-hero h1 span { color: var(--accent); }
.cine-hero p { color: var(--text-dim); margin-top: 0.3rem; font-size: 0.95rem; }

.sprocket-divider {
  display: flex; gap: 6px; margin: 0.6rem 0 1.2rem 0;
}
.sprocket-divider span {
  width: 8px; height: 8px; background: #2A2B31; border-radius: 2px;
}

.movie-card {
  background: var(--bg-panel);
  border: 1px solid #232429;
  border-radius: 6px;
  padding: 0.9rem 1.1rem;
  margin-bottom: 0.6rem;
  transition: border-color 0.15s ease;
}
.movie-card:hover { border-color: var(--accent); }
.movie-rank { color: var(--gold); font-family: 'Bebas Neue'; font-size: 1.3rem; margin-right: 0.5rem; }
.movie-title { font-size: 1.05rem; font-weight: 600; color: var(--text); }
.movie-meta { color: var(--text-dim); font-size: 0.82rem; margin-top: 0.15rem; }
.movie-reason {
  color: var(--accent-soft); font-size: 0.8rem; margin-top: 0.35rem;
  border-left: 2px solid var(--accent); padding-left: 0.55rem;
}
.movie-score {
  float: right; background: #1F2026; color: var(--gold);
  padding: 0.15rem 0.55rem; border-radius: 20px; font-size: 0.78rem;
}

.section-label {
  color: var(--gold); text-transform: uppercase; font-size: 0.78rem;
  letter-spacing: 0.12em; font-weight: 600; margin-bottom: 0.4rem;
}

.metric-box {
  background: var(--bg-panel); border: 1px solid #232429; border-radius: 6px;
  padding: 0.8rem; text-align: center;
}
.metric-box .val { font-family: 'Bebas Neue'; font-size: 1.8rem; color: var(--gold); }
.metric-box .lbl { color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; }

[data-testid="stSidebar"] { background: #0F1013; border-right: 1px solid #232429; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- caching ---
@st.cache_resource
def load_models():
    movies_df = load_movies()
    ratings_df = load_ratings()
    users_df = load_users()
    content_model = ContentBasedRecommender(movies_df)
    collab_model = CollaborativeRecommender(ratings_df, movies_df)
    hybrid_model = HybridRecommender(content_model, collab_model, movies_df, ratings_df)
    return movies_df, ratings_df, users_df, content_model, collab_model, hybrid_model


movies_df, ratings_df, users_df, content_model, collab_model, hybrid_model = load_models()


def render_movie_card(rank, rec, show_score=True):
    score_html = f"<span class='movie-score'>{rec['score']:.2f}</span>" if show_score else ""
    st.markdown(f"""
    <div class="movie-card">
      {score_html}
      <span class="movie-rank">{rank:02d}</span>
      <span class="movie-title">{rec['title']}</span>
      <div class="movie-meta">{rec['year']} • {rec['genres'].replace('|', ' / ')}</div>
      <div class="movie-reason">Why: {rec['reason']}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------- hero -----
st.markdown("""
<div class="cine-hero">
  <h1>CINE<span>MATCH</span></h1>
  <p>A hybrid recommendation engine — content-based + collaborative filtering, blended and explained.</p>
</div>
<div class="sprocket-divider">""" + "".join(["<span></span>"] * 40) + "</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎞️ Navigate")
    mode = st.radio(
        "Mode",
        ["Because you watched... (Content-based)", "For you (Collaborative)", "Hybrid recommendations", "System evaluation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Catalogue stats")
    st.markdown(f"""
    <div style="color:#9A9AA2; font-size:0.85rem; line-height:1.9;">
    🎬 {len(movies_df)} movies<br>
    👤 {ratings_df['user_id'].nunique()} synthetic users<br>
    ⭐ {len(ratings_df):,} ratings
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built for the Technical Assignment — Recommendation System.")

# ------------------------------------------------------ mode: content -----
if mode.startswith("Because"):
    st.markdown("<div class='section-label'>Content-Based Filtering</div>", unsafe_allow_html=True)
    st.write("Pick a movie you like. We find others with overlapping genres, director, and plot themes — using TF-IDF + cosine similarity.")

    col1, col2 = st.columns([3, 1])
    with col1:
        title = st.selectbox("Movie", sorted(movies_df["title"].tolist()), index=None, placeholder="Search a movie title…")
    with col2:
        top_n = st.slider("Results", 5, 20, 10)

    if title:
        recs, err = content_model.recommend(title, top_n=top_n)
        if err:
            st.error(err)
        else:
            st.markdown(f"<div class='section-label'>Because you watched '{title}'</div>", unsafe_allow_html=True)
            for i, r in enumerate(recs, 1):
                render_movie_card(i, r)
    else:
        st.info("👆 Select a movie to see similar titles and the reasoning behind each match.")

# --------------------------------------------------- mode: collaborative --
elif mode.startswith("For you"):
    st.markdown("<div class='section-label'>Collaborative Filtering</div>", unsafe_allow_html=True)
    st.write("Pick a synthetic user profile. We predict ratings via matrix factorization (SVD) on the full user–item ratings matrix, learning from users with similar taste patterns.")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.selectbox("User ID", ratings_df["user_id"].unique().tolist(), index=0)
    with col2:
        top_n = st.slider("Results", 5, 20, 10, key="collab_n")

    with st.expander(f"📜 What user {user_id} has already rated highly"):
        history = collab_model.user_rated_titles(user_id).head(10)
        st.dataframe(history, use_container_width=True, hide_index=True)

    recs, err = collab_model.recommend(user_id, top_n=top_n)
    if err:
        st.error(err)
    else:
        st.markdown(f"<div class='section-label'>Predicted for user {user_id}</div>", unsafe_allow_html=True)
        for i, r in enumerate(recs, 1):
            render_movie_card(i, r)

# --------------------------------------------------------- mode: hybrid ---
elif mode.startswith("Hybrid"):
    st.markdown("<div class='section-label'>Hybrid: Content + Collaborative, Blended</div>", unsafe_allow_html=True)
    st.write("Combines both signals with a weighted blend. New/sparse users automatically lean more on content similarity (cold-start handling); active users lean more on collaborative patterns.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_id = st.selectbox("User ID", ratings_df["user_id"].unique().tolist(), index=0, key="hybrid_user")
    with col2:
        top_n = st.slider("Results", 5, 20, 10, key="hybrid_n")
    with col3:
        manual_alpha = st.checkbox("Manual weight", value=False)

    alpha = None
    if manual_alpha:
        alpha = st.slider("Collaborative weight (α)", 0.0, 1.0, 0.5, 0.05)

    recs, err, used_alpha = hybrid_model.recommend_for_user(user_id, top_n=top_n, alpha=alpha)
    if err:
        st.error(err)
    else:
        n_ratings = len(ratings_df[ratings_df["user_id"] == user_id])
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-bottom:1rem;">
          <div class="metric-box" style="flex:1"><div class="val">{n_ratings}</div><div class="lbl">Ratings by user</div></div>
          <div class="metric-box" style="flex:1"><div class="val">{used_alpha:.2f}</div><div class="lbl">Collaborative weight (α)</div></div>
          <div class="metric-box" style="flex:1"><div class="val">{1-used_alpha:.2f}</div><div class="lbl">Content weight (1-α)</div></div>
        </div>
        """, unsafe_allow_html=True)
        for i, r in enumerate(recs, 1):
            render_movie_card(i, r)

# --------------------------------------------------------- mode: eval -----
else:
    st.markdown("<div class='section-label'>System Evaluation</div>", unsafe_allow_html=True)
    st.write("Offline metrics computed on a held-out split (see `src/evaluate.py`). Click to re-run live — takes a few seconds.")

    if st.button("▶ Run evaluation now"):
        from evaluate import evaluate_content, evaluate_collaborative
        with st.spinner("Evaluating content-based model…"):
            sample = movies_df["title"].sample(20, random_state=1).tolist()
            content_eval = evaluate_content(content_model, movies_df, sample, k=10)
        with st.spinner("Evaluating collaborative model on held-out split…"):
            collab_eval = evaluate_collaborative(CollaborativeRecommender, ratings_df, movies_df, k=10)

        st.markdown("<div class='section-label'>Content-Based (genre-overlap sanity check)</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-box'><div class='val'>{content_eval['n_titles_evaluated']}</div><div class='lbl'>Titles tested</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><div class='val'>{content_eval['avg_genre_overlap_rate']:.0%}</div><div class='lbl'>Avg genre overlap</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><div class='val'>{content_eval['avg_latency_ms']:.1f}ms</div><div class='lbl'>Avg latency</div></div>", unsafe_allow_html=True)

        st.markdown("<br><div class='section-label'>Collaborative Filtering (held-out Precision/Recall/NDCG@10)</div>", unsafe_allow_html=True)
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.markdown(f"<div class='metric-box'><div class='val'>{collab_eval['n_users_evaluated']}</div><div class='lbl'>Users evaluated</div></div>", unsafe_allow_html=True)
        d2.markdown(f"<div class='metric-box'><div class='val'>{collab_eval['precision@10']:.2f}</div><div class='lbl'>Precision@10</div></div>", unsafe_allow_html=True)
        d3.markdown(f"<div class='metric-box'><div class='val'>{collab_eval['recall@10']:.2f}</div><div class='lbl'>Recall@10</div></div>", unsafe_allow_html=True)
        d4.markdown(f"<div class='metric-box'><div class='val'>{collab_eval['ndcg@10']:.2f}</div><div class='lbl'>NDCG@10</div></div>", unsafe_allow_html=True)
        d5.markdown(f"<div class='metric-box'><div class='val'>{collab_eval['coverage']:.0%}</div><div class='lbl'>Catalogue coverage</div></div>", unsafe_allow_html=True)
    else:
        st.info("Click the button above to compute live metrics. Full methodology is documented in README.md → Evaluation Methodology.")
