"""Taste representation from several seed tracks, and a way to compare methods.

Two ways to turn a set of seeds into one score per catalogue track:

    centroid : similarity to the average of the seeds
    maxsim   : similarity to the closest seed

Compared on simulated users, because FMA has no listening histories:

    coherent : n seeds from one genre
    mixed    : n/2 seeds from genre A + n/2 from genre B

Usage from a notebook (after building `rec` on the "heard" space):

    from modeling.taste import from_recommender, compare
    E, genres, artists, ids = from_recommender(rec)
    summary, per_user = compare(E, genres, artists)
    summary
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- building blocks

def unit(V):
    """Scale rows to length 1, so cosine similarity is a plain dot product."""
    return V / np.linalg.norm(V, axis=-1, keepdims=True)


def from_recommender(rec):
    """Pull the embedding and labels out of a fitted TrackRecommender, aligned."""
    ids = rec.embedding_.index
    E = unit(rec.embedding_.to_numpy())
    meta = rec.tracks_.loc[ids]
    genres = meta[("track", "genre_top")].astype(str).to_numpy()
    artists = meta[("artist", "name")].astype(str).to_numpy()
    return E, genres, artists, ids


def score_centroid(E, seed_rows):
    c = unit(E[seed_rows].mean(axis=0))
    return E @ c


def score_maxsim(E, seed_rows):
    return (E @ E[seed_rows].T).max(axis=1)


METHODS = {"centroid": score_centroid, "maxsim": score_maxsim}


def coherence(E, seed_rows):
    """Length of the mean seed vector: ~1 = seeds agree, ~0 = they point apart."""
    return float(np.linalg.norm(E[seed_rows].mean(axis=0)))


def recommend(scores, seed_rows, artists, k=10):
    """Top-k rows, never a seed and never an artist the seeds already contain."""
    blocked = np.isin(artists, artists[seed_rows])
    s = np.where(blocked, -np.inf, scores)
    top = np.argpartition(-s, k)[:k]
    return top[np.argsort(-s[top])]


# ---------------------------------------------------------------- simulated users

def simulate_users(genres, n_users=200, n_seeds=6, seed=42):
    """Return a list of dicts: kind, seed rows, and the genre each half came from."""
    rng = np.random.default_rng(seed)
    by_genre = {g: np.flatnonzero(genres == g) for g in np.unique(genres)}
    names = list(by_genre)
    users = []
    for _ in range(n_users):
        g = rng.choice(names)
        users.append({"kind": "coherent", "groups": {g: rng.choice(by_genre[g], n_seeds, replace=False)}})
    half = n_seeds // 2
    for _ in range(n_users):
        a, b = rng.choice(names, 2, replace=False)
        users.append({"kind": "mixed", "groups": {
            a: rng.choice(by_genre[a], half, replace=False),
            b: rng.choice(by_genre[b], half, replace=False)}})
    return users


# ---------------------------------------------------------------- measuring

def _evaluate_one(E, genres, artists, user, method, k):
    groups = user["groups"]
    seed_rows = np.concatenate(list(groups.values()))
    recs = recommend(METHODS[method](E, seed_rows), seed_rows, artists, k)

    row = {
        "kind": user["kind"],
        "method": method,
        "coherence": coherence(E, seed_rows),
        # relevance: share of recommendations in any of the seed genres
        "relevance": float(np.isin(genres[recs], list(groups)).mean()),
    }
    if len(groups) == 2:
        # balance: which seed half is each recommendation closer to?
        # 1.0 = an even split, 0.0 = one half of the taste ignored entirely
        (_, a_rows), (_, b_rows) = groups.items()
        near_a = (E[recs] @ E[a_rows].T).max(axis=1) > (E[recs] @ E[b_rows].T).max(axis=1)
        share = near_a.mean()
        row["balance"] = float(1 - abs(2 * share - 1))
    return row


def compare(E, genres, artists, n_users=200, n_seeds=6, k=10, seed=42):
    """Run every method on the same simulated users. Returns (summary, per_user)."""
    users = simulate_users(genres, n_users, n_seeds, seed)
    per_user = pd.DataFrame([
        _evaluate_one(E, genres, artists, u, m, k) for u in users for m in METHODS
    ])
    summary = (per_user.groupby(["kind", "method"])[["relevance", "balance", "coherence"]]
               .agg(["mean", "std"]).round(3))
    return summary, per_user
