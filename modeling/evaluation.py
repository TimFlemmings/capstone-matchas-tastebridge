"""Judging a feature space by what the recommender needs from it.

Explained variance says how much of the original signal survived compression.
It says nothing about whether two tracks sitting near each other actually sound
alike — which is the only thing a content-based recommender runs on. The metric
here is a downstream proxy for that: of a track's k nearest neighbours, how many
share its top-level genre.

Run `python -m modeling.evaluation` from the repo root to reproduce the audit
table behind the team memo.
"""

from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logger = getLogger(__name__)

K_NEIGHBOURS = 10


def neighbour_genre_agreement(Z, labels: pd.Series, k: int = K_NEIGHBOURS,
                              metric: str = "cosine") -> float:
    """Fraction of each track's k nearest neighbours sharing its label.

    Z: the embedding to judge (n_tracks x n_dims) — scaled features, PCA space,
       or anything else with the same row order as `labels`.

    With 8 balanced genres, random guessing scores 0.125. Higher is better.
    The metric is a proxy: two tracks can be similar and carry different genre
    labels, so use it to compare configurations, not to claim the recommender
    is good.

    `metric` defaults to cosine to match TrackRecommender. It used to be left
    unset, which silently gave sklearn's euclidean default — so this function
    scored a different pipeline from the one that ships, and train_pca.py logged
    those numbers. It mattered: on the chroma experiments the two metrics rank
    the configurations in opposite orders.

        config                        euclidean   cosine
        baseline (518 -> 146)             0.378    0.413
        without 2 chroma families         0.394    0.429
        without chroma + kurt/skew        0.397    0.423

    Under cosine, dropping chroma alone wins. Under euclidean, dropping the
    shape statistics as well wins. Same data, opposite conclusion.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]  # a track is its own nearest neighbour — drop it
    # arrow-backed dtypes (pandas reads FMA metadata as one) reject 2-D take
    lab = np.asarray(labels.astype(str))
    return float((lab[idx] == lab[:, None]).mean())


def random_baseline(labels: pd.Series) -> float:
    """What neighbour_genre_agreement would score by chance on balanced classes."""
    return 1.0 / labels.nunique()


def genre_sets(tracks: pd.DataFrame) -> pd.Series:
    """Every genre id attached to each track, not just the top-level one.

    FMA tags a track with a hierarchy — 2.2 genres on average, up to 8 — stored
    as a stringified list in ("track", "genres_all").
    """
    import ast

    raw = tracks[("track", "genres_all")]
    return raw.apply(lambda v: frozenset(ast.literal_eval(v)) if isinstance(v, str) else frozenset())


def genre_overlap_at_k(embedding, genre_sets_: pd.Series, k: int = K_NEIGHBOURS,
                       metric: str = "cosine") -> float:
    """Average Jaccard overlap between a track's genres and its neighbours'.

    Exact genre_top matching scores a recommendation 0 or 1, which punishes
    exactly the cross-genre jumps an "adventurous" mode is supposed to make.
    Overlap grades them instead: a hip-hop track landing on an electronic track
    that shares two sub-genres scores well above one that shares none.
    """
    from sklearn.neighbors import NearestNeighbors

    sets = list(genre_sets_)
    nn = NearestNeighbors(metric=metric).fit(embedding)
    _, idx = nn.kneighbors(embedding, n_neighbors=k + 1)
    idx = idx[:, 1:]

    scores = np.empty(idx.shape)
    for i, row in enumerate(idx):
        own = sets[i]
        for j, other in enumerate(row):
            union = own | sets[other]
            scores[i, j] = len(own & sets[other]) / len(union) if union else 0.0
    return float(scores.mean())


def ndcg_at_k(embedding, labels: pd.Series, k: int = K_NEIGHBOURS,
              metric: str = "cosine") -> float:
    """Like precision@k, but a correct recommendation counts for more near the top.

    precision@k treats position 1 and position 10 alike. A user reads from the
    top, so ranking matters: this weights position p by 1/log2(p+1) and divides
    by the best achievable score, landing between 0 and 1.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(metric=metric).fit(embedding)
    _, idx = nn.kneighbors(embedding, n_neighbors=k + 1)
    idx = idx[:, 1:]

    lab = np.asarray(labels.astype(str))
    hits = lab[idx] == lab[:, None]

    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    return float((dcg / discounts.sum()).mean())


def catalogue_coverage(embedding, k: int = K_NEIGHBOURS, metric: str = "cosine") -> float:
    """Share of the catalogue that ever appears in anyone's recommendations.

    A recommender can score well on precision while only ever surfacing a small
    popular core. This is the check that catches that failure.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(metric=metric).fit(embedding)
    _, idx = nn.kneighbors(embedding, n_neighbors=k + 1)
    return float(np.unique(idx[:, 1:]).size / embedding.shape[0])


def _intra_list(values, neighbours, metric: str = "cosine") -> float:
    """Mean distance between the recommended tracks *themselves*, not to the seed.

    Every genre metric judges each recommendation against the seed. None of them
    notices when all ten recommendations are near-copies of one another — ten
    tracks by the same artist score a perfect precision@k and are a bad list.

    catalogue_coverage does not catch it either: that is a system-level count
    across all seeds, so a recommender can surface most of the catalogue while
    handing every individual listener ten interchangeable tracks.

    This is the per-listener version of that question, and the one an adventure
    control has to move.
    """
    from sklearn.metrics import pairwise_distances

    k = neighbours.shape[1]
    # upper triangle only — the k x k distance matrix is symmetric with a zero
    # diagonal, so a full mean would be diluted by both
    triu = np.triu_indices(k, k=1)
    scores = [pairwise_distances(values[row], metric=metric)[triu].mean()
              for row in neighbours]
    return float(np.mean(scores))


def depth_profile(embedding, labels: pd.Series, genre_sets_: pd.Series = None,
                  depths=(0, 3, 21, 105, 499), k: int = K_NEIGHBOURS,
                  metric: str = "cosine", sample: int = 1000, seed: int = 42) -> pd.DataFrame:
    """How relevance decays as the recommendation window slides down the ranking.

    The Familiar → Adventurous control reads k tracks starting at some depth in
    the neighbour list rather than always from the top. This measures what that
    costs: one row per depth, with the same rulers the baseline is scored on.

    Read it as a shape, not as a score. precision@k *should* fall as the window
    moves out — that is the feature working. What matters is where it falls off
    a cliff, and whether genre_overlap holds up better than exact matching,
    which would mean the far recommendations are still musically related rather
    than merely distant.
    """
    from sklearn.neighbors import NearestNeighbors

    lab = np.asarray(labels.astype(str))
    sets = list(genre_sets_) if genre_sets_ is not None else None
    n_tracks = embedding.shape[0]

    rows = np.arange(n_tracks)
    if sample is not None and sample < n_tracks:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=sample, replace=False))

    # one query deep enough for every depth, then slice it — the expensive part
    # is the search, and it does not need repeating per window
    deepest = max(depths) + k + 1
    index = NearestNeighbors(metric=metric).fit(embedding)
    Z = embedding.iloc[rows] if hasattr(embedding, "iloc") else embedding[rows]
    distances, idx = index.kneighbors(Z, n_neighbors=deepest)
    distances, idx = distances[:, 1:], idx[:, 1:]  # drop each seed itself
    values = embedding.values if hasattr(embedding, "values") else embedding

    out = []
    for depth in depths:
        window = slice(depth, depth + k)
        neighbours = idx[:, window]
        hits = lab[neighbours] == lab[rows][:, None]

        row = {
            "adventure": round(np.log(depth + 1) / np.log(max(depths) + 1), 2),
            "depth": depth,
            # novelty: how far the window sits from the seed. No genre anywhere
            # in it, and trivially gamed on its own — read it against precision.
            "novelty": float(distances[:, window].mean()),
            "precision_at_k": float(hits.mean()),
            # coverage is a *system*-level count: what the catalogue sees across
            # every seed. It cannot tell that one listener got ten near-copies.
            "coverage": float(np.unique(neighbours).size / n_tracks),
            # intra-list diversity is the per-listener version of that question.
            "intra_list_diversity": _intra_list(values, neighbours, metric),
        }
        if sets is not None:
            overlaps = np.empty(neighbours.shape)
            for i, nb_row in enumerate(neighbours):
                own = sets[rows[i]]
                for j, other in enumerate(nb_row):
                    union = own | sets[other]
                    overlaps[i, j] = len(own & sets[other]) / len(union) if union else 0.0
            row["genre_overlap_at_k"] = float(overlaps.mean())
        out.append(row)

    return pd.DataFrame(out)


def feature_family_summary(X: pd.DataFrame) -> pd.Series:
    """Column count per feature family (mfcc, chroma_cens, zcr, ...).

    PCA picks directions by variance, and a concept's variance grows with the
    number of columns representing it — so a lopsided count is a lopsided
    similarity space.
    """
    return pd.Series([c[0] for c in X.columns]).value_counts()


def statistic_summary(X: pd.DataFrame) -> pd.Series:
    """Column count per summary statistic (mean, std, kurtosis, skew, ...)."""
    return pd.Series([c[1] for c in X.columns]).value_counts()


def tail_summary(X: pd.DataFrame, X_scaled: pd.DataFrame = None) -> dict:
    """How far the features depart from the symmetric spread StandardScaler assumes."""
    kurt = X.kurtosis()
    out = {
        "n_columns": int(X.shape[1]),
        "columns_kurtosis_over_100": int((kurt > 100).sum()),
        "max_kurtosis": float(kurt.max()),
    }
    if X_scaled is not None:
        out["max_abs_z"] = float(np.abs(X_scaled.values).max())
    return out


def evaluate_pca_grid(X_scaled: pd.DataFrame, labels: pd.Series, variances=(0.90, 0.95, 0.99),
                      fixed_dims=(2, 10, 30, 60), k: int = K_NEIGHBOURS) -> pd.DataFrame:
    """Neighbourhood quality across PCA settings, plus the no-PCA reference.

    Returns one row per configuration with dims, variance kept and quality —
    the table the team memo is built on.
    """
    rows = [{
        "config": "scaled, no PCA",
        "dims": int(X_scaled.shape[1]),
        "variance_kept": 1.0,
        "quality": neighbour_genre_agreement(X_scaled.values, labels, k),
    }]

    for variance in variances:
        pca = PCA(n_components=variance).fit(X_scaled)
        rows.append({
            "config": f"PCA {variance:.0%}",
            "dims": int(pca.n_components_),
            "variance_kept": float(pca.explained_variance_ratio_.sum()),
            "quality": neighbour_genre_agreement(pca.transform(X_scaled), labels, k),
        })

    for n in fixed_dims:
        pca = PCA(n_components=n).fit(X_scaled)
        rows.append({
            "config": f"PCA, fixed {n}",
            "dims": n,
            "variance_kept": float(pca.explained_variance_ratio_.sum()),
            "quality": neighbour_genre_agreement(pca.transform(X_scaled), labels, k),
        })

    return pd.DataFrame(rows).sort_values("dims", ascending=False).reset_index(drop=True)


def cross_validated_precision(X: pd.DataFrame, labels: pd.Series, n_components=0.90,
                              metric: str = "cosine", k: int = K_NEIGHBOURS,
                              n_splits: int = 5, seed: int = 42,
                              genre_sets_: pd.Series = None) -> pd.DataFrame:
    """Precision@k measured on held-out tracks, fold by fold.

    Everything above this function fits the scaler and PCA on the whole
    catalogue and then scores the same tracks. That is defensible for a fixed
    catalogue you already hold, but it cannot answer "does this work for a track
    the pipeline has never seen". This can:

      - fit scaler and PCA on the training fold only
      - index the training tracks as the catalogue
      - project each held-out track into that space and retrieve from it

    A held-out score far below the in-sample one would mean the feature space
    was shaped around the tracks it was scored on.
    """
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors

    from modeling.features import fit_pca, scale_features

    lab = np.asarray(labels.astype(str))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(folds.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        X_train_scaled, scaler = scale_features(X_train)
        pca = fit_pca(X_train_scaled, n_components=n_components)

        Z_train = pca.transform(X_train_scaled)
        Z_test = pca.transform(scaler.transform(X_test))

        # held-out tracks are not in the catalogue, so no self-match to drop
        index = NearestNeighbors(metric=metric).fit(Z_train)
        _, neighbours = index.kneighbors(Z_test, n_neighbors=k)

        hits = lab[train_idx][neighbours] == lab[test_idx][:, None]

        # the same four rulers as in-sample, so the two protocols are comparable
        discounts = 1 / np.log2(np.arange(2, k + 2))
        row = {
            "fold": fold,
            "train_tracks": len(train_idx),
            "test_tracks": len(test_idx),
            "components": int(pca.n_components_),
            "precision_at_k": float(hits.mean()),
            "ndcg_at_k": float(((hits * discounts).sum(axis=1) / discounts.sum()).mean()),
            # coverage here means: how much of the training catalogue the held-out
            # queries manage to surface
            "coverage": float(np.unique(neighbours).size / len(train_idx)),
        }

        if genre_sets_ is not None:
            train_sets = [genre_sets_.iloc[i] for i in train_idx]
            test_sets = [genre_sets_.iloc[i] for i in test_idx]
            overlaps = np.empty(neighbours.shape)
            for i, nb_row in enumerate(neighbours):
                own = test_sets[i]
                for j, other in enumerate(nb_row):
                    union = own | train_sets[other]
                    overlaps[i, j] = len(own & train_sets[other]) / len(union) if union else 0.0
            row["genre_overlap_at_k"] = float(overlaps.mean())

        rows.append(row)
        logger.info("fold %s: precision@%s = %.3f", fold, k, row["precision_at_k"])

    return pd.DataFrame(rows)


def _main():
    import logging

    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

    from modeling.data import load_fma, genre_labels
    from modeling.features import scale_features

    X, tracks = load_fma(subset="small")
    labels = genre_labels(tracks)
    X_scaled, _ = scale_features(X)

    print(f"\ntracks: {X.shape[0]} | features: {X.shape[1]} | genres: {labels.nunique()}")

    print("\n=== columns per feature family ===")
    print(feature_family_summary(X).to_string())

    print("\n=== columns per statistic ===")
    print(statistic_summary(X).to_string())

    print("\n=== tail heaviness ===")
    for key, value in tail_summary(X, X_scaled).items():
        print(f"{key}: {value:.1f}" if isinstance(value, float) else f"{key}: {value}")

    print(f"\n=== neighbourhood quality (k={K_NEIGHBOURS} genre agreement) ===")
    print(f"random baseline: {random_baseline(labels):.3f}\n")
    print(evaluate_pca_grid(X_scaled, labels).to_string(index=False,
                                                        float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    _main()
