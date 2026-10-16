"""Reading a PCA space back in terms of the original audio features.

PCA answers "how many dimensions do I need". It does not answer the two
questions the mentors asked of it:

  1. Which of the 518 original features is the reduced space actually built on?
     A component is a mixture of all of them, so there is no feature list to
     read off — but there is a contribution share, and it can be computed.

  2. Is projecting 518 -> 146 the same as *dropping* 372 features? If the
     information PCA compresses is the information selection would throw away,
     the two should score alike. If they don't, one of them is losing something
     the recommender needs.

Question 2 is answered by the inverse experiment: rank the features by how much
of the retained variance they carry, build the space on the top 146, build it
again on the 372 PCA leans on least, and score all of them with the ruler the
recommender runs on.

Both answers came back negative, which is the useful kind:

  1. There is no list. At 146 components the contribution is flat — every one
     of the 518 features lands within a few percent of 1/518, and every feature
     family within 5% of its column share. 90% of the variance is simply spread
     across everything, so a "most important features" reading of this PCA is
     reading noise.

  2. Projecting and dropping are not interchangeable, and projecting wins:

         config                        dims   quality   vs random
         all features, no PCA           518     0.414       3.31x
         PCA -> 146                     146     0.413       3.30x
         top 146 features               146     0.345       2.76x
         random 146 features            146     0.384       3.07x
         bottom 372 features            372     0.414       3.31x
         bottom features -> PCA 146     146     0.411       3.29x

     Compressing 518 -> 146 costs 0.001. Selecting 146 costs 0.069 — and is
     beaten by picking 146 columns at random, because the highest-contributing
     features are highest-contributing together: they are near-copies of one
     another (chroma means, mostly), so keeping them covers fewer musical
     concepts than a random draw across the families does.

     The inverse set settles the rest. Throw away every feature PCA leans on
     most and the remaining 372 still score 0.414. The signal is not in a
     privileged subset of the features; it is spread across all of them, which
     is exactly the condition under which PCA is the right tool and feature
     selection is the wrong one.

    uv run python -m modeling.interpretation
"""

from logging import getLogger

import numpy as np
import pandas as pd

from modeling.evaluation import K_NEIGHBOURS, neighbour_genre_agreement, random_baseline
from modeling.features import fit_pca, scale_features

logger = getLogger(__name__)


def feature_contributions(pca, columns=None) -> pd.Series:
    """Share of the retained variance attributable to each original feature.

    Each component is a unit vector over the original features, so squaring its
    loadings gives that component's budget split across them. Weighting those
    splits by how much variance each component itself explains, and summing over
    the components PCA kept, gives one number per original feature: how much of
    what survived compression came from it. The shares sum to 1.

    This is the closest thing PCA offers to a feature-importance list. It is not
    one: a feature with a small share can still be the only carrier of something
    the listener hears. Treat it as "what the space is built on", not "what
    matters".

    On the 146-component baseline the answer is "all of it, evenly": the shares
    run from 0.0016 to 0.0023 against a flat 1/518 = 0.0019. Sorting them is
    still worth doing — `selection_vs_projection` needs the ranking — but the
    head of the list should not be read as the features the recommender depends
    on. Keeping only that head scores *worse* than a random draw.
    """
    # components_ is (n_components, n_features); rows are unit norm
    weighted = pca.explained_variance_ratio_[:, None] * (pca.components_ ** 2)
    share = weighted.sum(axis=0)
    share = share / share.sum()
    return pd.Series(share, index=columns if columns is not None else None).sort_values(
        ascending=False
    )


def family_contributions(contributions: pd.Series) -> pd.DataFrame:
    """Contribution per feature family, against how many columns it brought.

    The point of the comparison is the last column. chroma_* holds 252 of 518
    columns, and PCA picks directions by variance — so a family with half the
    columns will tend to dominate the space whether or not it carries half the
    musical signal. `weight_ratio` divides a family's contribution share by its
    column share:

        > 1  the family punches above its column count
        ~ 1  it contributes in proportion to how often it was counted
        < 1  it is padding — many columns, little of the retained variance

    A family sitting well below 1 with a large column count is the case for
    dropping it: it is buying dimensions at the cost of crowding out others.

    Measured on the 146-component baseline, no family is anywhere near either
    extreme — the ratios run from 0.97 (spectral_bandwidth) to 1.05 (spectral
    _centroid), and each chroma family sits at 1.00 with 84 columns each. So
    chroma's 49% of the space is bought entirely with column count, not with
    signal: the three families are not more informative per column than
    anything else, there are simply three of them measuring harmony while one
    family measures loudness. That is the argument for thinning chroma, and it
    is a fairness argument about representation rather than a claim that
    harmony matters less.
    """
    families = contributions.index.get_level_values(0)
    n_columns = pd.Series(families).value_counts()
    contribution = contributions.groupby(families).sum()

    out = pd.DataFrame({
        "n_columns": n_columns,
        "column_share": n_columns / n_columns.sum(),
        "contribution": contribution,
    })
    out["per_column"] = out["contribution"] / out["n_columns"]
    out["weight_ratio"] = out["contribution"] / out["column_share"]
    return out.sort_values("contribution", ascending=False)


def within_family(contributions: pd.Series, family: str) -> pd.DataFrame:
    """The same breakdown one level down, inside a single family.

    Asked for directly: chroma takes 49% of the columns, but is that spread
    evenly across its members or are a few of them carrying it? If a handful
    dominate, the family can be thinned instead of dropped whole — which is the
    difference between losing harmony from the similarity space and merely
    stopping it from drowning out everything else.

    Inside chroma_cens the seven statistics land between 13.3% and 15.5% of the
    family against a flat 14.3%, so nothing dominates there either. Thinning
    chroma therefore has to drop whole families (--drop-chroma keeps cens only)
    rather than cherry-pick statistics within one; there is no cheap subset to
    keep.
    """
    rows = contributions[contributions.index.get_level_values(0) == family]
    statistics = rows.index.get_level_values(1)

    out = pd.DataFrame({
        "n_columns": pd.Series(statistics).value_counts(),
        "contribution": rows.groupby(statistics).sum(),
    })
    out["share_of_family"] = out["contribution"] / out["contribution"].sum()
    return out.sort_values("contribution", ascending=False)


def selection_vs_projection(X: pd.DataFrame, labels: pd.Series, n_components=0.90,
                            k: int = K_NEIGHBOURS, metric: str = "cosine",
                            seed: int = 42) -> pd.DataFrame:
    """Score dropping features against projecting them, at equal dimensionality.

    Five configurations, one ruler (neighbour genre agreement, cosine, the
    distance the recommender ships with):

      all features        the uncompressed reference — Tim's baseline
      PCA                 the compressed space this branch runs on
      top-d features      the d features PCA leans on most, kept raw. This is
                          feature *selection* at the same width as the PCA space
      random-d features   the control. Without it, a good top-d score could just
                          mean "d columns is enough", nothing about the ranking
      bottom features     everything PCA leans on least — the inverse set. If
                          this scores near chance, the ranking found the signal;
                          if it holds up, the discarded tail carries music too
                          and compressing beats dropping

    Read the gap between `PCA` and `top-d features`: that is the price (or the
    gain) of projecting rather than selecting, measured instead of argued. On
    the baseline it is 0.413 against 0.345 — see the module docstring for the
    full table and what the random and inverse controls make of it.

    One number here is worth carrying into the team comparison on its own: the
    uncompressed 518-feature space scores 3.31x random and the 146-component
    space scores 3.30x. Two baselines built on those two spaces are measuring
    the same thing to within a rounding error, so a difference between them is
    a difference in protocol — k, distance, sample, label — not in features.
    """
    X_scaled, _ = scale_features(X)
    pca = fit_pca(X_scaled, n_components=n_components)
    dims = int(pca.n_components_)

    contributions = feature_contributions(pca, X.columns)
    top = list(contributions.index[:dims])
    bottom = list(contributions.index[dims:])
    random_columns = list(
        X.columns[np.random.default_rng(seed).choice(X.shape[1], size=dims, replace=False)]
    )

    def score(Z) -> float:
        return neighbour_genre_agreement(Z, labels, k=k, metric=metric)

    rows = [
        {"config": "all features, no PCA", "columns_in": int(X.shape[1]), "dims": int(X.shape[1]),
         "quality": score(X_scaled.values)},
        {"config": f"PCA -> {dims}", "columns_in": int(X.shape[1]), "dims": dims,
         "quality": score(pca.transform(X_scaled))},
        {"config": f"top {dims} features", "columns_in": dims, "dims": dims,
         "quality": score(X_scaled[top].values)},
        {"config": f"random {dims} features", "columns_in": dims, "dims": dims,
         "quality": score(X_scaled[random_columns].values)},
        {"config": f"bottom {len(bottom)} features", "columns_in": len(bottom),
         "dims": len(bottom), "quality": score(X_scaled[bottom].values)},
    ]

    # the inverse set compressed to the same width, so the last comparison is
    # not confounded by it being a wider space than everything above
    inverse_pca = fit_pca(X_scaled[bottom], n_components=dims)
    rows.append({
        "config": f"bottom features -> PCA {dims}",
        "columns_in": len(bottom),
        "dims": dims,
        "quality": score(inverse_pca.transform(X_scaled[bottom])),
    })

    out = pd.DataFrame(rows)
    out["vs_random"] = out["quality"] / random_baseline(labels)
    return out


def _main():
    import logging

    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

    from modeling.data import genre_labels, load_fma

    X, tracks = load_fma(subset="small")
    labels = genre_labels(tracks)

    X_scaled, _ = scale_features(X)
    pca = fit_pca(X_scaled, n_components=0.90)
    contributions = feature_contributions(pca, X.columns)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731

    print(f"\n=== what the {pca.n_components_}-component space is built on ===")
    print(family_contributions(contributions).to_string(float_format=fmt))

    print("\n=== inside chroma_cens ===")
    print(within_family(contributions, "chroma_cens").to_string(float_format=fmt))

    print("\n=== the 15 features carrying most of the retained variance ===")
    print(contributions.head(15).to_string(float_format=fmt))

    print(f"\n=== dropping vs projecting (k={K_NEIGHBOURS}, cosine) ===")
    print(f"random baseline: {random_baseline(labels):.3f}\n")
    print(selection_vs_projection(X, labels).to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    _main()
