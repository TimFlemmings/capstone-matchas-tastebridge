"""Score any embedding against what listeners actually said.

The project's whole argument is that genre agreement is the wrong ruler. This is
the right one, and it is cheap: for every rated pair, take the distance an
embedding puts between the two tracks, and correlate those distances against the
1-5 scores six people gave them blind. Negative is the good direction — a short
distance should come with a high score.

It takes any matrix with one row per track id, so an embedding from outside this
repository is scored on exactly the same footing as one of ours:

    uv run python -m modeling.embedding_vs_ear
    uv run python -m modeling.embedding_vs_ear --npz ../tim-effnet/data/effnet_embeddings_full.npz

The subsets matter as much as the headline. A space can look good only because
random pairs sit far away and score low, while saying nothing about which of the
model's own picks is better — which is the thing a listener actually sees. The
`only the top five` row is where that shows up, and it is the row to read first.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_npz(path: Path) -> pd.DataFrame:
    """An `ids` + `embeddings` archive as a track-indexed frame."""
    z = np.load(path)
    missing = [k for k in ("ids", "embeddings") if k not in z]
    if missing:
        raise KeyError(f"{path} has {list(z)}, expected 'ids' and 'embeddings'")
    return pd.DataFrame(z["embeddings"], index=z["ids"])


def scored_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: the mean human score, both track ids, and its kind."""
    return d.dropna(subset=["similarity_1_5"]).groupby("pair_id").agg(
        human=("similarity_1_5", "mean"), votes=("similarity_1_5", "size"),
        a=("track_a", "first"), b=("track_b", "first"), kind=("kind", "first"))


def distances(Z: pd.DataFrame, pairs: pd.DataFrame) -> list:
    return [cosine(Z.loc[a], Z.loc[b]) for a, b in zip(pairs.a, pairs.b)]


def compare(spaces: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    """Each space's correlation with the ear, over the whole set and its parts."""
    for name, Z in spaces.items():
        pairs = pairs.assign(**{name: distances(Z, pairs)})

    subsets = {
        "every pair": pairs,
        "without the random pairs": pairs[pairs.kind != "random"],
        "only the top five": pairs[pairs.kind.isin(["model", "heard_top", "all_top"])],
    }
    rows = []
    for label, sub in subsets.items():
        row = {"subset": label, "pairs": len(sub)}
        for name in spaces:
            if len(sub) < 8:
                row[name] = np.nan
                continue
            rho, p = spearmanr(sub[name], sub.human)
            row[name] = round(rho, 3)
            row[f"{name} p"] = round(p, 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("subset")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", type=Path, action="append", default=None,
                        metavar="PATH", help="an ids+embeddings archive to score; "
                                             "repeat for several")
    parser.add_argument("--ours", nargs="*", default=["all", "heard"],
                        help="named spaces from modeling.features to score beside them")
    args = parser.parse_args(argv)

    from modeling.data import load_fma
    from modeling.features import space
    from modeling.listening import responses

    pairs = scored_pairs(responses())
    logger.info("%s rated pairs", len(pairs))

    spaces = {}
    if args.ours:
        X, _ = load_fma("small")
        for name in args.ours:
            S = space(name, X)
            spaces[name] = pd.DataFrame(StandardScaler().fit_transform(S), index=S.index)
    for path in args.npz or []:
        E = load_npz(path)
        logger.info("%s: %s tracks x %s dims", path.name, *E.shape)
        spaces[path.stem[:18]] = pd.DataFrame(
            StandardScaler().fit_transform(E), index=E.index)

    # A track with no embedding would silently drop out of one space and not the
    # others, which would make the columns incomparable. Drop it from all of them.
    covered = set.intersection(*(set(Z.index) for Z in spaces.values()))
    short = pairs[~(pairs.a.isin(covered) & pairs.b.isin(covered))]
    if len(short):
        logger.warning("%s pairs dropped — not every space covers both tracks", len(short))
        pairs = pairs.drop(short.index)

    table = compare(spaces, pairs)
    print("\nSpearman between distance and the 1-5 human score.")
    print("Negative is good: a short distance should come with a high score.\n")
    print(table.to_string())
    print("\nRead `only the top five` first. A space can score well overall merely "
          "\nbecause random pairs are far away and rated low, and still be unable to "
          "\ntell which of its own recommendations a listener prefers.")


if __name__ == "__main__":
    main()
