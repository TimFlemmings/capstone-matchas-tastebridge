"""Reading the listening test back: from what people said to what to build next.

Raters answer two things about every pair — a 1-5 similarity score, and an
optional list of what connects the two tracks, chosen from a fixed vocabulary.
The scores say whether the model is right. The vocabulary says *where it is
blind*, and that is the half that turns into work.

The step is small enough to miss: count the words, then line them up against
the feature families the model actually has. A word people keep using with no
family behind it is a feature nobody has built yet.

Run it after every batch of returns:

    uv run python -m modeling.listening
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from modeling.rhythm import REPO_ROOT

logger = logging.getLogger(__name__)

TEST_DIR = REPO_ROOT / "data" / "listening_test"

# The rating form's vocabulary, mapped onto the feature families that could
# carry it. This is the whole translation from survey to feature engineering:
# a word with an empty right-hand side is a hole in the feature set.
VOCABULARY = {
    "timbre":          ["mfcc", "spectral_contrast", "spectral_centroid",
                        "spectral_bandwidth", "spectral_rolloff"],
    "harmony":         ["chroma_cens", "chroma_cqt", "chroma_stft", "tonnetz"],
    "energy":          ["rmse", "zcr"],
    "rhythm":          ["tempo", "onset_rate", "onset_strength", "pulse_clarity"],
    "instrumentation": ["hpss_ratio", "mfcc_harmonic", "mfcc_percussive"],
    "production":      [],
    "mood":            [],
    "vocals":          [],
    "nothing":         [],   # the control word — not a dimension, an absence
}

# Families that exist on disk today. rhythm.py and separation.py wrote the last
# two blocks; everything above them ships with FMA.
BUILT = {
    "mfcc", "spectral_contrast", "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff", "zcr", "rmse", "chroma_cens", "chroma_cqt",
    "chroma_stft", "tonnetz",
}
if (REPO_ROOT / "data" / "rhythm_features.csv").exists():
    BUILT |= {"tempo", "onset_rate", "onset_strength", "pulse_clarity"}
if (REPO_ROOT / "data" / "separation_features.csv").exists():
    BUILT |= {"hpss_ratio", "mfcc_harmonic", "mfcc_percussive"}


def responses(directory: Path = TEST_DIR / "responses") -> pd.DataFrame:
    """Every rating returned so far, joined to the answer key.

    Reads whatever is in the folder, so it works at two raters and at six.
    """
    files = sorted(directory.glob("*_ratings.csv"))
    if not files:
        raise FileNotFoundError(f"no *_ratings.csv in {directory}")
    ratings = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = pd.read_csv(TEST_DIR / "pairs_key.csv")
    logger.info("%s ratings from %s raters", len(ratings), ratings["rater"].nunique())
    return ratings.merge(key, on="pair_id")


def tags(d: pd.DataFrame) -> pd.DataFrame:
    """How often each word was used, and how the pairs carrying it scored."""
    # A rater who ticked nothing leaves an empty cell; exploding an empty list
    # yields NaN, which would otherwise arrive as a tag of its own.
    words = (d["connects"].fillna("").str.split().explode().dropna()
             .str.strip().str.lower())
    words = words[words != ""]

    rows = []
    for word in words.unique():
        carries = d["connects"].fillna("").str.contains(rf"\b{word}\b")
        rows.append({
            "tag": word,
            "mentions": int((words == word).sum()),
            "mean_score": round(float(d.loc[carries, "similarity_1_5"].mean()), 2),
        })
    return pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)


def gaps(tag_counts: pd.DataFrame) -> pd.DataFrame:
    """Each word beside the families that could carry it, and whether they exist.

    A word with mentions but no built family is the next thing to extract. That
    is the whole point of asking raters what connects a pair rather than only
    how alike it is.
    """
    rows = []
    for row in tag_counts.itertuples():
        if row.tag == "nothing":
            continue
        families = VOCABULARY.get(row.tag, [])
        missing = [f for f in families if f not in BUILT]
        rows.append({
            "tag": row.tag,
            "mentions": row.mentions,
            "mean_score": row.mean_score,
            "families": ", ".join(families) or "—",
            "status": ("NO FEATURE" if not families
                       else "not built" if missing
                       else "built"),
        })
    return pd.DataFrame(rows)


def flat_table(d: pd.DataFrame = None) -> pd.DataFrame:
    """Every rating as one row, with both tracks named.

    The ratings live in six files keyed only by pair id, and the names live in
    the answer key, which was deliberately kept away from the raters. Now that
    the test is closed there is no reason to keep them apart, and one table is
    what anyone actually wants to look at.
    """
    if d is None:
        d = responses()
    columns = [
        "rater", "pair_id", "kind", "rank", "in_core",
        "similarity_1_5", "connects",
        "seed", "seed_artist", "seed_title", "seed_genre",
        "other", "other_artist", "other_title", "other_genre",
        "track_a", "track_b",
    ]
    out = d[[c for c in columns if c in d.columns]].copy()
    out["same_genre"] = out["seed_genre"] == out["other_genre"]
    return out.sort_values(["rater", "pair_id"]).reset_index(drop=True)


def _main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=TEST_DIR / "responses")
    parser.add_argument("--csv", type=Path, nargs="?", const=TEST_DIR / "all_ratings.csv",
                        help="write every rating, with both tracks named, to one file")
    args = parser.parse_args()

    if args.csv:
        table = flat_table(responses(args.responses))
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"{len(table)} ratings x {table.shape[1]} columns -> {args.csv}")
        return

    d = responses(args.responses)

    print("\nscores by pair kind")
    print(d.groupby("kind")["similarity_1_5"].agg(["count", "mean"]).round(2).to_string())

    t = tags(d)
    print("\nwhat raters said connects the pairs")
    print(t.to_string(index=False))

    g = gaps(t)
    print("\nthe same words, against the features we have")
    print(g.to_string(index=False))

    holes = g[g["status"] == "NO FEATURE"]
    if len(holes):
        print(f"\n{len(holes)} word(s) people keep using with nothing behind them:")
        for row in holes.itertuples():
            print(f"  {row.tag:16} {row.mentions:3} mentions, mean score {row.mean_score}")


if __name__ == "__main__":
    _main()
