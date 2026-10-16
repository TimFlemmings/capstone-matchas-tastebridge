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

    # One key per round, concatenated. Round two's pair ids start with Q rather
    # than P, so a rating finds its own round's key and nothing collides —
    # which is what lets returns from both rounds share one folder.
    keys = sorted(TEST_DIR.glob("pairs_key*.csv"))
    if not keys:
        raise FileNotFoundError(f"no pairs_key*.csv in {TEST_DIR}")
    key = pd.concat([pd.read_csv(f) for f in keys], ignore_index=True)
    if key["pair_id"].duplicated().any():
        clash = sorted(key.loc[key["pair_id"].duplicated(), "pair_id"])
        raise ValueError(f"the same pair id appears in two keys: {clash[:5]}")
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


def blind_spots(d: pd.DataFrame = None, space_name: str = "heard"):
    """Where the model and the listeners disagree most, and what the words say.

    The counts above tell you which words people use. They do not tell you which
    ones the model is *failing* at, because a word can be used constantly on
    pairs the model already gets right.

    This asks the sharper question. For every rated pair, take the model's
    distance and the listeners' score, rank both, and subtract. A pair the model
    puts far away that people heard as close is a pair the model is missing
    something about — and whatever the raters ticked on it is the name of that
    something. Rhythm was found exactly this way, by hand.

    Two tables come back: the worst disagreements in each direction, and how
    each word's rate differs between them.
    """
    if d is None:
        d = responses()
    d = d.dropna(subset=["similarity_1_5"])

    from scipy.spatial.distance import cosine
    from sklearn.preprocessing import StandardScaler

    from modeling.data import load_fma
    from modeling.features import space

    X, _ = load_fma("small")
    S = space(space_name, X)
    Z = pd.DataFrame(StandardScaler().fit_transform(S), index=S.index)

    pairs = d.groupby("pair_id").agg(
        heard=("similarity_1_5", "mean"), votes=("similarity_1_5", "size"),
        a=("track_a", "first"), b=("track_b", "first"),
        seed_artist=("seed_artist", "first"), other_artist=("other_artist", "first"),
        seed_genre=("seed_genre", "first"), kind=("kind", "first"))
    pairs["distance"] = [cosine(Z.loc[a], Z.loc[b]) for a, b in zip(pairs.a, pairs.b)]

    # Ranks, not raw values: the two are on different scales and only their
    # ordering is being compared. Positive means people heard more connection
    # than the geometry has any account of.
    pairs["miss"] = (pairs["heard"].rank(pct=True)
                     - (1 - pairs["distance"].rank(pct=True)))

    words = d.assign(tag=d["connects"].fillna("").str.split()).explode("tag")
    words = words[words["tag"].astype(bool)]
    # Both ends are failures, and they are different failures. A high miss is a
    # connection the model cannot see; a low one is the model insisting on a
    # closeness nobody hears. Calling the second end "right" would hide half the
    # problem — it is the model overreaching, not agreeing.
    cut = pairs["miss"].quantile([0.25, 0.75])
    side = pd.Series("middle", index=pairs.index)
    side[pairs["miss"] >= cut[0.75]] = "blind"
    side[pairs["miss"] <= cut[0.25]] = "overreached"
    words = words.join(side.rename("side"), on="pair_id")

    rates = (words[words["side"] != "middle"]
             .pivot_table(index="tag", columns="side", values="pair_id",
                          aggfunc="count")
             .fillna(0))
    for column in ("blind", "overreached"):
        rates[column] = rates.get(column, 0)
    totals = side.value_counts()
    rates = rates.div([totals.get(c, 1) for c in rates.columns], axis=1).round(2)
    rates["gap"] = (rates["blind"] - rates["overreached"]).round(2)
    return pairs.sort_values("miss", ascending=False), rates.sort_values("gap", ascending=False)


def _main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=TEST_DIR / "responses")
    parser.add_argument("--csv", type=Path, nargs="?", const=TEST_DIR / "all_ratings.csv",
                        help="write every rating, with both tracks named, to one file")
    parser.add_argument("--blind-spots", action="store_true",
                        help="where the model and the listeners disagree, and what "
                             "the raters called those pairs — the next feature, if "
                             "there is one")
    parser.add_argument("--space", default="heard",
                        help="which feature space to measure the disagreement in")
    args = parser.parse_args()

    if args.blind_spots:
        worst, rates = blind_spots(responses(args.responses), args.space)
        show = ["seed_artist", "other_artist", "seed_genre", "kind",
                "heard", "distance", "miss"]
        print(f"\n=== the two ways the '{args.space}' space is wrong ===")
        print("\nBLIND — people heard a connection the geometry has no account of:")
        print(worst.head(8)[show].round(3).to_string())
        print("\nOVERREACHED — the model put these close and nobody agreed:")
        print(worst.tail(8)[show].round(3).to_string())
        print("\n=== how often each word was used at each end ===")
        print(rates.to_string())
        n_side = max(1, len(worst) // 4)
        print(f"\nEach column is a rate over about {n_side} pairs, so one pair moves it "
              f"by {1 / n_side:.2f}. Treat anything under that as noise.")
        print("A positive gap names something people hear and the model cannot. Check "
              "\nit against `modeling.listening` with no arguments: if that word already "
              "\nhas features, they are not doing their job; if it has none, it is the "
              "\nnext thing to build. That is how rhythm was found.")
        return

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
