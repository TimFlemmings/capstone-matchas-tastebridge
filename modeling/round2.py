"""Build a second listening round, on seeds the first one never touched.

Round one settled two things — the recommender beats a random pair to a human
ear, and rank 200 is heard as no less related than rank 1. It could not settle a
third: its eighty pairs were generated from the old 518-column space, because
the rhythm and drum columns did not exist yet. So the ratings *chose* the space
we ship and have never *tested* it. Nobody has heard what this model recommends.

That is what this builds. Six pairs per seed:

    2  from the shipped space's top five   — the thing that has never been heard
    2  from the old space's top five       — the same question, asked of its rival
    1  from the shipped space at rank 50 or 200 — the dial, in the new space
    1  drawn at random from the catalogue  — the floor

The mix is the point. Pairs taken only from the space under test sit close
together in it by construction, which would rig the correlation in its favour.
Round one had that bias pointing the other way — every pair came from the old
space's ranking, and the new space still matched the ratings better. Drawing
from both is how round two avoids owing anyone that explanation.

Everything else follows round one exactly — blind pair ids, a coin flip for
which clip plays first, volume matched from each track's RMS energy — so the
returns drop into `data/listening_test/responses/` beside the old ones and
`modeling.listening` merges the two rounds without being told.

    uv run python -m modeling.round2                  # 4 seeds, 24 pairs
    uv run python -m modeling.round2 --seeds 8        # a longer round
    uv run python -m modeling.round2 --raters 6

Read `FOR_TIM.md` before running it: what this cannot do is settle `heard`
against `heard_wide`, which needs a forced choice rather than a rating.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "fma_small"
TEST_DIR = REPO_ROOT / "data" / "listening_test"
OUT_DIR = TEST_DIR / "round2"
NOTEBOOK = REPO_ROOT / "notebooks" / "02_listening_test.ipynb"

DEPTHS = (50, 200)
TOP_K = 5
REFERENCE, FLOOR = 0.05, 0.15     # round one's volume settings, unchanged


def audio_path(track_id: int) -> Path:
    """FMA stores 000140.mp3 under a folder named by its first three digits."""
    name = f"{track_id:06d}"
    return AUDIO_DIR / name[:3] / f"{name}.mp3"


def rating_page_template() -> str:
    """The form round one used, read out of the notebook that defines it.

    Copying it here would leave two versions to keep in step, and the one the
    raters actually saw is the one in the notebook. Extracted rather than
    duplicated so a change to the form reaches both rounds.
    """
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"{NOTEBOOK} is missing — it defines the rating form")
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if "RATING_PAGE = " in source and "__RATER__" in source:
            body = source.split('RATING_PAGE = """', 1)[1]
            return body.split('"""', 1)[0]
    raise ValueError(f"no RATING_PAGE template found in {NOTEBOOK.name}")


def ranked_neighbours(Z, index, seeds, depth):
    """For each seed, the catalogue ordered by distance, nearest first."""
    positions = {t: i for i, t in enumerate(index)}
    nn = NearestNeighbors(metric="cosine").fit(Z)
    out = {}
    for s in seeds:
        _, idx = nn.kneighbors(Z[[positions[s]]], n_neighbors=depth + 2)
        out[s] = [int(index[i]) for i in idx[0] if int(index[i]) != s][:depth + 1]
    return out


def build_space(X: pd.DataFrame, random_state: int = 42):
    """Scale and project one feature space, exactly as train_pca.py does."""
    Z = PCA(n_components=0.90, random_state=random_state).fit_transform(
        StandardScaler().fit_transform(X))
    return Z


def choose_seeds(candidates, used, rng, n, genres=None):
    """Fresh seeds, with audio on disk and no part in round one.

    Reusing a seed would put the new ratings back inside the sample that chose
    the space, which is the whole thing round two exists to escape.

    Spread across genres rather than drawn freely, because a free draw of twelve
    came back five Pop and three Rock with no Hip-Hop at all. Pop is the genre
    this recommender handles worst — 0.199 against a 0.125 floor, because Pop is
    a market category and not a sound — so that round would have scored badly on
    the luck of the draw and told us nothing about where the model actually
    fails. One seed per genre first, then the remainder drawn at random.
    """
    pool = [t for t in candidates if t not in used and audio_path(t).exists()]
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} usable seeds outside round one, wanted {n}")
    if genres is None:
        return [int(t) for t in rng.choice(pool, size=n, replace=False)]

    by_genre = {}
    for t in pool:
        by_genre.setdefault(str(genres.get(t)), []).append(t)

    chosen = []
    for genre in sorted(by_genre):                      # one each, in a fixed order
        if len(chosen) < n:
            chosen.append(int(rng.choice(by_genre[genre])))
    rest = [t for t in pool if t not in chosen]
    if len(chosen) < n:
        chosen += [int(t) for t in rng.choice(rest, size=n - len(chosen), replace=False)]
    return list(rng.permutation(chosen[:n]).astype(int))


def build_pairs(seeds, heard_rank, all_rank, pool, rng):
    """Six pairs per seed, from four sources, with no track used twice per seed."""
    rows = []
    for s in seeds:
        taken = {s}

        def pick(candidates, kind, n=1):
            fresh = [t for t in candidates if t not in taken and audio_path(t).exists()]
            chosen = list(rng.choice(fresh, size=min(n, len(fresh)), replace=False))
            for t in chosen:
                taken.add(int(t))
                rank = candidates.index(int(t)) + 1 if int(t) in candidates else None
                rows.append({"kind": kind, "seed": s, "other": int(t), "rank": rank})
            if len(chosen) < n:
                logger.warning("seed %s: only %s of %s %s pairs available",
                               s, len(chosen), n, kind)

        pick(heard_rank[s][:TOP_K], "heard_top", n=2)
        pick(all_rank[s][:TOP_K], "all_top", n=2)
        pick([heard_rank[s][d - 1] for d in DEPTHS], "heard_deep", n=1)
        pick([t for t in pool if t not in taken], "random", n=1)
    return pd.DataFrame(rows)


def write_clips(pairs, clips_dir):
    clips_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for row in pairs.itertuples():
        for side, track_id in (("A", row.track_a), ("B", row.track_b)):
            destination = clips_dir / f"{row.pair_id}_{side}.mp3"
            shutil.copyfile(audio_path(int(track_id)), destination)
            total += destination.stat().st_size
    return len(pairs) * 2, total


def assign(pairs, n_raters, core_size, rng, seed):
    """Everyone hears a shared core; the rest is split so nothing goes unheard.

    The core is what lets the raters be measured against each other at all —
    round one's came to r = 0.53, and without it there is no way to say how much
    the six-rater average can be trusted.
    """
    core = []
    for kind, group in pairs.groupby("kind"):
        share = max(1, round(core_size * len(group) / len(pairs)))
        core += group.sample(share, random_state=seed)["pair_id"].tolist()

    rest = [str(p) for p in rng.permutation(
        [p for p in pairs["pair_id"] if p not in core])]
    rotation = -(-len(rest) // n_raters) if rest else 0

    rows = []
    for r in range(n_raters):
        share = rest[r * rotation:(r + 1) * rotation] if rotation else []
        for pair_id in core + share:
            rows.append({"rater": f"R{r + 1}", "pair_id": pair_id,
                         "in_core": pair_id in core})
    return pd.DataFrame(rows), core


def main(argv=None):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seeds", type=int, default=4, help="fresh seed tracks")
    parser.add_argument("--raters", type=int, default=6, help="how many people")
    parser.add_argument("--core", type=int, default=6,
                        help="pairs everybody hears, for rater agreement")
    parser.add_argument("--random-state", type=int, default=2026,
                        help="different from round one's 42, so the draws differ")
    parser.add_argument("--free-draw", action="store_true",
                        help="draw seeds at random instead of one per genre first")
    args = parser.parse_args(argv)

    from modeling.data import load_fma
    from modeling.features import space

    rng = np.random.default_rng(args.random_state)
    X_all, tracks = load_fma("small")

    logger.info("Building both feature spaces")
    heard = space("heard", X_all)
    old = space("all", X_all)
    common = heard.index.intersection(old.index)
    heard, old = heard.loc[common], old.loc[common]

    Z_heard = build_space(heard, args.random_state)
    Z_all = build_space(old, args.random_state)
    logger.info("heard: %s columns -> %s components | all: %s -> %s",
                heard.shape[1], Z_heard.shape[1], old.shape[1], Z_all.shape[1])

    key_path = TEST_DIR / "pairs_key.csv"
    used = set()
    if key_path.exists():
        round1 = pd.read_csv(key_path)
        used = set(round1["seed"]) | set(round1["other"])
        logger.info("Round one used %s tracks; excluding all of them", len(used))

    genre_of = tracks[("track", "genre_top")]
    seeds = choose_seeds(list(common), used, rng, args.seeds,
                         genres=None if args.free_draw else genre_of)
    heard_rank = ranked_neighbours(Z_heard, common, seeds, max(DEPTHS))
    all_rank = ranked_neighbours(Z_all, common, seeds, TOP_K)

    spoken = {t for s in seeds for t in heard_rank[s][:TOP_K] + all_rank[s][:TOP_K]}
    pool = [int(t) for t in common if t not in spoken and t not in seeds
            and t not in used]

    pairs = build_pairs(seeds, heard_rank, all_rank, pool, rng)
    pairs = pairs.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    pairs["pair_id"] = [f"Q{i:02d}" for i in range(1, len(pairs) + 1)]

    # Which clip plays first is decided by a coin, not by which one the model
    # chose: an ear that learns "the second one is always the recommendation"
    # stops rating the music.
    flip = rng.random(len(pairs)) < 0.5
    pairs["track_a"] = np.where(flip, pairs["other"], pairs["seed"])
    pairs["track_b"] = np.where(flip, pairs["seed"], pairs["other"])

    # Two clips at the same volume are heard as more alike whatever the music is
    # doing, and FMA is thousands of independent uploads with no shared
    # mastering. The reference is the 5th percentile rather than the true
    # minimum, or one near-silent clip sets the level for everything.
    rms = X_all[("rmse", "mean", "01")]
    heard_tracks = sorted(set(pairs["track_a"]) | set(pairs["track_b"]))
    target = rms.loc[heard_tracks].quantile(REFERENCE)
    volume = {int(t): float(np.clip(target / rms.loc[t], FLOOR, 1.0))
              for t in heard_tracks}

    clips_dir = OUT_DIR / "clips"
    n_clips, size = write_clips(pairs, clips_dir)
    logger.info("%s clips, %.0f MB", n_clips, size / 1e6)

    manifest = pd.DataFrame({
        "pair_id": pairs["pair_id"],
        "clip_a": pairs["pair_id"] + "_A.mp3",
        "clip_b": pairs["pair_id"] + "_B.mp3",
        "volume_a": [round(volume[int(t)], 3) for t in pairs["track_a"]],
        "volume_b": [round(volume[int(t)], 3) for t in pairs["track_b"]],
    })
    assignments, core = assign(pairs, args.raters, args.core, rng, args.random_state)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    assignments.to_csv(OUT_DIR / "assignments.csv", index=False)

    # The key carries the same columns round one's does, so the two concatenate.
    meta = tracks.loc[:, [("artist", "name"), ("track", "title"), ("track", "genre_top")]]
    meta.columns = ["artist", "title", "genre"]
    key = pairs.copy()
    for side in ("seed", "other"):
        joined = meta.reindex(key[side]).reset_index(drop=True)
        key[[f"{side}_{c}" for c in joined.columns]] = joined
    key["in_core"] = key["pair_id"].isin(core)
    key["round"] = 2
    key.to_csv(TEST_DIR / "pairs_key_r2.csv", index=False)

    template = rating_page_template()
    for rater, group in assignments.groupby("rater"):
        folder = OUT_DIR / "by_rater" / rater
        (folder / "clips").mkdir(parents=True, exist_ok=True)
        mine = manifest[manifest["pair_id"].isin(group["pair_id"])]
        mine.to_csv(folder / "manifest.csv", index=False)
        for row in mine.itertuples():
            for clip in (row.clip_a, row.clip_b):
                shutil.copyfile(clips_dir / clip, folder / "clips" / clip)
        page = (template
                .replace("__DATA__", json.dumps(mine.to_dict("records")))
                .replace("__RATER__", rater)
                .replace("__N__ pairs of 30-second", f"{len(mine)} pairs of 30-second")
                .replace("__N__", str(len(mine))))
        (folder / "rate.html").write_text(page, encoding="utf-8")
        mb = sum(f.stat().st_size for f in (folder / "clips").glob("*.mp3")) / 1e6
        logger.info("%s: %s pairs, %.0f MB", rater, len(mine), mb)

    print(f"\n=== round two: {len(seeds)} seeds, {len(pairs)} pairs ===\n")
    print(pairs["kind"].value_counts().to_string())
    print(f"\ncore heard by everyone : {len(core)} pairs")
    print(f"packages               : {OUT_DIR / 'by_rater'}")
    print(f"answer key             : {TEST_DIR / 'pairs_key_r2.csv'}")
    print("\nSeeds:")
    for s in seeds:
        row = meta.loc[s]
        print(f"  {s:>7}  {str(row['artist'])[:28]:30} [{row['genre']}]")
    print("\nReturns go in data/listening_test/responses/ as <rater>_ratings.csv,")
    print("beside round one's. `uv run python -m modeling.listening --csv` merges both.")
    print("\nWrite down what would count as a failure BEFORE the returns arrive.")


if __name__ == "__main__":
    main()
