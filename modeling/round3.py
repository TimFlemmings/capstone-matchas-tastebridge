"""Build a third listening round: Discogs-EffNet against `heard`, by ear.

Tim replaced the hand-built features with a pretrained network, and every number
behind that change is offline — genre agreement, and where Spotify's picks for
one seed land in our ranking. Nobody has heard what EffNet recommends. This
builds the round that does, on seeds rounds one and two never touched. Seven
pairs per seed:

    2  from EffNet's top five               — the thing under test
    2  from heard's top five                — the baseline it has to beat
    1  from EffNet's far band               — Tim's Bridge or Adventurous tier
    1  from heard's far band, same ranks    — so the far pairs cannot rig it
    1  drawn at random from the catalogue   — the floor

The fourth line is round two's lesson. Its deep pairs came from `heard` only,
and a space's far picks sit far from the seed in that same space by
construction, so they dragged its correlation down. Here both spaces send one.
Half the seeds use the Bridge band (ranks 30–150), half the Adventurous band
(500–2000), so both tiers are heard without lengthening the round.

The other lesson: round two had 13 same-artist pairs out of 72, and removing
them changed its conclusion. Every source here skips the seed's own artist, and
"top five" means the five nearest after that.

Everything else is round two's machinery, imported rather than copied — the
form, the clip layout, the volume matching, the shared core — so the returns
drop into `data/listening_test/responses/` and `modeling.listening` merges all
three rounds without being told.

    uv run python -m modeling.round3                 # 12 seeds, 84 pairs
    uv run python -m modeling.round3 --raters 6

The threshold is in `data/listening_test/ROUND3_PREREGISTRATION.md`, committed
before this file produced a single pair.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.round2 import (
    REFERENCE, FLOOR, TEST_DIR, TOP_K, assign, audio_path, build_space,
    choose_seeds, ranked_neighbours, rating_page_template, spell, write_clips,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS = REPO_ROOT / "data" / "effnet_embeddings_full.npz"
OUT_DIR = TEST_DIR / "round3"

# Tim's tier bands from notebook 09, as 1-based ranks after the seed is dropped.
BANDS = {"bridge": (30, 150), "adventurous": (500, 2000)}


def load_effnet(path: Path = EMBEDDINGS) -> pd.DataFrame:
    """The raw 1280-dim embeddings, indexed by track id.

    Raw, with cosine distance and no scaler or PCA, because that is exactly what
    `EffNetRecommender` serves. Testing a tidied-up version would be testing a
    model nobody runs.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — copy effnet_embeddings_full.npz from Tim "
            "(tim-effnet/data/) into data/")
    loaded = np.load(path)
    return pd.DataFrame(loaded["embeddings"], index=loaded["ids"].astype(int))


def build_pairs(seeds, band_of, effnet_rank, heard_rank, artist, pool, rng):
    """Seven pairs per seed, five sources, no track or artist repeated per seed."""
    rows = []
    for s in seeds:
        taken = {s}
        taken_artists = {artist.get(s)}

        def usable(ranking):
            return [t for t in ranking if t not in taken
                    and artist.get(t) not in taken_artists and audio_path(t).exists()]

        def pick(candidates, kind, ranking, n=1):
            if len(candidates) < n:
                logger.warning("seed %s: only %s of %s %s pairs available",
                               s, len(candidates), n, kind)
            chosen = [int(t) for t in rng.choice(candidates, size=min(n, len(candidates)),
                                                 replace=False)]
            for t in chosen:
                taken.add(t)
                rank = ranking.index(t) + 1 if ranking is not None else None
                rows.append({"kind": kind, "seed": s, "other": t, "rank": rank,
                             "band": band_of[s]})

        # Top five of each space, after the seed's artist is set aside. Each pick
        # is also struck from the other space's list, so the two never share one.
        pick(usable(effnet_rank[s])[:TOP_K], "effnet_top", effnet_rank[s], n=2)
        pick(usable(heard_rank[s])[:TOP_K], "heard_top", heard_rank[s], n=2)

        low, high = BANDS[band_of[s]]
        for name, ranking in (("effnet", effnet_rank[s]), ("heard", heard_rank[s])):
            band = [t for t in ranking[low - 1:high] if t in set(usable(ranking))]
            pick(band, f"{name}_{band_of[s]}", ranking)

        pick(usable(pool), "random", None)
    return pd.DataFrame(rows)


def main(argv=None):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seeds", type=int, default=12, help="fresh seed tracks")
    parser.add_argument("--raters", type=int, default=6, help="how many people")
    parser.add_argument("--core", type=int, default=7,
                        help="pairs everybody hears, for rater agreement")
    parser.add_argument("--random-state", type=int, default=3003,
                        help="different from rounds one and two, so the draws differ")
    args = parser.parse_args(argv)

    from modeling.data import load_fma
    from modeling.features import space

    rng = np.random.default_rng(args.random_state)
    X_all, tracks = load_fma("small")

    logger.info("Building heard and loading EffNet")
    heard = space("heard", X_all)
    effnet = load_effnet()
    # One catalogue for both, so rank 500 means the same thing on either side.
    common = heard.index.intersection(effnet.index)
    heard, effnet = heard.loc[common], effnet.loc[common]
    Z_heard = build_space(heard, args.random_state)
    logger.info("catalogue %s tracks | heard %s columns -> %s components | "
                "EffNet %s dims, raw", len(common), heard.shape[1],
                Z_heard.shape[1], effnet.shape[1])

    used = set()
    for key_name in ("pairs_key.csv", "pairs_key_r2.csv"):
        key_path = TEST_DIR / key_name
        if key_path.exists():
            earlier = pd.read_csv(key_path)
            used |= set(earlier["seed"]) | set(earlier["other"])
        else:
            logger.warning("%s not found — its tracks cannot be excluded", key_name)
    logger.info("Earlier rounds used %s tracks; excluding all of them", len(used))

    genre_of = tracks[("track", "genre_top")]
    artist = tracks[("artist", "id")].to_dict()
    # One candidate per artist, so no artist can supply two seeds. The first
    # draw gave Blue Dot Sessions two of twelve, which makes those two seeds one
    # observation twice; it also stops prolific uploaders getting more chances.
    candidates = (pd.Series(list(common), index=[artist.get(t) for t in common])
                  .groupby(level=0)
                  .apply(lambda ids: int(rng.choice(ids.to_numpy())))
                  .tolist())
    seeds = choose_seeds(candidates, used, rng, args.seeds, genres=genre_of)
    half = len(seeds) // 2
    band_of = {s: ("bridge" if i < half else "adventurous") for i, s in enumerate(seeds)}

    depth = max(high for _, high in BANDS.values())
    effnet_rank = ranked_neighbours(effnet.to_numpy(), common, seeds, depth)
    heard_rank = ranked_neighbours(Z_heard, common, seeds, depth)

    spoken = {t for s in seeds for t in effnet_rank[s][:TOP_K * 2] + heard_rank[s][:TOP_K * 2]}
    pool = [int(t) for t in common if t not in spoken and t not in seeds and t not in used]

    pairs = build_pairs(seeds, band_of, effnet_rank, heard_rank, artist, pool, rng)
    pairs = pairs.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    pairs["pair_id"] = [f"S{i:02d}" for i in range(1, len(pairs) + 1)]

    # Coin flip for play order, exactly as round two.
    flip = rng.random(len(pairs)) < 0.5
    pairs["track_a"] = np.where(flip, pairs["other"], pairs["seed"])
    pairs["track_b"] = np.where(flip, pairs["seed"], pairs["other"])

    rms = X_all[("rmse", "mean", "01")]
    heard_tracks = sorted(set(pairs["track_a"]) | set(pairs["track_b"]))
    target = rms.loc[heard_tracks].quantile(REFERENCE)
    volume = {int(t): float(np.clip(target / rms.loc[t], FLOOR, 1.0)) for t in heard_tracks}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)          # a rerun must not leave the last draw's clips behind
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
    assignments, core = assign(pairs, args.raters, args.core, rng, args.random_state,
                               suffix="c")
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    assignments.to_csv(OUT_DIR / "assignments.csv", index=False)

    meta = tracks.loc[:, [("artist", "name"), ("track", "title"), ("track", "genre_top")]]
    meta.columns = ["artist", "title", "genre"]
    key = pairs.copy()
    for side in ("seed", "other"):
        joined = meta.reindex(key[side]).reset_index(drop=True)
        key[[f"{side}_{c}" for c in joined.columns]] = joined
    key["in_core"] = key["pair_id"].isin(core)
    key["round"] = 3
    key.to_csv(TEST_DIR / "pairs_key_r3.csv", index=False)

    template = rating_page_template()
    zips = OUT_DIR / "zips"
    zips.mkdir(parents=True, exist_ok=True)
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
                .replace("__N__ pairs of 30-second", f"{spell(len(mine))} pairs of 30-second")
                .replace("__N__", str(len(mine))))
        for broken in ("\\\\n", "\\\\u"):
            if broken in page:
                raise ValueError(f"{rater}: the rating form has doubled escapes")
        (folder / "rate.html").write_text(page, encoding="utf-8")
        shutil.make_archive(str(zips / f"round3_{rater}"), "zip", folder.parent, rater)
        logger.info("%s: %s pairs", rater, len(mine))

    print(f"\n=== round three: {len(seeds)} seeds, {len(pairs)} pairs ===\n")
    print(pairs["kind"].value_counts().to_string())
    print(f"\ncore heard by everyone : {len(core)} pairs")
    print(f"packages               : {zips}")
    print(f"answer key             : {TEST_DIR / 'pairs_key_r3.csv'}")
    print("\nSeeds:")
    for s in seeds:
        row = meta.loc[s]
        print(f"  {s:>7}  {str(row['artist'])[:28]:30} [{row['genre']}]  {band_of[s]}")
    print("\nReturns go in data/listening_test/responses/ as <rater>_ratings.csv.")
    print("The threshold is in ROUND3_PREREGISTRATION.md — read it before the returns.")


if __name__ == "__main__":
    main()
