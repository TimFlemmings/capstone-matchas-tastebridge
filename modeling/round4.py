"""Build round four: what the live app recommends, heard tier by tier.

Pre-registered in data/listening_test/ROUND4_PREREGISTRATION.md before anything
here was run. Round three tested a recommender that no longer ships; this one
takes its pairs from the live app itself, so it tests exactly what a user gets.

Two steps, because only a browser can reach the app from where this was built:

    uv run python -m modeling.round4 draw
        draws the seeds and writes round4/seeds.csv

    # call POST /recommendations for every seed at adventure 0.1, 0.5, 0.9 and
    # save the replies as round4/app_responses.json:
    #   {"<seed>": {"home": [[id, rank, distance], ...], "bridge": [...],
    #               "adventure": [...]}, ...}

    uv run python -m modeling.round4 build
        picks the pairs, cuts nothing (whole 30 s clips, as before), levels the
        volume, and writes clips, one rating form per rater, zips and the key

Everything lands in data/listening_test/round4/, which is gitignored — the
answer key included — until the analysis is done.
"""

import argparse
import json
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.round2 import (FLOOR, REFERENCE, TEST_DIR, assign,
                             rating_page_template, spell)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "fma_medium"
OUT_DIR = TEST_DIR / "round4"
APP = "https://mrln77-tasty-demo.hf.space"
TIERS = {"home": 0.1, "bridge": 0.5, "adventure": 0.9}
PER_TIER, FIRST = 2, 5     # two pairs per tier, from the first five returned
RANDOM_STATE = 4004


def audio_path(track_id: int) -> Path:
    name = f"{int(track_id):06d}"
    return AUDIO_DIR / name[:3] / f"{name}.mp3"


def used_before() -> tuple[set, set]:
    """Every track and artist that appeared in rounds one to three."""
    tracks, artists = set(), set()
    for key in sorted(TEST_DIR.glob("pairs_key*.csv")):
        k = pd.read_csv(key)
        tracks |= set(k["seed"]) | set(k["other"])
        artists |= set(k["seed_artist"].dropna()) | set(k["other_artist"].dropna())
    return tracks, artists


def medium_only():
    """Tracks in fma_medium but not fma_small: none has been in a listening test."""
    from modeling.data import load_tracks
    t = load_tracks("medium")
    meta = pd.DataFrame({"artist": t[("artist", "name")], "title": t[("track", "title")],
                         "genre": t[("track", "genre_top")]}).dropna(subset=["genre"])
    return meta[[audio_path(i).exists() for i in meta.index]]


def draw(n_seeds: int = 6, not_in_app=()):
    """`not_in_app`: drawn tracks the live app 404s on (it serves 24,757 of
    fma_medium's 25,000); they are removed and the draw repeated."""
    rng = np.random.default_rng(RANDOM_STATE)
    meta = medium_only().drop(index=list(not_in_app), errors="ignore")
    tracks, artists = used_before()
    meta = meta[~meta.index.isin(tracks) & ~meta.artist.isin(artists)]
    genres = meta.genre.value_counts()
    genres = genres[genres >= 50].index.tolist()
    seeds, seen = [], set()
    for g in rng.permutation(genres):
        pool = meta[(meta.genre == g) & ~meta.artist.isin(seen)]
        pick = int(rng.choice(pool.index))
        seeds.append(pick)
        seen.add(meta.loc[pick, "artist"])
        if len(seeds) == n_seeds:
            break
    out = meta.loc[seeds].rename_axis("seed").reset_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "seeds.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nseed ids: {json.dumps([int(s) for s in seeds])}")


def build(n_raters: int = 6, core: int = 7):
    rng = np.random.default_rng(RANDOM_STATE + 1)
    seeds = pd.read_csv(OUT_DIR / "seeds.csv").set_index("seed")
    replies = json.loads((OUT_DIR / "app_responses.json").read_text())
    from modeling.data import load_tracks
    t = load_tracks(None)
    meta = pd.DataFrame({"artist": t[("artist", "name")], "title": t[("track", "title")],
                         "genre": t[("track", "genre_top")]})

    rows = []
    taken = set(seeds.index)
    for seed in seeds.index:
        artist = meta.loc[seed, "artist"]
        for tier in TIERS:
            first = [r for r in replies[str(seed)][tier]
                     if meta.loc[int(r[0]), "artist"] != artist][:FIRST]
            if len(first) < PER_TIER:
                raise ValueError(f"seed {seed} {tier}: only {len(first)} usable picks")
            for i in rng.choice(len(first), PER_TIER, replace=False):
                tid, rank, dist = first[i]
                rows.append({"kind": tier, "seed": seed, "other": int(tid),
                             "rank": int(rank), "distance": float(dist)})
                taken.add(int(tid))
    # The floor: never heard before, a different artist, and not one of the picks.
    tracks_before, _ = used_before()
    pool = medium_only()
    pool = pool[~pool.index.isin(tracks_before | taken)]
    for seed in seeds.index:
        cand = pool[pool.artist != meta.loc[seed, "artist"]]
        other = int(rng.choice(cand.index))
        pool = pool.drop(other)
        rows.append({"kind": "random", "seed": seed, "other": other,
                     "rank": np.nan, "distance": np.nan})

    pairs = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    pairs["pair_id"] = [f"T{i:02d}" for i in range(1, len(pairs) + 1)]
    flip = rng.random(len(pairs)) < 0.5
    pairs["track_a"] = np.where(flip, pairs["other"], pairs["seed"])
    pairs["track_b"] = np.where(flip, pairs["seed"], pairs["other"])

    # Volume levelled from RMS energy, exactly as rounds one to three.
    from modeling.data import load_features
    heard = sorted(set(pairs.track_a) | set(pairs.track_b))
    rms = load_features(heard)[("rmse", "mean", "01")]
    target = rms.quantile(REFERENCE)
    volume = {int(i): float(np.clip(target / rms.get(i, target), FLOOR, 1.0)) for i in heard}

    clips = OUT_DIR / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for r in pairs.itertuples():
        for side, tid in (("A", r.track_a), ("B", r.track_b)):
            shutil.copyfile(audio_path(tid), clips / f"{r.pair_id}_{side}.mp3")

    manifest = pd.DataFrame({
        "pair_id": pairs.pair_id,
        "clip_a": pairs.pair_id + "_A.mp3", "clip_b": pairs.pair_id + "_B.mp3",
        "volume_a": [round(volume[int(i)], 3) for i in pairs.track_a],
        "volume_b": [round(volume[int(i)], 3) for i in pairs.track_b]})
    assignments, core_ids = assign(pairs, n_raters, core, rng, RANDOM_STATE, suffix="d")
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    assignments.to_csv(OUT_DIR / "assignments.csv", index=False)

    key = pairs.copy()
    for side in ("seed", "other"):
        j = meta.reindex(key[side]).reset_index(drop=True)
        key[[f"{side}_{c}" for c in j.columns]] = j
    key["in_core"] = key.pair_id.isin(core_ids)
    key["round"] = 4
    key.to_csv(OUT_DIR / "pairs_key_r4.csv", index=False)

    template = rating_page_template()
    (OUT_DIR / "zips").mkdir(exist_ok=True)
    for rater, group in assignments.groupby("rater"):
        folder = OUT_DIR / "by_rater" / rater
        (folder / "clips").mkdir(parents=True, exist_ok=True)
        mine = manifest[manifest.pair_id.isin(group.pair_id)]
        mine.to_csv(folder / "manifest.csv", index=False)
        for r in mine.itertuples():
            for clip in (r.clip_a, r.clip_b):
                shutil.copyfile(clips / clip, folder / "clips" / clip)
        page = (template.replace("__DATA__", json.dumps(mine.to_dict("records")))
                .replace("__RATER__", rater)
                .replace("__N__ pairs of 30-second", f"{spell(len(mine))} pairs of 30-second")
                .replace("__N__", str(len(mine))))
        for broken in ("\\\\n", "\\\\u"):
            if broken in page:
                raise ValueError(f"{rater}: doubled escapes in the rating form")
        (folder / "rate.html").write_text(page, encoding="utf-8")
        with zipfile.ZipFile(OUT_DIR / "zips" / f"round4_{rater}.zip", "w") as z:
            for f in sorted(folder.rglob("*")):
                if f.is_file() and f.name != "manifest.csv":
                    z.write(f, Path(rater) / f.relative_to(folder))
        logger.info("%s: %s pairs", rater, len(mine))

    print(f"\n=== round four: {len(seeds)} seeds, {len(pairs)} pairs ===")
    print(pairs.kind.value_counts().to_string())
    print(f"core heard by everyone: {len(core_ids)} pairs")
    print(f"zips: {OUT_DIR / 'zips'}")


def main(argv=None):
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("step", choices=["draw", "build"])
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--raters", type=int, default=6)
    p.add_argument("--core", type=int, default=7)
    p.add_argument("--not-in-app", type=int, nargs="*", default=[])
    a = p.parse_args(argv)
    draw(a.seeds, a.not_in_app) if a.step == "draw" else build(a.raters, a.core)


if __name__ == "__main__":
    main()
