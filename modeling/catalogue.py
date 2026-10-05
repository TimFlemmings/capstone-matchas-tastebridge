"""The app's track catalogue: fma_medium tracks with both embeddings, re-uploads collapsed.

One loader for both apps (`app/main.py`, `app_effnet.py`), so they serve the
same tracks:

- **Both embeddings.** EffNet drives recommendations, CLAP drives text search;
  a track missing either is left out. That's 24,980 of fma_medium's 25,000 --
  the other 20 are files FMA ships truncated (notebook 11, part A).
- **No re-uploads.** FMA holds some tracks twice under different ids (e.g.
  *Rachel Ries -- Ghost* as 91101 and 109480). Two tracks whose vectors agree
  at cosine >= 0.999 in *both* spaces are the same recording; only one is kept,
  so a seed is never recommended its own twin. FMA cuts each clip from a
  different point in each upload, so a re-upload's clip isn't always
  identical audio (*Ghost*: 0.89 / 0.92); same normalised artist + title
  *and* >= 0.85 in both spaces counts too. The similarity floor matters:
  below ~0.8 same-titled pairs are mostly different songs ("Zea Live at
  Beginner" is four different recordings).
- **No silence.** Every fully silent file has the same vectors, so they all
  land in one "duplicate" group; if a group's surviving copy is silent, the
  whole group goes (3 tracks in fma_medium: 44374, 48949, 107535).

The list is precomputed into
  `data/processed/duplicates_medium.csv` (small, committed) by:

    uv run python -m modeling.catalogue
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.data import load_tracks

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EFFNET_PATH = REPO_ROOT / "data" / "effnet_embeddings_medium.npz"
CLAP_PATH = REPO_ROOT / "data" / "clap_embeddings_medium.npz"
DUPLICATES_PATH = REPO_ROOT / "data" / "processed" / "duplicates_medium.csv"
WEB_AUDIO_DIR = REPO_ROOT / "data" / "web_audio"
ORIGINAL_AUDIO_DIR = REPO_ROOT / "data" / "fma_medium"
DUPLICATE_COSINE = 0.999
SAME_TITLE_COSINE = 0.85
SILENCE_RMS = 1e-4


@dataclass
class Catalogue:
    """Row i of `effnet` and `clap` is track `ids[i]`; `tracks` is indexed by the same ids."""

    tracks: pd.DataFrame
    ids: np.ndarray
    effnet: np.ndarray
    clap: np.ndarray
    removed: dict  # dropped track id -> the id it duplicates, or None if it was silent

    def resolve(self, track_id: int) -> int:
        """The catalogue id to use for `track_id`: itself, or the kept copy of a
        re-upload. Raises `TrackUnavailable` with a message fit to show a user."""
        track_id = int(track_id)
        if track_id in self.tracks.index:
            return track_id
        if track_id in self.removed:
            if self.removed[track_id] is not None:
                return int(self.removed[track_id])
            raise TrackUnavailable(f"Track {track_id} is silent in the Free Music Archive, so TasteBridge leaves it out.")
        raise TrackUnavailable(f"Track {track_id} isn't in the TasteBridge catalogue.")


class TrackUnavailable(KeyError):
    """A track id the catalogue can't serve; `str()` is a user-facing message."""

    def __str__(self):
        return self.args[0]


def local_audio_path(track_id, web_dir=WEB_AUDIO_DIR, original_dir=ORIGINAL_AUDIO_DIR) -> Path:
    """The clip to serve for a track: the 128 kbps loudness-matched web copy
    (`modeling.transcode_audio`, the same file the cloud serves) if it exists,
    else FMA's original."""
    name = f"{int(track_id):06d}"
    web = Path(web_dir) / name[:3] / f"{name}.mp3"
    return web if web.is_file() else Path(original_dir) / name[:3] / f"{name}.mp3"


def _aligned(effnet_path=EFFNET_PATH, clap_path=CLAP_PATH):
    """(ids, effnet, clap) over the tracks present in both files, in id order."""
    e, c = np.load(effnet_path), np.load(clap_path)
    ids = np.intersect1d(e["ids"], c["ids"])
    e_rows = pd.Series(np.arange(len(e["ids"])), index=e["ids"]).loc[ids].to_numpy()
    c_rows = pd.Series(np.arange(len(c["ids"])), index=c["ids"]).loc[ids].to_numpy()
    return ids, e["embeddings"][e_rows], c["embeddings"][c_rows]


def _close_pairs(emb: np.ndarray, threshold: float, chunk: int = 2048) -> set:
    """Every (i, j), i < j, with cosine(emb[i], emb[j]) >= threshold."""
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    pairs = set()
    for start in range(0, len(unit), chunk):
        sims = unit[start:start + chunk] @ unit.T
        rows, cols = np.nonzero(sims >= threshold)
        pairs.update((start + r, c) for r, c in zip(rows, cols) if start + r < c)
    return pairs


def _is_silent(track_id) -> bool:
    import librosa

    from modeling.clap_embed import audio_path
    audio, _ = librosa.load(str(audio_path(track_id)), sr=8000, mono=True)
    return float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS


def _same_title_pairs(ids, effnet, clap, tracks: pd.DataFrame, threshold: float) -> set:
    """(i, j), i < j: same normalised artist + title, cosine >= threshold in both spaces."""
    import re

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", str(s).casefold()) if pd.notna(s) else ""

    meta = tracks.reindex(ids)
    key = pd.Series([f"{norm(a)}|{norm(t)}" for a, t in zip(meta[("artist", "name")], meta[("track", "title")])])
    e = effnet / np.linalg.norm(effnet, axis=1, keepdims=True)
    c = clap / np.linalg.norm(clap, axis=1, keepdims=True)
    pairs = set()
    for k, rows in key.groupby(key).groups.items():
        if len(rows) < 2 or k.startswith("|") or k.endswith("|"):
            continue
        rows = sorted(rows)
        for a, i in enumerate(rows):
            for j in rows[a + 1:]:
                if e[i] @ e[j] >= threshold and c[i] @ c[j] >= threshold:
                    pairs.add((i, j))
    return pairs


def find_duplicates(ids, effnet, clap, tracks: pd.DataFrame, threshold: float = DUPLICATE_COSINE,
                    same_title_threshold: float = SAME_TITLE_COSINE) -> pd.DataFrame:
    """One row per dropped track: `track_id`, the `kept_id` it duplicates, and the `reason`.

    Groups are connected components of the "near-identical in both spaces"
    graph. Which copy survives: the fma_small one if there is one (so every
    evaluation done on fma_small still refers to a track in the catalogue),
    then the most-listened, then the lowest id. A group whose survivor is
    silent is dropped whole (`kept_id` empty, reason "silent").
    """
    pairs = (_close_pairs(effnet, threshold) & _close_pairs(clap, threshold)) \
        | _same_title_pairs(ids, effnet, clap, tracks, same_title_threshold)

    parent = list(range(len(ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[root(i)] = root(j)

    groups = {}
    for i, j in pairs:
        for k in (i, j):
            groups.setdefault(root(k), set()).add(int(ids[k]))

    in_small = tracks[("set", "subset")] == "small"
    listens = tracks[("track", "listens")]
    rows = []
    for members in groups.values():
        kept = min(members, key=lambda t: (not in_small.get(t, False), -listens.get(t, 0), t))
        if _is_silent(kept):
            rows += [{"track_id": t, "kept_id": None, "reason": "silent"} for t in sorted(members)]
        else:
            rows += [{"track_id": t, "kept_id": kept, "reason": "re-upload"} for t in sorted(members - {kept})]
    out = pd.DataFrame(rows, columns=["track_id", "kept_id", "reason"]).sort_values("track_id").reset_index(drop=True)
    out["kept_id"] = out["kept_id"].astype("Int64")
    return out


def load_catalogue(tracks: pd.DataFrame = None, effnet_path=EFFNET_PATH, clap_path=CLAP_PATH,
                   duplicates_path=DUPLICATES_PATH) -> Catalogue:
    """`tracks` defaults to fma_medium's rows of the project's tracks.csv; pass
    your own if your metadata lives elsewhere (it's filtered to the catalogue)."""
    ids, effnet, clap = _aligned(effnet_path, clap_path)
    if tracks is None:
        tracks = load_tracks("medium")
    if Path(duplicates_path).exists():
        removed_df = pd.read_csv(duplicates_path, dtype={"kept_id": "Int64"})
        removed = {int(t): (None if pd.isna(k) else int(k)) for t, k in zip(removed_df["track_id"], removed_df["kept_id"])}
    else:
        logger.warning("%s missing -- run `python -m modeling.catalogue`; re-uploads will not be collapsed",
                       duplicates_path)
        removed = {}
    dropped = np.array(list(removed), dtype=np.int64)
    keep = ~np.isin(ids, dropped) & np.isin(ids, tracks.index)
    ids, effnet, clap = ids[keep], effnet[keep], clap[keep]
    logger.info("Catalogue: %s tracks (%s re-uploads and silent tracks removed)", len(ids), len(dropped))
    return Catalogue(tracks=tracks.loc[ids], ids=ids, effnet=effnet, clap=clap, removed=removed)


def _main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ids, effnet, clap = _aligned()
    tracks = load_tracks("medium")
    dupes = find_duplicates(ids, effnet, clap, tracks)
    DUPLICATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dupes.to_csv(DUPLICATES_PATH, index=False)
    logger.info("%s tracks removed -> %s\n%s", len(dupes), DUPLICATES_PATH, dupes["reason"].value_counts().to_string())


if __name__ == "__main__":
    _main()
