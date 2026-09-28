"""Rhythm features, extracted from the audio itself.

FMA ships 518 pre-computed features and not one of them describes rhythm. The
eleven families are spectral (mfcc, spectral_*, zcr, rmse) or harmonic
(chroma_*, tonnetz); tempo, pulse and note density are simply absent.

That is a hole worth filling, because listeners keep pointing at it. In the
first two listening-test returns, `rhythm` was the second most cited reason two
tracks sound related — 16 mentions against 17 for `mood` — on pairs the raters
scored 4.12 out of 5 on average. The model cannot currently see the thing
people say they are hearing.

This module computes that missing family from the mp3s and writes it in the
same shape as `features.csv`, so the two concatenate:

    columns  : MultiIndex (family, statistic, number), as FMA writes them
    index    : track_id
    families : tempo, onset_rate, onset_strength, pulse_clarity

Run it once; the result is a small csv that loads in a second.

    uv run python -m modeling.rhythm                 # all 8,000, every core
    uv run python -m modeling.rhythm --limit 50      # a quick smoke test
"""

import argparse
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Read from an env var, not a CLI arg passed straight to extract_catalogue:
# ProcessPoolExecutor workers are spawned fresh and re-import this module, so a
# runtime-assigned Python variable would not reach them, but an env var will.
AUDIO_DIR = Path(os.environ.get("FMA_AUDIO_DIR", REPO_ROOT / "data" / "fma_small"))
OUT_PATH = Path(os.environ.get("FMA_RHYTHM_OUT", REPO_ROOT / "data" / "rhythm_features.csv"))

# The rate librosa resamples to. 22050 Hz is well above anything rhythmic and
# halves the decode cost against the mp3s' native 44100.
SAMPLE_RATE = 22050

# FMA summarises every feature with these seven, in this order. Matching it
# means the new columns sit alongside the old ones without translation.
STATISTICS = ["mean", "std", "skew", "kurtosis", "median", "min", "max"]


def audio_path(track_id: int) -> Path:
    """FMA stores 000123.mp3 under a directory named after its first 3 digits."""
    name = f"{int(track_id):06d}"
    return AUDIO_DIR / name[:3] / f"{name}.mp3"


def _summarise(x: np.ndarray) -> list:
    """The seven FMA statistics of one signal."""
    return [
        float(np.mean(x)), float(np.std(x)), float(stats.skew(x)),
        float(stats.kurtosis(x)), float(np.median(x)),
        float(np.min(x)), float(np.max(x)),
    ]


def extract(track_id: int) -> dict | None:
    """Rhythm descriptors for one track, or None if the file will not decode.

    A handful of FMA clips are truncated or empty. Returning None rather than
    raising keeps one bad file from ending a four-hour run.
    """
    # librosa is imported inside the worker: it pulls in numba, which is slow to
    # import and pointless in the parent process that only collects results.
    import librosa

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path(track_id), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 — any decode failure is the same to us
        logger.warning("track %s did not decode: %s", track_id, exc)
        return None

    if y.size < sr:  # under a second of audio is not a rhythm
        logger.warning("track %s is too short (%.2fs)", track_id, y.size / sr)
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The onset envelope is the shared basis: a signal that peaks wherever a
        # note or hit begins. Everything below is a different reading of it.
        onset = librosa.onset.onset_strength(y=y, sr=sr)

        # A flat envelope means silence, or a drone with no note beginnings at
        # all. There is no rhythm to describe, and skew and kurtosis of a
        # constant signal are NaN — so drop the track rather than write one in
        # that every downstream scaler would then have to special-case.
        if float(np.std(onset)) < 1e-9:
            logger.warning("track %s has a flat onset envelope — no rhythm", track_id)
            return None

        # Beats per minute, from the envelope's dominant periodicity.
        tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=sr))[0])

        # How many note onsets per second — note density, which separates a
        # sparse ambient piece from a busy one at the same tempo.
        onsets = librosa.onset.onset_detect(onset_envelope=onset, sr=sr)
        onset_rate = float(len(onsets) / (y.size / sr))

        # Predominant local pulse: how strongly the track actually keeps the
        # beat it is nominally at. A metronomic track scores high, rubato low.
        # This is what separates "same tempo" from "same groove".
        pulse = librosa.beat.plp(onset_envelope=onset, sr=sr)

    return {
        "track_id": int(track_id),
        "tempo": [tempo],
        "onset_rate": [onset_rate],
        "onset_strength": _summarise(onset),
        "pulse_clarity": _summarise(pulse),
    }


def _columns() -> pd.MultiIndex:
    """FMA's (family, statistic, number) column index for what extract returns."""
    cols = [("tempo", "mean", "01"), ("onset_rate", "mean", "01")]
    for family in ("onset_strength", "pulse_clarity"):
        cols += [(family, stat, "01") for stat in STATISTICS]
    return pd.MultiIndex.from_tuples(cols, names=["feature", "statistic", "number"])


def extract_catalogue(track_ids, workers: int | None = None) -> pd.DataFrame:
    """Every track's rhythm features, extracted in parallel.

    Decoding is CPU-bound and independent per track, so this scales with cores.
    Tracks that fail to decode are dropped, exactly as `load_features` drops
    rows with NaNs — the recommender never sees a partially described track.
    """
    track_ids = [int(t) for t in track_ids]
    rows, failed = [], []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(extract, track_ids, chunksize=16), start=1):
            if result is None:
                failed.append(track_ids[i - 1])
            else:
                rows.append(result)
            if i % 250 == 0:
                logger.info("%s / %s tracks", i, len(track_ids))

    if failed:
        logger.warning("%s tracks failed to decode: %s", len(failed), failed[:10])

    data = [r["tempo"] + r["onset_rate"] + r["onset_strength"] + r["pulse_clarity"]
            for r in rows]
    out = pd.DataFrame(
        data,
        index=pd.Index([r["track_id"] for r in rows], name="track_id"),
        columns=_columns(),
    ).sort_index()

    # librosa normalises the pulse curve to [0, 1] per track, so its min and max
    # are the same number for every track in the catalogue. A column with no
    # variance carries no information and makes StandardScaler divide by zero.
    constant = out.columns[out.std() < 1e-9]
    if len(constant):
        logger.info("Dropping %s constant columns: %s", len(constant), list(constant))
        out = out.drop(columns=constant)
    return out


def load(path: Path = OUT_PATH) -> pd.DataFrame:
    """Read back what this module wrote, with the MultiIndex intact."""
    return pd.read_csv(path, index_col=0, header=[0, 1, 2])


def _main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only the first N tracks")
    parser.add_argument("--workers", type=int, help="processes (default: every core)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--subset", default="small", choices=["small", "medium", "large"],
                         help="cumulative FMA subset (medium includes small; needs "
                              "FMA_AUDIO_DIR pointed at a directory that has those mp3s)")
    args = parser.parse_args()

    from modeling.data import load_tracks

    track_ids = load_tracks(args.subset).index
    if args.limit:
        track_ids = track_ids[: args.limit]

    logger.info("Extracting rhythm features for %s tracks", len(track_ids))
    features = extract_catalogue(track_ids, workers=args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out)
    logger.info("Wrote %s tracks x %s columns to %s",
                features.shape[0], features.shape[1], args.out)


if __name__ == "__main__":
    _main()
