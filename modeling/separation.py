"""Harmonic-percussive features: what is making the sound, not just how it sounds.

None of the 518 FMA features, and none of the rhythm features in `rhythm.py`,
say whether a track is carried by drums or by instruments. MFCC describes
texture, chroma describes pitch, rhythm describes timing — but a solo guitar and
a drum loop can land close on all three.

`librosa.effects.hpss` splits the audio on a geometric fact about spectrograms:
a sustained note draws a horizontal line (one frequency, held), and a hit draws
a vertical one (every frequency, for an instant). Median-filtering along each
axis separates them into two signals.

Three families come out of that split:

    hpss_ratio       1 column   log energy ratio, percussive over harmonic
    mfcc_harmonic   40 columns  the timbre of the instruments alone
    mfcc_percussive 40 columns  the timbre of the drums alone

On a 320-track genre-balanced sample the ratio scored eta^2 = 0.143 against
genre — better than every rhythm column but the top four, and from one column.
The two split MFCC blocks are the untested half: the idea is that "bright drums
over a dark pad" is a description the unsplit MFCC cannot give, because it only
ever sees the two mixed together.

Raters named `instrumentation` seven times in the first two returns. This is the
nearest thing to it that can be computed.

    uv run python -m modeling.separation                 # all 8,000, every core
    uv run python -m modeling.separation --limit 50      # a quick smoke test
"""

import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.rhythm import REPO_ROOT, SAMPLE_RATE, audio_path

logger = logging.getLogger(__name__)

OUT_PATH = REPO_ROOT / "data" / "separation_features.csv"

# Enough coefficients to describe a timbre, kept to mean and std rather than
# FMA's seven statistics: this is a hypothesis being tested, and 7 x 20 x 2
# would put 280 unproven columns next to 153 measured ones.
N_MFCC = 20
STATISTICS = ["mean", "std"]


def extract(track_id: int) -> dict | None:
    """Split one track and describe each half, or None if it will not decode."""
    import librosa

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path(track_id), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("track %s did not decode: %s", track_id, exc)
        return None

    if y.size < sr:
        logger.warning("track %s is too short (%.2fs)", track_id, y.size / sr)
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        harmonic, percussive = librosa.effects.hpss(y)

        h_energy = float(np.sum(harmonic ** 2))
        p_energy = float(np.sum(percussive ** 2))
        if h_energy <= 0 or p_energy <= 0:
            logger.warning("track %s has an empty half after the split", track_id)
            return None

        # A ratio is multiplicative: four times more percussive and four times
        # less are the same distance from even. Raw, the tail runs past 20 and
        # swamps the scale — on the sample, the log lifted eta^2 from 0.038 to
        # 0.143. Take the log here so no downstream step has to remember to.
        ratio = float(np.log(p_energy / h_energy))

        out = {"track_id": int(track_id), "hpss_ratio": [ratio]}
        for name, part in (("mfcc_harmonic", harmonic), ("mfcc_percussive", percussive)):
            m = librosa.feature.mfcc(y=part, sr=sr, n_mfcc=N_MFCC)
            out[name] = [float(v) for v in m.mean(axis=1)] + [float(v) for v in m.std(axis=1)]

    if any(not np.isfinite(v) for values in out.values() if isinstance(values, list) for v in values):
        logger.warning("track %s produced a non-finite value", track_id)
        return None
    return out


def _columns() -> pd.MultiIndex:
    """FMA's (family, statistic, number) index, so these sit beside the rest."""
    cols = [("hpss_ratio", "mean", "01")]
    for family in ("mfcc_harmonic", "mfcc_percussive"):
        for stat in STATISTICS:
            cols += [(family, stat, f"{i:02d}") for i in range(1, N_MFCC + 1)]
    return pd.MultiIndex.from_tuples(cols, names=["feature", "statistic", "number"])


def extract_catalogue(track_ids, workers: int | None = None) -> pd.DataFrame:
    """Every track, in parallel. Tracks that fail to decode are dropped."""
    track_ids = [int(t) for t in track_ids]
    rows, failed = [], []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(extract, track_ids, chunksize=8), start=1):
            if result is None:
                failed.append(track_ids[i - 1])
            else:
                rows.append(result)
            if i % 250 == 0:
                logger.info("%s / %s tracks", i, len(track_ids))

    if failed:
        logger.warning("%s tracks failed: %s", len(failed), failed[:10])

    data = [r["hpss_ratio"] + r["mfcc_harmonic"] + r["mfcc_percussive"] for r in rows]
    out = pd.DataFrame(
        data,
        index=pd.Index([r["track_id"] for r in rows], name="track_id"),
        columns=_columns(),
    ).sort_index()

    constant = out.columns[out.std() < 1e-9]
    if len(constant):
        logger.info("Dropping %s constant columns", len(constant))
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
    args = parser.parse_args()

    from modeling.data import load_tracks

    track_ids = load_tracks("small").index
    if args.limit:
        track_ids = track_ids[: args.limit]

    logger.info("Splitting %s tracks — this is the slow one", len(track_ids))
    features = extract_catalogue(track_ids, workers=args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out)
    logger.info("Wrote %s tracks x %s columns to %s",
                features.shape[0], features.shape[1], args.out)


if __name__ == "__main__":
    _main()
