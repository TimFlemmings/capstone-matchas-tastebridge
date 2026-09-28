"""The 'heard' feature space (mfcc + rhythm + hpss_ratio) for a track that
never went through FMA's own pipeline — a Spotify or YouTube download, say.

Two different problems, handled two different ways:

  rhythm and hpss_ratio are ours: `rhythm.py` and `separation.py` already
  define exactly how they're computed. Rather than re-implement that logic
  (and risk it drifting from the catalogue), this module stages the external
  file at a throwaway track id inside data/fma_small/ and calls `rhythm.extract`
  and `separation.extract` unchanged — the same code path every catalogue
  track went through.

  mfcc is FMA's: `features.csv` was built by a script we don't own
  (github.com/mdeff/fma/blob/master/features.py). `extract_mfcc` below is that
  script's mfcc block, copied verbatim. It was checked against track 2's real
  features.csv row and correlates 0.97-0.99 on mean/std/median/min/max, but
  only 0.73-0.90 on skew/kurtosis — FMA's features.csv predates librosa's
  `rmse` -> `rms` rename, so it was built on a materially older librosa whose
  mel-filterbank defaults have since drifted. This is the closest reproduction
  possible without that exact old environment; treat mfcc distances for an
  external track as approximate, not exact, and skew/kurtosis-heavy
  conclusions with particular caution.

Usage:

    from modeling.external import extract_heard_features
    row = extract_heard_features("data/raw/spotify_tests/Y2Mate.is - Elyjah Slaps the Space.mp3")
"""

import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats

from modeling.rhythm import STATISTICS as RHYTHM_STATISTICS
from modeling.rhythm import audio_path
from modeling.rhythm import extract as rhythm_extract
from modeling.separation import extract as separation_extract

# Keeps a staged file from ever colliding with a real FMA-small id (max ~155000).
PLACEHOLDER_ID = 990_000_001

MFCC_STATISTICS = ["mean", "std", "skew", "kurtosis", "median", "min", "max"]


def extract_mfcc(path) -> dict:
    """FMA's mfcc block (features.py, mdeff/fma), reproduced parameter-for-parameter.

    Returns {"mfcc": [140 values]}, coefficient-major then statistic
    (coef 1's 7 stats, then coef 2's, ...) — extract_heard_features lines this
    up against the real columns rather than assuming an order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, sr = librosa.load(path, sr=None, mono=True)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

    values = []
    for coef in mfcc:  # 20 rows, one per coefficient, each a time series
        values += [
            float(np.mean(coef)), float(np.std(coef)), float(stats.skew(coef)),
            float(stats.kurtosis(coef)), float(np.median(coef)),
            float(np.min(coef)), float(np.max(coef)),
        ]
    return {"mfcc": values}


def _mfcc_series(path) -> pd.Series:
    """Indexed exactly like FMA's own mfcc columns: (feature, statistic, number)."""
    values = extract_mfcc(path)["mfcc"]
    index = pd.MultiIndex.from_tuples(
        [("mfcc", stat, f"{i+1:02d}") for i in range(20) for stat in MFCC_STATISTICS]
    )
    return pd.Series(values, index=index)


def _stage(path: Path, track_id: int) -> Path:
    dest = audio_path(track_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(Path(path).resolve())
    return dest


def extract_heard_features(path, heard_columns: pd.Index, track_id: int = PLACEHOLDER_ID) -> pd.Series:
    """One row in the fitted 'heard' space's raw-feature columns, for a file outside FMA.

    `heard_columns` should be `full_table(...)`'s columns restricted to the
    "heard" set (see modeling.features.space("heard")) — i.e. the exact column
    index the scaler/PCA were fit on. Passing it in, rather than hard-coding a
    layout, is what lets this stay correct if the catalogue ever drops a
    different constant column.
    """
    path = Path(path)
    dest = _stage(path, track_id)
    try:
        rhythm_row = rhythm_extract(track_id)
        sep_row = separation_extract(track_id)
    finally:
        dest.unlink()

    if rhythm_row is None:
        raise ValueError(f"{path}: rhythm extraction failed (too short, or a flat onset envelope)")
    if sep_row is None:
        raise ValueError(f"{path}: harmonic/percussive separation failed (too short, or an empty half)")

    rhythm_index = pd.MultiIndex.from_tuples(
        [("tempo", "mean", "01"), ("onset_rate", "mean", "01")]
        + [(family, stat, "01") for family in ("onset_strength", "pulse_clarity")
           for stat in RHYTHM_STATISTICS]
    )
    rhythm_series = pd.Series(
        rhythm_row["tempo"] + rhythm_row["onset_rate"] + rhythm_row["onset_strength"] + rhythm_row["pulse_clarity"],
        index=rhythm_index,
    )
    hpss_series = pd.Series(sep_row["hpss_ratio"], index=pd.MultiIndex.from_tuples([("hpss_ratio", "mean", "01")]))
    mfcc_series = _mfcc_series(path)

    flat = pd.concat([mfcc_series, rhythm_series, hpss_series])
    flat.index = flat.index.set_names(["feature", "statistic", "number"])

    missing = heard_columns[~heard_columns.isin(flat.index)]
    if len(missing):
        raise KeyError(f"extraction did not produce: {list(missing)}")

    out = flat.reindex(heard_columns)
    out.name = f"external:{path.name}"
    return out
