"""Loading the FMA metadata for the music recommender.

Paths are resolved relative to this file, not the current working directory, so
the same code works from a notebook in notebooks/, from a script at the repo
root, or from anywhere else.
"""

from logging import getLogger
from pathlib import Path

import pandas as pd

logger = getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fma_metadata"


def load_tracks(subset: str = "small") -> pd.DataFrame:
    """Track metadata, optionally filtered to one of the FMA subsets.

    subset: "small", "medium", "large", or None for everything.
    """
    logger.info("Loading track metadata")
    tracks = pd.read_csv(DATA_DIR / "tracks.csv", index_col=0, header=[0, 1])
    if subset is not None:
        tracks = tracks[tracks[("set", "subset")] == subset]
    return tracks


def load_features(track_ids=None) -> pd.DataFrame:
    """Pre-extracted audio features (518 columns), rows with NaNs dropped."""
    logger.info("Loading audio features (this file is ~900MB)")
    features = pd.read_csv(DATA_DIR / "features.csv", index_col=0, header=[0, 1, 2])
    if track_ids is not None:
        features = features.loc[track_ids]
    return features.dropna()


def load_fma(subset: str = "small"):
    """Features and matching metadata for one subset, aligned on track id.

    Returns (X, tracks) where X has one row per track and 518 feature columns,
    and tracks is restricted to exactly the tracks that survived in X.
    """
    tracks = load_tracks(subset)
    X = load_features(tracks.index)
    logger.info("Loaded %s tracks with %s features", X.shape[0], X.shape[1])
    return X, tracks.loc[X.index]


def genre_labels(tracks: pd.DataFrame) -> pd.Series:
    """Top-level genre per track — handy for colouring plots and sanity checks."""
    return tracks[("track", "genre_top")]
