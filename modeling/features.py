"""Scaling and dimensionality reduction for the FMA audio features."""

from logging import getLogger
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


# The 11 FMA feature families, grouped by what they measure. This is the single
# place the split is defined — import it rather than retyping a column list, so
# two people scoring two models are scoring the same columns.
TIMBRE = ["mfcc", "spectral_contrast", "spectral_centroid", "spectral_bandwidth",
          "spectral_rolloff", "zcr", "rmse"]      # 224 columns: texture and energy
HARMONY = ["chroma_cens", "chroma_cqt", "chroma_stft", "tonnetz"]   # 294: pitch content

# Measured, not assumed — neighbour genre agreement@10, cosine, fma_small:
#
#     space                        columns   agreement   vs random
#     timbre                           224       0.458       3.66x
#     all                              518       0.414       3.31x
#     harmony                          294       0.278       2.22x
#
# The harmony block is 57% of the feature set and the weakest part of it: every
# chroma family scores ~0.22 on its own, against 0.412 for mfcc alone. Carrying
# it does not add a second opinion, it dilutes the first one — dropping it is
# worth more than any amount of tuning downstream of it.
#
# "timbre" is therefore the default similarity space. Harmony is kept as a named
# set rather than deleted for two reasons: it is the natural divergence axis in
# the two-space design, and genre agreement is only a proxy for what a listener
# hears — key and mode may well carry perceived similarity that genre labels do
# not record. Tim's listening study is what settles that; until it does, the
# columns stay reachable.
FEATURE_SETS = {
    "all": None,               # every column, the uncompressed baseline
    "timbre": TIMBRE,
    "harmony": HARMONY,
    "no_chroma": None,         # everything except the two redundant chroma variants
}


def feature_set(X: pd.DataFrame, name: str = "timbre") -> pd.DataFrame:
    """The named subset of feature families. See FEATURE_SETS for the measurements."""
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r} — choose from {sorted(FEATURE_SETS)}")

    families = X.columns.get_level_values(0)
    if name == "all":
        return X
    if name == "no_chroma":
        keep = ~families.isin(["chroma_cqt", "chroma_stft"])
    else:
        keep = families.isin(FEATURE_SETS[name])

    logger.info("Feature set %r: %s -> %s columns", name, X.shape[1], int(keep.sum()))
    return X.loc[:, keep]


def scale_features(X: pd.DataFrame, method: str = "standard"):
    """Scale every feature column. Returns (X_scaled, fitted_scaler).

    "standard" centres on the mean and divides by the standard deviation —
    both of which the FMA features' heavy tails drag around (86 of 518 columns
    have kurtosis above 100, and the most extreme value still sits at z=88).
    "robust" uses the median and IQR instead, so outliers pull on it far less.
    """
    if method not in SCALERS:
        raise ValueError(f"unknown scaler {method!r} — choose from {sorted(SCALERS)}")
    scaler = SCALERS[method]().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    return X_scaled, scaler


def drop_families(X: pd.DataFrame, families) -> pd.DataFrame:
    """Drop whole feature families, e.g. ["chroma_cqt", "chroma_stft"].

    The three chroma_* families carry 252 of the 518 columns between them while
    measuring much the same thing, so they weight the similarity space heavily
    toward harmony.
    """
    families = list(families)
    keep = ~X.columns.get_level_values(0).isin(families)
    logger.info("Dropping families %s: %s -> %s columns", families, X.shape[1], int(keep.sum()))
    return X.loc[:, keep]


def drop_statistics(X: pd.DataFrame, statistics) -> pd.DataFrame:
    """Drop whole summary statistics, e.g. ["kurtosis", "skew"] (74 columns each)."""
    statistics = list(statistics)
    keep = ~X.columns.get_level_values(1).isin(statistics)
    logger.info("Dropping statistics %s: %s -> %s columns", statistics, X.shape[1], int(keep.sum()))
    return X.loc[:, keep]


def explained_variance_curve(X_scaled: pd.DataFrame) -> np.ndarray:
    """Cumulative explained variance for every possible number of components."""
    pca_full = PCA().fit(X_scaled)
    return np.cumsum(pca_full.explained_variance_ratio_)


def components_for_variance(cumulative_variance: np.ndarray, threshold: float = 0.90) -> int:
    """Smallest number of components reaching `threshold` of the total variance."""
    return int(np.argmax(cumulative_variance >= threshold) + 1)


def fit_pca(X_scaled: pd.DataFrame, n_components=0.90) -> PCA:
    """Fit PCA at a target size.

    n_components takes either form, and sklearn reads them differently:
      0.90 (float) -> keep however many components reach 90% of the variance
      60   (int)   -> keep exactly 60 components

    Always pass one of them. PCA() with no argument keeps all 518 components,
    which makes explained_variance_ratio_.sum() a trivial 1.0 and reduces
    nothing.
    """
    pca = PCA(n_components=n_components).fit(X_scaled)
    logger.info(
        "PCA kept %s components (%.1f%% of variance)",
        pca.n_components_,
        pca.explained_variance_ratio_.sum() * 100,
    )
    return pca


def transform(X: pd.DataFrame, scaler: StandardScaler, pca: PCA) -> pd.DataFrame:
    """Apply a fitted scaler + PCA to new tracks, keeping the track ids."""
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return pd.DataFrame(X_pca, index=X.index, columns=[f"pc{i}" for i in range(1, X_pca.shape[1] + 1)])


def build_pca_space(X: pd.DataFrame, n_components=0.90):
    """Full pipeline: scale, fit PCA, project. Returns (X_pca, scaler, pca).

    n_components takes a variance target (0.90) or a fixed count (60).
    """
    X_scaled, scaler = scale_features(X)
    pca = fit_pca(X_scaled, n_components=n_components)
    return transform(X, scaler, pca), scaler, pca


def pca_tag(n_components) -> str:
    """A short name for a PCA setting, for run names and artifact filenames.

    0.90 -> "90pct"    (a variance target)
    60   -> "60dims"   (a fixed component count)
    """
    if isinstance(n_components, float):
        return f"{n_components * 100:g}pct"
    return f"{n_components}dims"


def save_artifacts(
    scaler: StandardScaler, pca: PCA, directory: Path = MODELS_DIR, suffix: str = ""
):
    """Persist scaler and pca.

    `suffix` keeps runs at different variance targets from overwriting each
    other, e.g. suffix="_90" -> scaler_90.pkl / pca_90.pkl. Returns the two
    paths written, in (scaler, pca) order.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scaler_path = directory / f"scaler{suffix}.pkl"
    pca_path = directory / f"pca{suffix}.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    logger.info("Saved scaler and pca to %s", directory)
    return scaler_path, pca_path


def load_artifacts(directory: Path = MODELS_DIR, suffix: str = ""):
    directory = Path(directory)
    return (
        joblib.load(directory / f"scaler{suffix}.pkl"),
        joblib.load(directory / f"pca{suffix}.pkl"),
    )
