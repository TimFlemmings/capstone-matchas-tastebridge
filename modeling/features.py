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
