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
# Families we extracted ourselves because FMA does not ship them. rhythm.py and
# separation.py write them; they are not in features.csv and cannot be derived
# from it. See 04_new_features.ipynb for what each column measures.
RHYTHM = ["tempo", "onset_rate", "onset_strength", "pulse_clarity"]   # 13 columns
PERCUSSIVE = ["hpss_ratio"]                                           # 1 column

FEATURE_SETS = {
    "all": None,               # every column, the uncompressed baseline
    "timbre": TIMBRE,
    "harmony": HARMONY,
    "no_chroma": None,         # everything except the two redundant chroma variants
    # The two spaces the listening test left standing. They differ only in
    # whether the 84 non-mfcc timbre columns come along, and the two rulers
    # disagree about that: genre prefers "heard_wide", listeners "heard".
    "heard": ["mfcc"] + RHYTHM + PERCUSSIVE,          # 154 columns
    "heard_wide": TIMBRE + RHYTHM + PERCUSSIVE,       # 238 columns
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
        # A set naming a family this table does not have is a caller who handed
        # us FMA's 518 columns and asked for a space that needs the extracted
        # ones too. Selecting quietly would return mfcc alone and everything
        # downstream would look fine — Tim found exactly that: `train_pca
        # --feature-set heard` fitting on 140 columns while the recommender
        # asked for 154. Use space() for those; it joins the files first.
        wanted = FEATURE_SETS[name]
        absent = [f for f in wanted if f not in set(families)]
        if absent:
            raise ValueError(
                f"feature set {name!r} needs {', '.join(absent)}, which are not in "
                "this table — they come from rhythm.py and separation.py. Call "
                f"modeling.features.space({name!r}, X), which joins those files, "
                "rather than feature_set()."
            )
        keep = families.isin(wanted)

    logger.info("Feature set %r: %s -> %s columns", name, X.shape[1], int(keep.sum()))
    return X.loc[:, keep]


def full_table(X: pd.DataFrame) -> pd.DataFrame:
    """FMA's 518 columns beside the ones we extracted from the audio ourselves.

    features.csv is a table someone else computed, and rhythm and instrumentation
    are not in it. They were read back off the mp3s by rhythm.py and
    separation.py, which write their own files — so the recommender only sees
    them if something joins the three, which is what this does.

    Tracks missing from any of the three are dropped: seven of the 8,000 are
    files FMA ships truncated, or silent. A partly described track has no
    business in a distance calculation.
    """
    from modeling.rhythm import OUT_PATH as RHYTHM_PATH, load as load_rhythm
    from modeling.separation import OUT_PATH as SEP_PATH, load as load_separation

    blocks = [X]
    for path, loader, what in ((RHYTHM_PATH, load_rhythm, "rhythm"),
                               (SEP_PATH, load_separation, "separation")):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — run `uv run python -m modeling.{what}` first. "
                "It is gitignored, so a fresh checkout has to rebuild it."
            )
        blocks.append(loader())

    common = blocks[0].index
    for b in blocks[1:]:
        common = common.intersection(b.index)
    joined = pd.concat([b.loc[common] for b in blocks], axis=1)

    logger.info("Full table: %s tracks x %s columns (%s dropped as incomplete)",
                joined.shape[0], joined.shape[1], X.shape[0] - len(common))
    return joined


def space(name: str = "heard", X: pd.DataFrame = None) -> pd.DataFrame:
    """The named feature space, assembled from whichever files hold its columns.

    `feature_set` can only pick from what it is handed; this reaches for the
    extracted families too, so "heard" and "heard_wide" resolve to real columns
    rather than to silently empty selections.
    """
    if X is None:
        from modeling.data import load_fma
        X, _ = load_fma()
    wanted = FEATURE_SETS.get(name)
    if wanted is None:                      # "all" and "no_chroma" are FMA-only
        return feature_set(X, name)

    table = full_table(X) if set(wanted) - set(TIMBRE) - set(HARMONY) else X
    keep = table.columns.get_level_values(0).isin(wanted)
    out = table.loc[:, keep]
    logger.info("Space %r: %s columns", name, out.shape[1])
    return out


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


def load_artifacts(directory: Path = MODELS_DIR, suffix: str = "", space: str = None):
    """The fitted scaler and PCA for one space.

    `train_pca.py` names what it writes `_<variance tag>_<feature set>`, and the
    recommender asks for `_<feature set>`. Both spellings are in use, so both are
    tried before giving up — and giving up names the command that creates them,
    because they are gitignored and a fresh clone has none.
    """
    directory = Path(directory)
    tried = [suffix]
    if space is not None:
        tried += [f"_90pct_{space}", f"_{space}"]
    for candidate in dict.fromkeys(tried):          # keep order, drop repeats
        scaler = directory / f"scaler{candidate}.pkl"
        pca = directory / f"pca{candidate}.pkl"
        if scaler.exists() and pca.exists():
            logger.info("Loading artifacts %r from %s", candidate, directory)
            return joblib.load(scaler), joblib.load(pca)

    looked = ", ".join(f"scaler{c}.pkl" for c in dict.fromkeys(tried))
    raise FileNotFoundError(
        f"No fitted scaler and PCA in {directory} (looked for {looked}). "
        "They are gitignored, so a fresh clone has to fit them:\n"
        f"    uv run python -m modeling.train_pca --feature-set {space or 'heard'} "
        "--n-components 0.90"
    )
