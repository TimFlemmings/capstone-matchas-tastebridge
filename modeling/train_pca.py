"""Fit the PCA feature space for the recommender and log the run to MLflow.

Every option here maps to one of the proposals in the feature-space review, so
a proposal can be tested from the command line instead of by editing a notebook:

    # the finalised space — timbre families, 90% of the variance
    uv run python -m modeling.train_pca

    # the original all-518 baseline, for comparison
    uv run python -m modeling.train_pca --feature-set all

    # proposal 1: a fixed, smaller space
    uv run python -m modeling.train_pca --n-components 60

    # proposal 2: a scaler that outliers pull on less
    uv run python -m modeling.train_pca --n-components 60 --scaler robust

    # proposal 3: stop letting chroma carry half the feature set
    uv run python -m modeling.train_pca --n-components 60 --drop-chroma

`--feature-set` defaults to "timbre", not to all 518 columns: the harmony block
is 57% of the features and scores 2.22x random against timbre's 3.66x, so
carrying it dilutes the space rather than widening it. modeling.features records
the measurement. Artifact names carry the set, so runs at different feature sets
do not overwrite each other — and an older recommender loading `_90pct` still
finds the all-518 artifacts it was fitted with.

Runs land in the Music_Recommendation_PCA experiment, all carrying the
neighbour_genre_agreement metric so they can be compared against each other
whatever was changed. That score is measured with cosine distance, matching
TrackRecommender — the distance is logged as a parameter because it changes not
just the numbers but which configuration wins.
"""

import argparse
import logging

import mlflow

from modeling.config import MUSIC_EXPERIMENT_NAME, TRACKING_URI
from modeling.data import genre_labels, load_fma
from modeling.evaluation import K_NEIGHBOURS, neighbour_genre_agreement, random_baseline
from modeling.features import (
    FEATURE_SETS,
    SCALERS,
    drop_families,
    drop_statistics,
    fit_pca,
    pca_tag,
    save_artifacts,
    scale_features,
    space,
    transform,
)

logger = logging.getLogger(__name__)

# the two chroma variants dropped by --drop-chroma; cens is kept as it generally
# behaves best for similarity work
REDUNDANT_CHROMA = ["chroma_cqt", "chroma_stft"]

# heavy-tailed by construction, 74 columns each
SHAPE_STATISTICS = ["kurtosis", "skew"]


def n_components_type(value: str):
    """A variance target if written with a decimal point, else a fixed count.

    0.90 -> keep enough components for 90% of the variance
    60   -> keep exactly 60 components
    """
    number = float(value)
    if "." in value or "e" in value.lower():
        if not 0 < number <= 1:
            raise argparse.ArgumentTypeError("a variance target must be between 0 and 1")
        return number
    return int(number)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m modeling.train_pca",
        description="Fit and log the PCA feature space for the music recommender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-components",
        type=n_components_type,
        default=0.90,
        metavar="N",
        help="0.90 for a variance target, 60 for a fixed component count",
    )
    parser.add_argument(
        "--feature-set", choices=sorted(FEATURE_SETS), default="timbre",
        help="which feature families go in; see modeling.features.FEATURE_SETS",
    )
    parser.add_argument(
        "--scaler", choices=sorted(SCALERS), default="standard",
        help="how features are scaled before PCA",
    )
    parser.add_argument(
        "--subset", default="small",
        help="which FMA subset to load (small, medium, large)",
    )
    parser.add_argument(
        "--drop-chroma", action="store_true",
        help=f"drop {', '.join(REDUNDANT_CHROMA)}, keeping chroma_cens only",
    )
    parser.add_argument(
        "--drop-shape-stats", action="store_true",
        help=f"drop the {' and '.join(SHAPE_STATISTICS)} columns (74 each)",
    )
    parser.add_argument(
        "--k", type=int, default=K_NEIGHBOURS, metavar="K",
        help="neighbours used by the quality metric",
    )
    parser.add_argument(
        "--metric", default="cosine", choices=["cosine", "euclidean", "manhattan"],
        help="distance the quality score is measured with; cosine matches the recommender",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="skip the neighbour-quality metric (faster, but the run can't be compared)",
    )
    parser.add_argument("--run-name", default=None, help="override the generated MLflow run name")
    return parser.parse_args(argv)


def build_run_name(args) -> str:
    parts = [f"PCA_{pca_tag(args.n_components)}", args.feature_set]
    if args.scaler != "standard":
        parts.append(args.scaler)
    if args.drop_chroma:
        parts.append("nochroma")
    if args.drop_shape_stats:
        parts.append("noshapestats")
    return "_".join(parts)


def run(args):
    X, tracks = load_fma(subset=args.subset)
    labels = genre_labels(tracks)

    # space(), not feature_set(): the shipped sets need the rhythm and drum
    # columns we extracted ourselves, and only space() joins those files in.
    X = space(args.feature_set, X)
    if args.drop_chroma:
        X = drop_families(X, REDUNDANT_CHROMA)
    if args.drop_shape_stats:
        X = drop_statistics(X, SHAPE_STATISTICS)

    X_scaled, scaler = scale_features(X, method=args.scaler)
    pca = fit_pca(X_scaled, n_components=args.n_components)

    quality = None
    if not args.no_eval:
        logger.info("Scoring neighbourhood quality (k=%s)", args.k)
        quality = neighbour_genre_agreement(transform(X, scaler, pca), labels,
                                            k=args.k, metric=args.metric)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MUSIC_EXPERIMENT_NAME)
    run_name = args.run_name or build_run_name(args)
    tag = pca_tag(args.n_components)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset": f"fma_{args.subset}",
            "feature_set": args.feature_set,
            "original_features_count": int(X.shape[1]),
            "n_samples": int(X.shape[0]),
            "scaler": args.scaler,
            "pca_target": args.n_components,
            "dropped_chroma": args.drop_chroma,
            "dropped_shape_stats": args.drop_shape_stats,
            # logged because runs scored under different distances are not
            # comparable — cosine and euclidean rank the chroma experiments
            # in opposite orders
            "eval_metric": args.metric,
        })
        mlflow.log_metrics({
            "n_components_selected": pca.n_components_,
            "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        })
        if quality is not None:
            mlflow.log_metrics({
                "neighbour_genre_agreement": quality,
                "random_baseline": random_baseline(labels),
            })

        suffix = f"_{tag}_{args.feature_set}" + ("_robust" if args.scaler == "robust" else "")
        suffix += "_nochroma" if args.drop_chroma else ""
        suffix += "_noshapestats" if args.drop_shape_stats else ""
        scaler_path, pca_path = save_artifacts(scaler, pca, suffix=suffix)
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(pca_path))

        mlflow.set_tag("stage", "feature-space")

        print(f"\nrun            : {run_name}")
        print(f"features in    : {X.shape[1]}")
        print(f"components out : {pca.n_components_}")
        print(f"variance kept  : {pca.explained_variance_ratio_.sum():.3f}")
        if quality is not None:
            print(f"quality (k={args.k}) : {quality:.3f}   (random = {random_baseline(labels):.3f})")
        else:
            print("quality        : skipped (--no-eval)")
        print(f"artifacts      : {scaler_path.name}, {pca_path.name}")


def main(argv=None):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
    run(parse_args(argv))


if __name__ == "__main__":
    main()
