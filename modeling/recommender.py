"""The baseline recommender: nearest neighbours in the PCA feature space.

PCA on its own recommends nothing — it only reshapes the features. This is the
model: index every track by its position in the reduced space, and answer "what
sounds like this?" with the closest tracks to it.

Baseline by design. No genre filtering, no popularity weighting, no learned
similarity — just distance. Everything else gets measured against it.

    # build it and see what it recommends
    uv run python -m modeling.recommender --track-id 2

    # the same seed, further out
    uv run python -m modeling.recommender --track-id 2 --adventure 0.7

    # score it
    uv run python -m modeling.recommender --evaluate
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from modeling.features import MODELS_DIR, transform

logger = getattr(logging, "getLogger")(__name__)

DEFAULT_N = 10

# How deep into the ranked neighbour list the adventure slider can reach. Rank
# 500 of 8,000 is still the top 6% by similarity — far from the seed, but never
# random, which is the whole point of steering rather than shuffling.
MAX_DEPTH = 500


def depth_for(adventure: float, max_depth: int = MAX_DEPTH) -> int:
    """Where to start reading the neighbour list, for an adventure of 0…1.

    The mapping is geometric, not linear, because distance grows slowly with
    rank: on a typical seed, ranks 1-5 sit at cosine 0.42-0.48, ranks 31-35 at
    0.55, ranks 201-205 at 0.66. A linear slider would spend most of its travel
    inside a band the listener cannot hear apart. This one spends half its
    travel inside the first 22 neighbours. Returned as a 0-based offset, so the
    window it opens starts at these neighbours, counting from 1:

        0.00 →   1st     the nearest tracks, as the baseline behaves
        0.25 →   5th
        0.50 →  22nd
        0.75 → 106th
        1.00 → 500th
    """
    if not 0.0 <= adventure <= 1.0:
        raise ValueError(f"adventure must be between 0 and 1, got {adventure}")
    return int(round(max_depth ** adventure)) - 1


class TrackRecommender:
    """Content-based nearest-neighbour recommender over a PCA feature space.

    metric="cosine" compares the direction of two tracks' feature vectors and
    ignores their magnitude, which suits audio descriptors better than raw
    euclidean distance — loudness shouldn't decide similarity.
    """

    def __init__(self, scaler, pca, metric: str = "cosine"):
        self.scaler = scaler
        self.pca = pca
        self.metric = metric
        self.index_ = None
        self.track_ids_ = None
        self.tracks_ = None

    def fit(self, X: pd.DataFrame, tracks: pd.DataFrame = None):
        """Index the catalogue. X is raw features; scaling and PCA are applied here."""
        Z = transform(X, self.scaler, self.pca)
        self.track_ids_ = np.asarray(X.index)
        self.index_ = NearestNeighbors(metric=self.metric).fit(Z)
        self.embedding_ = Z
        self.tracks_ = tracks
        logger.info("Indexed %s tracks in %s dimensions", Z.shape[0], Z.shape[1])
        return self

    def _position(self, track_id) -> int:
        matches = np.flatnonzero(self.track_ids_ == track_id)
        if matches.size == 0:
            raise KeyError(f"track {track_id} is not in the indexed catalogue")
        return int(matches[0])

    def _with_metadata(self, track_ids, distances, ranks) -> pd.DataFrame:
        out = pd.DataFrame({
            "track_id": track_ids,
            "rank": ranks,
            "distance": distances,
        }).set_index("track_id")

        if self.tracks_ is not None:
            meta = self.tracks_.loc[out.index]
            out["artist"] = meta[("artist", "name")].values
            out["title"] = meta[("track", "title")].values
            out["genre"] = meta[("track", "genre_top")].values
        return out

    def similar_to(self, track_id, n: int = DEFAULT_N, adventure: float = 0.0) -> pd.DataFrame:
        """n tracks near `track_id`, read from a window of the ranked neighbour list.

        The model already ranks the entire catalogue by distance; `adventure`
        only decides where in that ranking the window of n starts. At 0 it is
        the ten nearest tracks — the baseline, unchanged. Higher values slide
        the window outward into music that is still sorted by similarity but no
        longer the obvious answer.

        This is the Familiar → Adventurous control, and it is deliberately not
        randomisation: rank 200 of 8,000 is a considered recommendation that
        happens to be far away, whereas a random track is noise the listener
        learns to distrust.
        """
        row = self._position(track_id)
        start = depth_for(adventure)
        # +1 for the seed itself, which is always its own nearest neighbour
        distances, idx = self.index_.kneighbors(
            self.embedding_.iloc[[row]], n_neighbors=start + n + 1
        )
        distances, idx = distances[0][1:], idx[0][1:]  # drop the seed
        window = slice(start, start + n)
        return self._with_metadata(
            self.track_ids_[idx[window]],
            distances[window],
            np.arange(start + 1, start + n + 1),
        )

    def taste_profile(self, track_ids) -> np.ndarray:
        """One point in the PCA space standing for a whole list of seed tracks.

        The centroid — the average position of the seeds. With cosine distance
        this reads as the average *direction* of the taste, so a list of three
        hip-hop tracks lands firmly in hip-hop territory.

        The interesting case is a deliberately mixed list: the centroid falls
        between the styles, which may be a genuine bridge between them or may
        be empty space resembling neither. Worth checking before building the
        Taste Bridge on it.
        """
        rows = [self._position(t) for t in track_ids]
        return self.embedding_.iloc[rows].mean(axis=0).to_numpy()

    def similar_to_many(self, track_ids, n: int = DEFAULT_N,
                        adventure: float = 0.0) -> pd.DataFrame:
        """Recommend for a list of seed tracks instead of a single one.

        Seeds are excluded from the result — recommending a track back to
        someone who just named it is not a recommendation. `adventure` slides
        the window exactly as in similar_to, but around the taste centroid.
        """
        track_ids = list(track_ids)
        # as a one-row frame, not a bare array: the index was fitted on named
        # columns, and sklearn warns when a query arrives without them
        centroid = pd.DataFrame([self.taste_profile(track_ids)],
                                columns=self.embedding_.columns)
        start = depth_for(adventure)

        # ask for extra so there is room to drop the seeds afterwards
        distances, idx = self.index_.kneighbors(
            centroid, n_neighbors=start + n + len(track_ids)
        )
        distances, idx = distances[0], idx[0]

        seeds = set(track_ids)
        keep = [(d, i) for d, i in zip(distances, idx) if self.track_ids_[i] not in seeds]
        keep = keep[start:start + n]

        return self._with_metadata(
            [self.track_ids_[i] for _, i in keep],
            [d for d, _ in keep],
            np.arange(start + 1, start + len(keep) + 1),
        )

    def describe(self, track_id) -> str:
        if self.tracks_ is None:
            return str(track_id)
        row = self.tracks_.loc[track_id]
        return f"{row[('artist', 'name')]} — {row[('track', 'title')]} [{row[('track', 'genre_top')]}]"

    def save(self, path: Path = MODELS_DIR / "recommender.pkl") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved recommender to %s", path)
        return path


def load(path: Path = MODELS_DIR / "recommender.pkl") -> TrackRecommender:
    return joblib.load(Path(path))


def precision_at_k(recommender: TrackRecommender, labels: pd.Series, k: int = DEFAULT_N,
                   sample: int = None, seed: int = 42) -> float:
    """Share of recommended tracks sharing the seed track's genre.

    This is the recommender-level version of the metric in modeling.evaluation —
    same idea, but measured through the actual recommend path (cosine distance,
    the fitted index) rather than on the raw embedding.
    """
    positions = np.arange(len(recommender.track_ids_))
    if sample is not None and sample < positions.size:
        positions = np.random.default_rng(seed).choice(positions, size=sample, replace=False)

    # one batched query rather than a Python loop of single lookups — the loop
    # version rescanned the whole catalogue per seed track and was minutes slower
    _, neighbours = recommender.index_.kneighbors(
        recommender.embedding_.iloc[positions], n_neighbors=k + 1
    )
    neighbours = neighbours[:, 1:]  # drop each track itself

    lab = np.asarray(labels.astype(str).loc[recommender.track_ids_])
    return float((lab[neighbours] == lab[positions][:, None]).mean())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m modeling.recommender",
        description="Baseline nearest-neighbour recommender over the PCA feature space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--track-id", type=int, default=None,
                        help="show recommendations for this track (default: a random one)")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help="how many tracks to recommend")
    parser.add_argument("--adventure", type=float, default=0.0,
                        help="0 = the nearest tracks, 1 = the edge of the top 6%%")
    parser.add_argument("--depth-profile", action="store_true",
                        help="score the recommender at every adventure level")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "manhattan"],
                        help="distance used for similarity")
    parser.add_argument("--pca-suffix", default="_90pct",
                        help="which saved scaler/pca pair to load from models/")
    parser.add_argument("--subset", default="small", help="FMA subset to index")
    parser.add_argument("--evaluate", action="store_true",
                        help="score precision@k over a sample of the catalogue")
    parser.add_argument("--sample", type=int, default=1000,
                        help="tracks sampled when scoring with --evaluate")
    parser.add_argument("--save", action="store_true", help="persist the fitted recommender")
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
    args = parse_args(argv)

    from modeling.data import genre_labels, load_fma
    from modeling.features import load_artifacts

    X, tracks = load_fma(subset=args.subset)
    scaler, pca = load_artifacts(suffix=args.pca_suffix)
    rec = TrackRecommender(scaler, pca, metric=args.metric).fit(X, tracks)

    track_id = args.track_id
    if track_id is None:
        track_id = int(np.random.default_rng(42).choice(rec.track_ids_))

    print(f"\nseed track : {rec.describe(track_id)}")
    print(f"space      : {rec.embedding_.shape[1]} components, {args.metric} distance")
    print(f"adventure  : {args.adventure:.2f} — window starts at neighbour "
          f"{depth_for(args.adventure) + 1}\n")
    recs = rec.similar_to(track_id, n=args.n, adventure=args.adventure)
    print(recs.to_string(float_format=lambda v: f"{v:.4f}"))

    if args.depth_profile:
        from modeling.evaluation import depth_profile, genre_sets

        print("\n=== relevance against adventure ===")
        profile = depth_profile(
            rec.embedding_, genre_labels(tracks), genre_sets_=genre_sets(tracks),
            k=args.n, metric=args.metric, sample=args.sample,
        )
        print(profile.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.evaluate:
        labels = genre_labels(tracks)
        score = precision_at_k(rec, labels, k=args.n, sample=args.sample)
        print(f"\nprecision@{args.n} over {args.sample} tracks: {score:.3f}"
              f"   (random = {1 / labels.nunique():.3f})")

    if args.save:
        rec.save()


if __name__ == "__main__":
    main()
