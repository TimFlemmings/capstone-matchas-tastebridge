"""Nearest-neighbour recommender over Discogs-EffNet embeddings.

Same shape as `modeling.recommender.TrackRecommender` — `similar_to`,
`taste_profile`, the same `adventure` dial via the same `depth_for()` — but
built on the 1280-dim embedding `modeling.essentia_embed` writes, straight
cosine distance, no scaler or PCA. That mirrors what cosine.club itself does
(raw embedding, cosine similarity) and what notebook 08 measured: on the
external-seed check this embedding placed Spotify's and YouTube's own picks
over 20x closer (relative rank) than the hand-built `heard` space did.

Needs only numpy/pandas/scikit-learn to *use* a fitted embedding — extraction
(`modeling/essentia_embed.py`) is the only piece that needs `essentia`, so
this class works from the project's normal environment.

    from modeling.effnet_recommender import EffNetRecommender
    rec = EffNetRecommender.from_npz("data/effnet_embeddings_full.npz", tracks)
    rec.similar_to(2, n=10)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from modeling.recommender import DEFAULT_N, MAX_DEPTH, depth_for

logger = logging.getLogger(__name__)


class EffNetRecommender:
    """Content-based nearest-neighbour recommender over Discogs-EffNet embeddings.

    Cosine distance, same reasoning as TrackRecommender: direction carries the
    similarity, not magnitude.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.index_ = None
        self.track_ids_ = None
        self.tracks_ = None
        self.embedding_ = None

    def fit(self, ids: np.ndarray, embeddings: np.ndarray, tracks: pd.DataFrame = None) -> "EffNetRecommender":
        self.track_ids_ = np.asarray(ids)
        self.embedding_ = pd.DataFrame(embeddings, index=self.track_ids_)
        self.index_ = NearestNeighbors(metric=self.metric).fit(embeddings)
        self.tracks_ = tracks
        logger.info("Indexed %s tracks in %s dimensions", embeddings.shape[0], embeddings.shape[1])
        return self

    @classmethod
    def from_npz(cls, path, tracks: pd.DataFrame = None, metric: str = "cosine") -> "EffNetRecommender":
        """Load embeddings written by `modeling.essentia_embed`'s `--out` file."""
        loaded = np.load(Path(path))
        ids, embeddings = loaded["ids"], loaded["embeddings"]
        if tracks is not None:
            tracks = tracks.loc[tracks.index.intersection(ids)]
            keep = np.isin(ids, tracks.index)
            ids, embeddings = ids[keep], embeddings[keep]
        return cls(metric=metric).fit(ids, embeddings, tracks)

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

    def similar_to(self, track_id, n: int = DEFAULT_N, adventure: float = 0.0,
                    max_depth: int = MAX_DEPTH) -> pd.DataFrame:
        """n tracks near `track_id`, read from a window of the ranked neighbour list.

        Same `depth_for()` dial as TrackRecommender — a window into the
        already-ranked list, not a second model. Notebook 08's depth test
        measured what that window buys here: genre agreement@10 goes
        0.53 -> 0.38 -> 0.22 from rank 10 to 110 to 310, a real gradual
        decay rather than a cliff, so raising `adventure` does trend toward
        "different genre" without jumping there immediately.
        """
        row = self._position(track_id)
        start = depth_for(adventure, max_depth=max_depth)
        distances, idx = self.index_.kneighbors(
            self.embedding_.iloc[[row]], n_neighbors=start + n + 1
        )
        distances, idx = distances[0][1:], idx[0][1:]  # drop the seed itself
        window = slice(start, start + n)
        return self._with_metadata(
            self.track_ids_[idx[window]],
            distances[window],
            np.arange(start + 1, start + n + 1),
        )

    def taste_profile(self, track_ids) -> np.ndarray:
        """Centroid position for a list of seeds — same idea as TrackRecommender's."""
        rows = [self._position(t) for t in track_ids]
        return self.embedding_.iloc[rows].mean(axis=0).to_numpy()
