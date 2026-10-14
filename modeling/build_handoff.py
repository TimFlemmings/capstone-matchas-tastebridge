"""Builds everything notebooks 08/09 (the essentia311-kernel notebooks) read
from disk instead of computing live -- the heard-space side of the handoff.
Runs in the *main* environment (3.13), not `.venv-essentia`.

Three outputs, all under data/processed/essentia_handoff/:

    all_small_tracks.csv     every fma_small track id (for the full 8k EffNet
                              embedding run -- modeling.essentia_embed)
    sample_tracks.csv        a ~250/genre stratified sample + the seeds notebooks
                              06/07/08/09 all reuse, with metadata attached (for
                              the faster ~2k-track EffNet sample used early in 08)
    heard_top10_seed{2,3538}.csv   heard-space top-10 for the two control seeds,
                              so 08 can print them beside EffNet's own picks
    heard_external_scoring.csv     heard-space distance/percentile for the 18
                              Spotify/YouTube recommendations against the
                              external seed -- notebook 08's side-by-side table

Needs data/rhythm_features.csv and data/separation_features.csv to already
exist (`uv run python -m modeling.rhythm` then `modeling.separation` first).

    uv run python -m modeling.build_handoff
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.data import load_fma, genre_labels
from modeling.external import extract_heard_features
from modeling.features import fit_pca, scale_features, space, transform
from modeling.recommender import TrackRecommender

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "processed" / "essentia_handoff"
SAMPLE_PER_GENRE = 250
MUST_INCLUDE = [2, 3538, 704, 1083, 56690]  # every catalogue seed reused across 06-09
SEED_TOP10_FOR = [2, 3538]
SEED_PATH = REPO_ROOT / "data" / "raw" / "youtube_tests" / "Y2Mate.is - Elyjah Slaps the Space.mp3"


def _cosine_distance(embeddings: np.ndarray, v: np.ndarray) -> np.ndarray:
    vu = v / np.linalg.norm(v)
    eu = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return 1 - eu @ vu


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. every fma_small track id, for the full-catalogue EffNet run --
    # deliberately the *unfiltered* 8,000, not the heard-space-joined set
    # below (which drops 7 tracks missing rhythm/separation data): EffNet
    # doesn't need heard-space's features at all, and the "full" catalogue
    # notebooks 08/09 already used (7,994 tracks) was built from this same
    # unfiltered list, minus 6 that failed to decode during embedding.
    tracks_all = load_fma("small")[1]
    tracks_all.index.to_series().to_csv(OUT / "all_small_tracks.csv", header=["track_id"], index=False)
    logger.info("wrote all_small_tracks.csv (%s tracks)", len(tracks_all))

    X_heard = space("heard")
    tracks = tracks_all.loc[X_heard.index]
    Xs, scaler = scale_features(X_heard, "standard")
    pca = fit_pca(Xs, n_components=0.90)
    rec = TrackRecommender(scaler, pca).fit(X_heard, tracks)

    # 2. stratified sample + metadata
    labels = genre_labels(tracks)
    rng = np.random.default_rng(11)
    sample_ids = []
    for g in labels.dropna().unique():
        pool = labels[labels == g].index.to_numpy()
        sample_ids.extend(rng.choice(pool, size=min(SAMPLE_PER_GENRE, len(pool)), replace=False))
    sample_ids = sorted(set(int(t) for t in sample_ids) | set(MUST_INCLUDE))
    sample_meta = tracks.loc[sample_ids, [("artist", "name"), ("track", "title"),
                                           ("track", "genre_top"), ("track", "genres_all")]]
    sample_meta.columns = ["artist", "title", "genre_top", "genres_all"]
    sample_meta.to_csv(OUT / "sample_tracks.csv")
    logger.info("wrote sample_tracks.csv (%s tracks)", len(sample_ids))

    # 3. heard-space top-10 for the reused control seeds
    for seed_id in SEED_TOP10_FOR:
        top = rec.similar_to(seed_id, n=10)[["distance", "artist", "title", "genre"]].reset_index()
        top.to_csv(OUT / f"heard_top10_seed{seed_id}.csv", index=False)
    logger.info("wrote heard_top10_seed{%s}.csv", ",".join(map(str, SEED_TOP10_FOR)))

    # 4. heard-space scoring of the 18 Spotify/YouTube recommendations
    seed_row = extract_heard_features(SEED_PATH, X_heard.columns)
    seed_emb = transform(seed_row.to_frame().T, scaler, pca).to_numpy()[0]
    catalog_dist = np.sort(_cosine_distance(rec.embedding_.to_numpy(), seed_emb))
    size = len(catalog_dist)

    rows = []
    for folder in ["spotify_tests", "youtube_tests"]:
        for p in sorted((REPO_ROOT / "data" / "raw" / folder).glob("*.mp3")):
            if "Elyjah" in p.name:
                continue
            feat = extract_heard_features(p, X_heard.columns)
            emb = transform(feat.to_frame().T, scaler, pca).to_numpy()[0]
            d = float(_cosine_distance(emb[np.newaxis, :], seed_emb)[0])
            rank = int(np.searchsorted(catalog_dist, d)) + 1
            rows.append({"platform": folder, "file": p.name, "heard_distance": d, "heard_percentile": rank / size})
    pd.DataFrame(rows).to_csv(OUT / "heard_external_scoring.csv", index=False)
    logger.info("wrote heard_external_scoring.csv (%s tracks)", len(rows))


if __name__ == "__main__":
    main()
