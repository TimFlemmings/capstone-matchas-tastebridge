from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from modeling.connecting_factors import ConnectingFactors
from modeling.effnet_recommender import EffNetRecommender
from modeling.tiers import build_connecting_factors, explore_tiers, tier_for_adventure


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "capstone-matchas" / "data"
FMA_DATA_DIR = Path(os.environ.get(
    "FMA_DATA_DIR", DEFAULT_DATA_DIR)).expanduser()
FMA_AUDIO_DIR = Path(os.environ.get(
    "FMA_AUDIO_DIR", FMA_DATA_DIR / "fma_small")).expanduser()
FMA_METADATA_DIR = Path(os.environ.get(
    "FMA_METADATA_DIR", FMA_DATA_DIR / "fma_metadata")).expanduser()
EMBEDDINGS_PATH = Path(
    os.environ.get("EFFNET_EMBEDDINGS_PATH", REPO_ROOT /
                   "data" / "effnet_embeddings_full.npz")
).expanduser()
API_PUBLIC_BASE = os.environ.get(
    "API_PUBLIC_BASE", "http://localhost:8001").rstrip("/")
MAX_SEARCH_RESULTS = 25

def load_effnet_catalogue() -> tuple[pd.DataFrame, EffNetRecommender, ConnectingFactors, pd.DataFrame]:
    tracks_path = FMA_METADATA_DIR / "tracks.csv"
    missing = [str(path) for path in (tracks_path, FMA_AUDIO_DIR,
                                      EMBEDDINGS_PATH) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Required EffNet/FMA files are missing: " + ", ".join(missing))

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    tracks = tracks[tracks[("set", "subset")] == "small"]
    recommender = EffNetRecommender.from_npz(EMBEDDINGS_PATH, tracks=tracks)
    indexed_ids = pd.Index(recommender.track_ids_.astype(int))
    tracks = tracks.loc[indexed_ids]
    projection = PCA(n_components=2, svd_solver="randomized", random_state=17)
    map_coordinates = projection.fit_transform(
        recommender.embedding_.to_numpy())
    map_coordinates -= map_coordinates.min(axis=0)
    coordinate_range = map_coordinates.max(axis=0)
    coordinate_range[coordinate_range == 0] = 1
    map_coordinates /= coordinate_range
    map_points = pd.DataFrame(
        map_coordinates, index=indexed_ids, columns=["x", "y"]
    )
    connecting_factors = build_connecting_factors(
        recommender, tracks, genres_csv=FMA_METADATA_DIR / "genres.csv")
    return tracks, recommender, connecting_factors, map_points


@asynccontextmanager
async def lifespan(app: FastAPI):
    (app.state.tracks, app.state.recommender, app.state.connecting_factors,
     app.state.map_points) = load_effnet_catalogue()
    yield


app = FastAPI(title="Tastebridge EffNet API",
              version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    track_id: str = Field(..., description="FMA small track id")
    adventure: float = Field(default=0.35, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)


def track_payload(track_id: int, row: pd.Series) -> dict:
    title = row.get(("track", "title"), "Untitled")
    artist = row.get(("artist", "name"), "Unknown artist")
    genre = row.get(("track", "genre_top"), "Unknown")
    duration = row.get(("track", "duration"), 0)
    seconds = int(duration) if pd.notna(duration) else 0
    return {
        "id": str(track_id),
        "title": str(title) if pd.notna(title) else "Untitled",
        "artist": str(artist) if pd.notna(artist) else "Unknown artist",
        "genre": str(genre) if pd.notna(genre) else "Unknown",
        "duration": f"{seconds // 60}:{seconds % 60:02d}",
        "coverUrl": f"https://picsum.photos/seed/tastebridge-{track_id}/400/400",
        "audioUrl": f"{API_PUBLIC_BASE}/audio/{track_id}",
    }


@app.get("/")
def root():
    return {"message": "Tastebridge EffNet API is running", "model": "Discogs-EffNet embeddings + cosine k-NN"}


@app.get("/health")
def health(request: Request):
    recommender: EffNetRecommender = request.app.state.recommender
    return {
        "status": "ok",
        "model": "Discogs-EffNet",
        "tracks": len(recommender.track_ids_),
        "embedding_dimensions": int(recommender.embedding_.shape[1]),
    }


@app.get("/tracks")
def search_tracks(request: Request, query: str = Query(default="", max_length=200)):
    tracks: pd.DataFrame = request.app.state.tracks
    normalized = query.strip().casefold()
    results = []
    for track_id, row in tracks.iterrows():
        item = track_payload(int(track_id), row)
        searchable = f"{item['title']} {item['artist']} {item['genre']}".casefold(
        )
        if not normalized or normalized in searchable:
            results.append(item)
            if len(results) == MAX_SEARCH_RESULTS:
                break
    return {"tracks": results}


@app.get("/taste-map")
def taste_map(request: Request):
    tracks: pd.DataFrame = request.app.state.tracks
    map_points: pd.DataFrame = request.app.state.map_points
    points = []
    for track_id, coordinates in map_points.iterrows():
        item = track_payload(int(track_id), tracks.loc[int(track_id)])
        item["x"] = float(coordinates["x"])
        item["y"] = float(coordinates["y"])
        points.append(item)
    return {
        "method": "PCA projection of Discogs-EffNet embeddings (visualization only)",
        "points": points,
    }


@app.get("/audio/{track_id}")
def get_audio(track_id: str, request: Request):
    if not track_id.isdigit() or int(track_id) not in request.app.state.tracks.index:
        raise HTTPException(status_code=404, detail="Track not found")
    padded_id = f"{int(track_id):06d}"
    path = FMA_AUDIO_DIR / padded_id[:3] / f"{padded_id}.mp3"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


@app.post("/recommendations")
def recommendations(payload: RecommendationRequest, request: Request):
    recommender: EffNetRecommender = request.app.state.recommender
    tracks: pd.DataFrame = request.app.state.tracks
    try:
        seed_id = int(payload.track_id)
        seed_artist = tracks.loc[seed_id, ("artist", "name")]
    except (ValueError, KeyError):
        raise HTTPException(
            status_code=404, detail="Seed track not found") from None

    # Tim's tuned Familiar / Bridge / Adventurous tiers (notebook 09, round 3):
    # the slider picks a tier, the tier applies its rank band, connection
    # gate, genre-diversity caps, decade filter and country preference.
    tier = tier_for_adventure(payload.adventure)
    picks = explore_tiers(
        recommender, request.app.state.connecting_factors, tracks,
        seed_id=seed_id, seed_artist=seed_artist, n=payload.limit, only=tier,
    )[tier]

    output = []
    for track_id, distance, rank, factors in picks:
        track_id = int(track_id)
        item = track_payload(track_id, tracks.loc[track_id])
        item["rank"] = int(rank)
        item["distance"] = float(distance)
        item["tier"] = tier
        item["connectingFactors"] = list(factors)
        output.append(item)
    return {"recommendations": output}
