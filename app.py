from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from modeling.features import feature_set
from modeling.recommender import TrackRecommender


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "capstone-matchas" / "data"
FMA_DATA_DIR = Path(
    os.environ.get(
        "FMA_DATA_DIR",
        DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else REPO_ROOT / "data",
    )
).expanduser()
FMA_AUDIO_DIR = Path(os.environ.get(
    "FMA_AUDIO_DIR", FMA_DATA_DIR / "fma_small")).expanduser()
FMA_METADATA_DIR = Path(os.environ.get(
    "FMA_METADATA_DIR", FMA_DATA_DIR / "fma_metadata")).expanduser()
FMA_MODELS_DIR = Path(os.environ.get(
    "FMA_MODELS_DIR", FMA_DATA_DIR.parent / "models")).expanduser()
API_PUBLIC_BASE = os.environ.get(
    "API_PUBLIC_BASE", "http://localhost:8000").rstrip("/")
GENERIC_COVER_URL = (
    "https://images.unsplash.com/photo-1470225620780-dba8ba36b745"
    "?auto=format&fit=crop&w=900&q=85"
)


def load_recommender() -> tuple[pd.DataFrame, TrackRecommender]:
    tracks_path = FMA_METADATA_DIR / "tracks.csv"
    features_path = FMA_METADATA_DIR / "features.csv"
    scaler_path = FMA_MODELS_DIR / "scaler_90pct_timbre.pkl"
    pca_path = FMA_MODELS_DIR / "pca_90pct_timbre.pkl"
    required_paths = (tracks_path, features_path,
                      scaler_path, pca_path, FMA_AUDIO_DIR)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Required FMA files are missing: " + ", ".join(missing))

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    tracks = tracks[tracks[("set", "subset")] == "small"]
    features = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
    features = features.loc[features.index.intersection(tracks.index)].dropna()
    tracks = tracks.loc[features.index]
    timbre_features = feature_set(features, "timbre")

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    if timbre_features.shape[1] != scaler.n_features_in_:
        raise RuntimeError(
            f"Timbre feature count ({timbre_features.shape[1]}) does not match "
            f"the scaler ({scaler.n_features_in_})."
        )

    recommender = TrackRecommender(scaler, pca).fit(timbre_features, tracks)
    return tracks, recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tracks, app.state.recommender = load_recommender()
    yield


app = FastAPI(title="Tastebridge API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    track_id: str = Field(..., description="Seed track id")
    adventure: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)


@app.get("/")
def read_root():
    return {"message": "Tastebridge API is running"}


@app.get("/health")
def healthcheck(request: Request):
    return {"status": "ok", "tracks": len(request.app.state.tracks)}


def _track_payload(track_id: int, row: pd.Series) -> dict:
    title = row.get(("track", "title"), "Untitled")
    artist = row.get(("artist", "name"), "Unknown artist")
    genre = row.get(("track", "genre_top"), "Unknown")
    duration = row.get(("track", "duration"), 0)
    duration_seconds = int(duration) if pd.notna(duration) else 0

    return {
        "id": str(track_id),
        "title": str(title) if pd.notna(title) else "Untitled",
        "artist": str(artist) if pd.notna(artist) else "Unknown artist",
        "genre": str(genre) if pd.notna(genre) else "Unknown",
        "duration": f"{duration_seconds // 60}:{duration_seconds % 60:02d}",
        "coverUrl": GENERIC_COVER_URL,
        "audioUrl": f"{API_PUBLIC_BASE}/audio/{track_id}",
    }


@app.get("/tracks")
def get_tracks(request: Request, query: str = ""):
    tracks: pd.DataFrame = request.app.state.tracks
    normalized = query.strip().lower()
    results = []
    for track_id, row in tracks.iterrows():
        payload = _track_payload(int(track_id), row)
        if not normalized or normalized in f"{payload['title']} {payload['artist']} {payload['genre']}".lower():
            results.append(payload)
    return {"tracks": results}


@app.get("/audio/{track_id}")
def get_audio(track_id: str, request: Request):
    if not track_id.isdigit() or int(track_id) not in request.app.state.tracks.index:
        raise HTTPException(status_code=404, detail="Track not found")
    padded_id = f"{int(track_id):06d}"
    audio_path = FMA_AUDIO_DIR / padded_id[:3] / f"{padded_id}.mp3"
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg", filename=audio_path.name)


@app.post("/recommendations")
def get_recommendations(payload: RecommendationRequest, request: Request):
    recommender: TrackRecommender = request.app.state.recommender
    tracks: pd.DataFrame = request.app.state.tracks
    try:
        neighbors = recommender.similar_to(
            int(payload.track_id), n=payload.limit, adventure=payload.adventure
        )
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=404, detail="Seed track not found") from None

    recommendations = []
    for track_id, neighbor in neighbors.iterrows():
        item = _track_payload(int(track_id), tracks.loc[int(track_id)])
        item["rank"] = int(neighbor["rank"])
        item["distance"] = float(neighbor["distance"])
        recommendations.append(item)
    return {"recommendations": recommendations}
