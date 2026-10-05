from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import get_token
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from modeling.catalogue import Catalogue, TrackUnavailable, load_catalogue, local_audio_path
from modeling.connecting_factors import ConnectingFactors
from modeling.effnet_recommender import EffNetRecommender
from modeling.text_search import TextSearch, search_catalogue
from modeling.tiers import (MEDIUM_BANDS, MEDIUM_WIDEN_TO, RECOMMENDATION_COUNT, build_connecting_factors,
                            explore_tiers, tier_for_adventure)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "capstone-matchas" / "data"
# the sibling checkout's data if there is one (the team's original layout),
# else this repository's own data/ -- same fallback as app.py
FMA_DATA_DIR = Path(os.environ.get(
    "FMA_DATA_DIR",
    DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else REPO_ROOT / "data",
)).expanduser()
FMA_AUDIO_DIR = Path(os.environ.get(
    "FMA_AUDIO_DIR", FMA_DATA_DIR / "fma_medium")).expanduser()
# 128 kbps loudness-matched copies (modeling/transcode_audio.py) -- served in
# preference to FMA_AUDIO_DIR when present, since they're what the cloud serves
WEB_AUDIO_DIR = Path(os.environ.get(
    "WEB_AUDIO_DIR", REPO_ROOT / "data" / "web_audio")).expanduser()
FMA_METADATA_DIR = Path(os.environ.get(
    "FMA_METADATA_DIR", FMA_DATA_DIR / "fma_metadata")).expanduser()
EMBEDDINGS_PATH = Path(
    os.environ.get("EFFNET_EMBEDDINGS_PATH", REPO_ROOT /
                   "data" / "effnet_embeddings_medium.npz")
).expanduser()
CLAP_EMBEDDINGS_PATH = Path(
    os.environ.get("CLAP_EMBEDDINGS_PATH", REPO_ROOT /
                   "data" / "clap_embeddings_medium.npz")
).expanduser()
# "" makes audio URLs relative ("/audio/2"), for when the frontend is served
# by this same server (the hosted Space)
API_PUBLIC_BASE = os.environ.get(
    "API_PUBLIC_BASE", "http://localhost:8001").rstrip("/")
# Hosted only: clips that aren't on disk are streamed from this Hugging Face
# dataset instead (private, so it needs HF_TOKEN), Range requests passed through
HF_AUDIO_REPO = os.environ.get("HF_AUDIO_REPO", "")
HF_AUDIO_PREFIX = os.environ.get("HF_AUDIO_PREFIX", "web_audio").strip("/")
# Hosted only: the built frontend (`npm run build`), served from "/"
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", "")
MAX_SEARCH_RESULTS = 25
# per-track tempo estimate (BPM), extracted from data/rhythm_features_medium.csv
TEMPO_PATH = REPO_ROOT / "data" / "processed" / "track_tempo.csv"
TEMPO: dict = {}

def load_effnet_catalogue() -> tuple[Catalogue, pd.DataFrame, EffNetRecommender, ConnectingFactors, TextSearch]:
    tracks_path = FMA_METADATA_DIR / "tracks.csv"
    missing = [str(path) for path in (tracks_path, EMBEDDINGS_PATH, CLAP_EMBEDDINGS_PATH)
               if not path.exists()]
    if not HF_AUDIO_REPO and not FMA_AUDIO_DIR.exists() and not WEB_AUDIO_DIR.exists():
        missing.append(f"{FMA_AUDIO_DIR} (or {WEB_AUDIO_DIR})")
    if missing:
        raise RuntimeError(
            "Required EffNet/FMA files are missing: " + ", ".join(missing))

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    tracks = tracks[tracks[("set", "subset")].isin(["small", "medium"])]
    # fma_medium tracks with both embeddings, re-uploads and silent files removed
    catalogue = load_catalogue(tracks=tracks, effnet_path=EMBEDDINGS_PATH, clap_path=CLAP_EMBEDDINGS_PATH)
    tracks = catalogue.tracks
    recommender = EffNetRecommender().fit(catalogue.ids, catalogue.effnet, tracks)
    connecting_factors = build_connecting_factors(
        recommender, tracks, genres_csv=FMA_METADATA_DIR / "genres.csv")
    text_search = TextSearch(catalogue.ids, catalogue.clap).load()
    if TEMPO_PATH.exists():
        TEMPO.update(pd.read_csv(TEMPO_PATH, index_col=0)["bpm"].dropna().astype(int).to_dict())
    return catalogue, tracks, recommender, connecting_factors, text_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    (app.state.catalogue, app.state.tracks, app.state.recommender, app.state.connecting_factors,
     app.state.text_search) = load_effnet_catalogue()
    app.state.hub = httpx.AsyncClient(follow_redirects=True, timeout=30) if HF_AUDIO_REPO else None
    yield
    if app.state.hub is not None:
        await app.state.hub.aclose()


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
# shrinks the JSON responses; audio/* is excluded by default, so ranged clips stream as-is
app.add_middleware(GZipMiddleware, minimum_size=1000)


class RecommendationRequest(BaseModel):
    track_id: str = Field(..., description="FMA track id")
    adventure: float = Field(default=0.35, ge=0.0, le=1.0)


def track_payload(track_id: int, row: pd.Series) -> dict:
    title = row.get(("track", "title"), "Untitled")
    artist = row.get(("artist", "name"), "Unknown artist")
    genre = row.get(("track", "genre_top"), "Unknown")
    duration = row.get(("track", "duration"), 0)
    seconds = int(duration) if pd.notna(duration) else 0
    location = row.get(("artist", "location"))
    created = row.get(("track", "date_created"))  # when the track was added to FMA
    return {
        "id": str(track_id),
        "title": str(title) if pd.notna(title) else "Untitled",
        "artist": str(artist) if pd.notna(artist) else "Unknown artist",
        "genre": str(genre) if pd.notna(genre) else "Unknown",
        "duration": f"{seconds // 60}:{seconds % 60:02d}",
        "coverUrl": f"https://picsum.photos/seed/tastebridge-{track_id}/400/400",
        "audioUrl": f"{API_PUBLIC_BASE}/audio/{track_id}",
        "location": str(location).strip() if pd.notna(location) else None,
        "dateCreated": str(created)[:10] if pd.notna(created) else None,
        "bpm": TEMPO.get(int(track_id)),
    }


def root():
    return {"message": "Tastebridge EffNet API is running", "model": "Discogs-EffNet embeddings + cosine k-NN"}


if not FRONTEND_DIST:  # otherwise "/" is the frontend, mounted at the bottom of this file
    app.get("/")(root)


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
    results = []
    for track_id, matched_by in search_catalogue(tracks, request.app.state.text_search, query,
                                                 n=MAX_SEARCH_RESULTS):
        item = track_payload(track_id, tracks.loc[track_id])
        item["matchedBy"] = matched_by  # "name" | "genre" | "sound"
        results.append(item)
    return {"tracks": results}


def resolve_track(request: Request, track_id: str) -> int:
    """Catalogue id for a requested id -- a removed re-upload maps to its kept
    copy; anything else unavailable is an HTTP error with a readable message."""
    catalogue = request.app.state.catalogue
    if not str(track_id).strip().isdigit():
        raise HTTPException(status_code=400, detail=f"{track_id!r} isn't a track id.")
    try:
        return catalogue.resolve(int(track_id))
    except TrackUnavailable as exc:
        status = 410 if int(track_id) in catalogue.removed else 404  # 410 Gone: existed, deliberately removed
        raise HTTPException(status_code=status, detail=str(exc)) from None


@app.get("/audio/{track_id}")
async def get_audio(track_id: str, request: Request):
    resolved = resolve_track(request, track_id)
    if resolved != int(track_id):  # a re-upload: play the copy we kept
        return RedirectResponse(f"/audio/{resolved}", status_code=307)
    path = local_audio_path(resolved, web_dir=WEB_AUDIO_DIR, original_dir=FMA_AUDIO_DIR)
    if path.is_file():
        return FileResponse(path, media_type="audio/mpeg", filename=path.name)
    if HF_AUDIO_REPO:
        return await stream_from_hub(request, resolved)
    raise HTTPException(status_code=404, detail=f"The audio for track {resolved} is missing on this server.")


def hub_audio_url(track_id: int) -> str:
    name = f"{track_id:06d}"
    path = "/".join(p for p in (HF_AUDIO_PREFIX, name[:3], f"{name}.mp3") if p)
    return f"https://huggingface.co/datasets/{HF_AUDIO_REPO}/resolve/main/{path}"


async def stream_from_hub(request: Request, track_id: int) -> StreamingResponse:
    """Pass the clip through from the Hugging Face dataset. The browser's Range
    header goes upstream and the status/range headers come back, so seeking
    works. The token is dropped by httpx when the Hub redirects to its CDN."""
    url = hub_audio_url(track_id)
    headers = {}
    if token := get_token():
        headers["Authorization"] = f"Bearer {token}"
    if "range" in request.headers:
        headers["Range"] = request.headers["range"]
    hub: httpx.AsyncClient = request.app.state.hub
    upstream = await hub.send(hub.build_request("GET", url, headers=headers), stream=True)
    if upstream.status_code not in (200, 206):
        await upstream.aclose()
        status = 404 if upstream.status_code == 404 else 502
        raise HTTPException(status_code=status, detail=f"The audio for track {track_id} couldn't be loaded "
                                                       f"(storage answered {upstream.status_code}).")
    passed = {k: upstream.headers[k] for k in ("content-length", "content-range", "accept-ranges", "etag")
              if k in upstream.headers}
    passed["cache-control"] = "public, max-age=86400"
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, media_type="audio/mpeg",
                             headers=passed, background=BackgroundTask(upstream.aclose))


@app.post("/recommendations")
def recommendations(payload: RecommendationRequest, request: Request):
    recommender: EffNetRecommender = request.app.state.recommender
    tracks: pd.DataFrame = request.app.state.tracks
    seed_id = resolve_track(request, payload.track_id)
    seed_artist = tracks.loc[seed_id, ("artist", "name")]

    # Tim's tuned Familiar / Bridge / Adventurous tiers (notebook 09, round 3):
    # the slider picks a tier, the tier applies its rank band, connection
    # gate, genre-diversity caps, decade filter and country preference.
    tier = tier_for_adventure(payload.adventure)
    picks = explore_tiers(
        recommender, request.app.state.connecting_factors, tracks,
        seed_id=seed_id, seed_artist=seed_artist, n=RECOMMENDATION_COUNT, only=tier,
        widen_to=MEDIUM_WIDEN_TO, **MEDIUM_BANDS,
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


if FRONTEND_DIST:  # last, so every API route above takes precedence
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
