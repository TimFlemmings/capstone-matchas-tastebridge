"""HTTP bridge between the EffNet tier recommender and the React frontend.

Implements exactly the contract `frontend/README.md` defines, plus one extra
route the contract didn't originally cover:

    GET  /tracks?query=<text>                                  -> { "tracks": Track[] }
    POST /recommendations {track_id, adventure, limit}          -> { "recommendations": Recommendation[] }
    GET  /audio/{track_id}                                      -> audio/mpeg (the FMA clip itself)

Runs entirely in the project's normal (3.13) environment. It loads the
*precomputed* Discogs-EffNet embeddings (`data/effnet_embeddings_full.npz`),
so it never imports Essentia -- that dependency stays confined to the offline
extraction step (`modeling/essentia_embed.py`, run under `.venv-essentia`).

    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from modeling.connecting_factors import ConnectingFactors
from modeling.data import load_tracks
from modeling.effnet_recommender import EffNetRecommender
from modeling.rhythm import audio_path
from modeling.tiers import DEFAULT_BANDS, explore_tiers, tier_for_adventure

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = REPO_ROOT / "data" / "effnet_embeddings_full.npz"

app = FastAPI(title="TasteBridge recommender API")

# Vite's dev server; add the deployed frontend origin here too once there is one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_tracks = None
_rec: EffNetRecommender | None = None
_cf: ConnectingFactors | None = None


@app.on_event("startup")
def load_model() -> None:
    global _tracks, _rec, _cf
    _tracks = load_tracks("small")
    _rec = EffNetRecommender.from_npz(EMBEDDINGS_PATH, _tracks)
    _cf = ConnectingFactors.build(_rec.tracks_)


class RecommendRequest(BaseModel):
    track_id: str
    adventure: float = 0.0
    limit: int = 10


def _duration_str(seconds) -> str:
    seconds = int(seconds) if seconds == seconds else 0  # NaN check without importing pandas here
    return f"{seconds // 60}:{seconds % 60:02d}"


def _cover_url(track_id) -> str:
    # FMA ships no cover art; a deterministic placeholder keeps every card
    # visually distinct instead of showing a broken image.
    return f"https://picsum.photos/seed/tastebridge-{track_id}/400/400"


def _preview_url(request: Request, track_id) -> str:
    return str(request.base_url) + f"audio/{track_id}"


def _track_payload(request: Request, track_id) -> dict:
    row = _tracks.loc[track_id]
    return {
        "id": str(track_id),
        "title": row[("track", "title")] or "Untitled",
        "artist": row[("artist", "name")] or "Unknown artist",
        "genre": row[("track", "genre_top")] or "Unknown",
        "duration": _duration_str(row[("track", "duration")]),
        "coverUrl": _cover_url(track_id),
        "audioUrl": _preview_url(request, track_id),
    }


@app.get("/tracks")
def search_tracks(request: Request, query: str = ""):
    q = query.strip().lower()
    if not q:
        matches = _tracks.index[:25]
    else:
        title = _tracks[("track", "title")].fillna("").str.lower()
        artist = _tracks[("artist", "name")].fillna("").str.lower()
        genre = _tracks[("track", "genre_top")].fillna("").str.lower()
        mask = title.str.contains(q, regex=False) | artist.str.contains(q, regex=False) | genre.str.contains(q, regex=False)
        matches = _tracks.index[mask][:25]
    return {"tracks": [_track_payload(request, tid) for tid in matches]}


@app.get("/audio/{track_id}")
def stream_audio(track_id: int):
    path = audio_path(track_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no audio file for track {track_id}")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name, content_disposition_type="inline")


@app.post("/recommendations")
def recommend(req: RecommendRequest, request: Request):
    try:
        seed_id = int(req.track_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"track_id must be an FMA track id, got {req.track_id!r}")
    if seed_id not in _tracks.index:
        raise HTTPException(status_code=404, detail=f"track {seed_id} not found")

    tier = tier_for_adventure(req.adventure)
    seed_row = _tracks.loc[seed_id]
    tiers = explore_tiers(
        _rec, _cf, _tracks,
        seed_id=seed_id,
        seed_artist=seed_row[("artist", "name")],
        n=req.limit,
        **{k: DEFAULT_BANDS[k] for k in ("familiar", "bridge", "adventurous")},
    )

    picks = tiers[tier][: req.limit]
    recommendations = []
    for rank_position, (track_id, distance, rank, factors) in enumerate(picks, start=1):
        payload = _track_payload(request, int(track_id))
        payload["rank"] = rank_position
        payload["distance"] = round(float(distance), 3)
        recommendations.append(payload)
    return {"recommendations": recommendations}
