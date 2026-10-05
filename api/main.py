"""HTTP bridge between the EffNet tier recommender and the React frontend.

Implements exactly the contract `frontend/README.md` defines, plus one extra
route the contract didn't originally cover:

    GET  /tracks?query=<text>                                  -> { "tracks": Track[] }  (names, genres, or a described sound)
    POST /recommendations {track_id, adventure}                 -> { "recommendations": Recommendation[] }
    GET  /audio/{track_id}                                      -> audio/mpeg (the FMA clip itself)

Runs entirely in the project's normal (3.13) environment. It serves the
fma_medium catalogue (`modeling.catalogue`: ~24.9k tracks, re-uploads and
silent files removed) from *precomputed* embeddings -- Discogs-EffNet for
recommendations, CLAP for text search -- so it never imports Essentia; that
dependency stays confined to the offline extraction step
(`modeling/essentia_embed.py`, run under `.venv-essentia`).

    uv run uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations


from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from modeling.catalogue import Catalogue, TrackUnavailable, load_catalogue, local_audio_path
from modeling.connecting_factors import ConnectingFactors
from modeling.effnet_recommender import EffNetRecommender
from modeling.text_search import TextSearch, search_catalogue
from modeling.tiers import (MEDIUM_BANDS, MEDIUM_WIDEN_TO, RECOMMENDATION_COUNT, build_connecting_factors,
                            explore_tiers, tier_for_adventure)

app = FastAPI(title="TasteBridge recommender API")

# Vite's dev server; add the deployed frontend origin here too once there is one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_catalogue: Catalogue | None = None
_tracks = None
_rec: EffNetRecommender | None = None
_cf: ConnectingFactors | None = None
_search: TextSearch | None = None


@app.on_event("startup")
def load_model() -> None:
    global _catalogue, _tracks, _rec, _cf, _search
    _catalogue = catalogue = load_catalogue()
    _tracks = catalogue.tracks
    _rec = EffNetRecommender().fit(catalogue.ids, catalogue.effnet, _tracks)
    _cf = build_connecting_factors(_rec, _tracks)
    _search = TextSearch(catalogue.ids, catalogue.clap).load()


class RecommendRequest(BaseModel):
    track_id: str
    adventure: float = 0.0


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
    tracks = []
    for tid, matched_by in search_catalogue(_tracks, _search, query[:200], n=25):
        payload = _track_payload(request, tid)
        payload["matchedBy"] = matched_by  # "name" | "genre" | "sound"
        tracks.append(payload)
    return {"tracks": tracks}


def _resolve(track_id) -> int:
    """Catalogue id for a requested id -- a removed re-upload maps to its kept
    copy; anything else unavailable is an HTTP error with a readable message."""
    try:
        return _catalogue.resolve(int(track_id))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{track_id!r} isn't a track id.")
    except TrackUnavailable as exc:
        status = 410 if int(track_id) in _catalogue.removed else 404  # 410 Gone: existed, deliberately removed
        raise HTTPException(status_code=status, detail=str(exc))


@app.get("/audio/{track_id}")
def stream_audio(track_id: int):
    resolved = _resolve(track_id)
    if resolved != track_id:  # a re-upload: play the copy we kept
        return RedirectResponse(f"/audio/{resolved}", status_code=307)
    path = local_audio_path(track_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"The audio for track {track_id} is missing on this server.")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name, content_disposition_type="inline")


@app.post("/recommendations")
def recommend(req: RecommendRequest, request: Request):
    seed_id = _resolve(req.track_id)

    tier = tier_for_adventure(req.adventure)
    seed_row = _tracks.loc[seed_id]
    tiers = explore_tiers(
        _rec, _cf, _tracks,
        seed_id=seed_id,
        seed_artist=seed_row[("artist", "name")],
        n=RECOMMENDATION_COUNT,
        only=tier,
        widen_to=MEDIUM_WIDEN_TO,
        **MEDIUM_BANDS,
    )

    picks = tiers[tier][:RECOMMENDATION_COUNT]
    recommendations = []
    for rank_position, (track_id, distance, rank, factors) in enumerate(picks, start=1):
        payload = _track_payload(request, int(track_id))
        payload["rank"] = rank_position
        payload["distance"] = round(float(distance), 3)
        recommendations.append(payload)
    return {"recommendations": recommendations}
