# The recommender API — what it is, how to run it, what it returns

This is the bridge between `frontend/` (React/Vite) and the EffNet tier
recommender from notebooks 08/09. It implements exactly the contract
`frontend/README.md` already specified — nothing on the frontend needed to
change.

Code: [`app/main.py`](app/main.py). Tier logic: [`modeling/tiers.py`](modeling/tiers.py)
(the `explore_tiers` function from notebook 09, extracted so both the API and
the notebook use the same implementation instead of two copies drifting apart).

## Run it

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Needs `data/effnet_embeddings_full.npz` (the full-catalogue Discogs-EffNet
embeddings from notebook 08) and the FMA metadata to already be in place —
same data either of you already has for 08/09. **No Essentia import anywhere
in this path** — it runs in the main environment, not `.venv-essentia`. See
`ESSENTIA_SETUP.md` if either of those data files are missing.

Point the frontend at it:

```bash
# frontend/.env.local (gitignored, each of you sets your own)
VITE_RECOMMENDER_API_URL=http://localhost:8000
```

## `GET /tracks?query=<text>`

Case-insensitive substring match against title, artist, and genre. Returns up
to 25 results; an empty query returns the first 25 catalogue tracks.

```bash
curl "http://localhost:8000/tracks?query=food"
```

```json
{
  "tracks": [
    {
      "id": "2",
      "title": "Food",
      "artist": "AWOL",
      "genre": "Hip-Hop",
      "duration": "2:48",
      "coverUrl": "https://picsum.photos/seed/tastebridge-2/400/400",
      "audioUrl": "http://localhost:8000/audio/2"
    }
  ]
}
```

## `POST /recommendations`

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"track_id": "2", "adventure": 0.5, "limit": 3}'
```

```json
{
  "recommendations": [
    {
      "id": "13668",
      "title": "No Time For Lookin' Back",
      "artist": "This One",
      "genre": "Electronic",
      "duration": "1:21",
      "coverUrl": "https://picsum.photos/seed/tastebridge-13668/400/400",
      "audioUrl": "http://localhost:8000/audio/13668",
      "rank": 1,
      "distance": 0.282
    }
  ]
}
```

`track_id` is an FMA track id, sent and returned as a string (frontend's
`Track.id` type). `adventure` is 0.0–1.0 and currently maps onto notebook 09's
three tiers with an even three-way split — 0–0.33 Familiar, 0.33–0.67 Bridge,
0.67–1.0 Adventurous (`tier_for_adventure` in `modeling/tiers.py`). That split
point is arbitrary; nobody has tuned where the slider should actually hand off
between tiers, unlike the rank bands themselves, which are notebook 09's
measured settings.

## `GET /audio/{track_id}`

Streams the FMA clip itself (`audio/mpeg`), with HTTP Range support so an
`<audio>` element can seek. Backed by `modeling.rhythm.audio_path`, i.e. the
same local `data/fma_small/` files 08/09 already use — nothing extra to set
up. 404s if the track id isn't in the small subset.

```bash
curl -H "Range: bytes=0-1023" http://localhost:8000/audio/2 -D - -o /dev/null
# HTTP/1.1 206 Partial Content
# content-type: audio/mpeg
# accept-ranges: bytes
# content-range: bytes 0-1023/960738
```

Every `Track`/`Recommendation` the API returns includes a `audioUrl`
pointing here — the frontend's play buttons (`frontend/src/App.tsx`) are wired
to it via a real `<audio>` element. In preview mode there's no `audioUrl`
(the sample catalogue has no backing audio), so the play buttons are disabled
rather than lying about playing something.

## Known gaps, not bugs

- **`coverUrl` is a placeholder.** FMA ships no cover art, so this is a
  deterministic `picsum.photos` image per track id, not real artwork.
- **What's confirmed working end to end**: search, recommendations, audio
  streaming with seek (`206 Partial Content`), the adventure dial visibly
  changing the distance band (~0.28 at low adventure vs 0.4+ at high, on
  track 2), and CORS between `localhost:5173` and `localhost:8000`.
