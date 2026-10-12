# The recommender API — what it is, how to run it, what it returns

This is the bridge between `frontend/` (React/Vite) and the EffNet tier
recommender from notebooks 08/09. It implements exactly the contract
`frontend/README.md` already specified — nothing on the frontend needed to
change.

The live app runs the fuller server in [`app_effnet.py`](app_effnet.py)
(the same recommender, plus the endpoints the Taste Map and onboarding use);
this lighter API is handy for scripting and testing. Setup for both:
[`SETUP.md`](SETUP.md).

Code: [`api/main.py`](api/main.py). Tier logic: [`modeling/tiers.py`](modeling/tiers.py)
(the `explore_tiers` function from notebook 09, extracted so both the API and
the notebook use the same implementation instead of two copies drifting apart).

## Run it

```bash
uv run uvicorn api.main:app --reload --port 8000
```

(The folder was `app/` until the root-level `app.py` — the PCA app — started
shadowing it: `app.main` resolved to that file, so the old command stopped
working.)

Serves the fma_medium catalogue (`modeling/catalogue.py`, 24,781 tracks after
removing truncated files, re-uploads and silent clips). Needs `data/effnet_embeddings_medium.npz`
(Discogs-EffNet, recommendations), `data/clap_embeddings_medium.npz` (CLAP,
text search) and the FMA metadata — see [`SETUP.md`](SETUP.md) for where to get them. **No Essentia import anywhere
in this path** — it runs in the main environment, not `.venv-essentia`. See
`ESSENTIA_SETUP.md` if either of those data files are missing.

Point the frontend at it:

```bash
# frontend/.env.local (gitignored, each of you sets your own)
VITE_RECOMMENDER_API_URL=http://localhost:8000
```

## `GET /tracks?query=<text>`

Up to 25 results, each with a `matchedBy` field
(`modeling.text_search.search_catalogue`):

1. `"genre"` — the query is exactly a genre name ("jazz"): that genre's
   tracks, ordered by how well they fit "<genre> music" according to CLAP.
2. `"name"` — title or artist contains the query (case-insensitive).
3. `"sound"` — the rest is filled with tracks that *sound* like the query
   ("dreamy synth pads"), by CLAP text-to-audio similarity.

An empty query returns the first 25 catalogue tracks.

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
  -d '{"track_id": "2", "adventure": 0.5}'
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
`Track.id` type). `adventure` is 0.0–1.0 and maps onto the three tiers with an
even three-way split (`tier_for_adventure` in `modeling/tiers.py`): 0–0.33 is
Familiar, 0.33–0.67 Bridge, 0.67–1.0 Adventurous. The app's slider only stops at
0, 0.5 and 1, labelled **Home**, **Bridge** and **Adventure**; the code keeps the
names Familiar and Adventurous. The rank bands are notebook 09's settings,
rescaled to the 25k catalogue as `MEDIUM_BANDS` (Home 1–40, Bridge 81–420,
Adventure 1,401–6,000) so each tier covers the same cosine distances it did on
the original 8,000 tracks. Every request returns 15 picks.

## `GET /audio/{track_id}`

Streams the FMA clip itself (`audio/mpeg`), with HTTP Range support so an
`<audio>` element can seek. Backed by `modeling.catalogue.local_audio_path`:
the 128 kbps loudness-matched clip in `data/web_audio/` (the file the cloud
serves) if present, else the original in `data/fma_medium/`. 404s if the
track id isn't in the catalogue.

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
