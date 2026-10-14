# Tastey front end

The React + Vite app. It does no ranking itself: it sends the seed and the dial position to the server and shows what comes back. All HTTP calls live in [`src/api/recommendations.ts`](src/api/recommendations.ts).

## Run

Start the server first (see [`SETUP.md`](../SETUP.md)), then:

```bash
npm ci
cp .env.effnet.example .env.effnet     # VITE_RECOMMENDER_API_URL=http://localhost:8001
npm run dev:effnet                     # http://localhost:5174
```

Plain `npm run dev` (port 5173) talks to `http://localhost:8000`, the lighter [`api/main.py`](../api/main.py), unless `VITE_RECOMMENDER_API_URL` says otherwise. Opened from any other address with no API set, the app falls back to a clearly labelled **preview mode**: 8 sample tracks with demo audio, handy for layout work. `npm run build` type-checks and builds to `dist/`. The hosted app is built with `VITE_RECOMMENDER_API_URL=/`, meaning "the same site that serves this page" (see `deploy/build_space.py`).

## Where things are

| File | What it does |
|---|---|
| `src/App.tsx` | Pages (Discover, Library, Taste Map), the player, the play queue, saving and seeding |
| `src/state/store.ts` | Redux state: playback, the journey (Taste Map), the Library; journey and Library saved to `localStorage` |
| `src/api/recommendations.ts` | The API client, `Track` / `Recommendation` types, preview data |
| `src/components/Onboarding.tsx` | First-visit picker and "Start somewhere new" |
| `src/components/TasteTree.tsx`, `JourneyMap.tsx` | Taste Map: Tree view and Map view |
| `src/components/Waveform.tsx` | Waveform seek bar, decoded from the clip in the browser |
| `src/components/SearchBox.tsx` | Search with its results dropdown |
| `src/components/SpaceBackdrop.tsx`, `icons.tsx` | Background art, the logo, play and seed icons |
| `src/connections.ts` | Turns the server's connecting factors into words ("same era", "24 bridgons") |
| `src/styles.css` | All styles; colours are CSS variables at the top (`--accent`, `--accent-2`, `--accent-gradient`) |

## API it expects

| Endpoint | Returns |
|---|---|
| `GET /tracks?query=<text>` | `{ tracks: Track[] }`: by name, genre, or a described sound |
| `GET /tracks/by-id?ids=1,2,3` | `{ tracks: Track[] }` in that order |
| `POST /recommendations` `{ track_id, adventure }` | `{ recommendations: Recommendation[] }`: 15 picks; `adventure` 0, 0.5 or 1 for Home, Bridge, Adventure |
| `GET /journey-map?ids=…` | `{ points: [{ id, x, y }] }`: 2D layout for the Map view |
| `GET /audio/{id}` | the clip, with Range support for seeking |

A `Track` has `id`, `title`, `artist`, `genre`, `duration`, `coverUrl`, `audioUrl`, and where known `location`, `dateCreated` and `bpm`. A `Recommendation` adds `rank`, `distance` and `connectingFactors`. Cover images are placeholders, since FMA ships no artwork.
