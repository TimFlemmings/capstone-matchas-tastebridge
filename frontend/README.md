# Tastebridge Frontend

The React/Vite interface is intentionally separate from the Python recommender. It currently runs against clearly labeled preview data. Set `VITE_RECOMMENDER_API_URL` to use a compatible service; only `src/api/recommendations.ts` owns the HTTP integration.

## Run

```bash
npm install
npm run dev
```

## EffNet App

For complete setup instructions, including the required local embedding and FMA files, see [`TIM_SETUP.md`](../TIM_SETUP.md).

The separate EffNet backend uses `data/effnet_embeddings_full.npz` plus local FMA data. Set `FMA_DATA_DIR` and `EFFNET_EMBEDDINGS_PATH` for your machine, then start it from the repository root:

```bash
uv run uvicorn app_effnet:app --host 0.0.0.0 --port 8001
```

In another terminal, start the same interface against that backend:

```bash
npm run dev:effnet --prefix frontend
```

Open `http://localhost:5174`. The existing PCA app remains on ports 8000 and 5173.

For the local FMA-backed API, start Uvicorn from the repository root:

```bash
uv run uvicorn app:app --reload --port 8000
```

By default, the API looks for a sibling `capstone-matchas/data` directory and its `models/` directory. Override `FMA_DATA_DIR`, `FMA_AUDIO_DIR`, `FMA_METADATA_DIR`, or `FMA_MODELS_DIR` when your data is stored elsewhere. The API builds its 8,000-track PCA recommender during startup and serves the matching MP3 at `/audio/{track_id}`.

## API contract

The adapter expects `GET /tracks?query=<text>` to return `{ "tracks": Track[] }`, and `POST /recommendations` to accept `{ "track_id": string, "adventure": number, "limit": number }` and return `{ "recommendations": Recommendation[] }`.

Track fields are `id`, `title`, `artist`, `genre`, `duration`, and `coverUrl`. Recommendation fields add `rank` and `distance`. The UI forwards the adventure value and renders the returned ordering; it does not implement model ranking.