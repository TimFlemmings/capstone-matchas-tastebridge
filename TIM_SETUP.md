# Tastebridge EffNet App: Setup for Tim

This guide runs the separate Tastebridge frontend with the Discogs-EffNet recommender. It does not replace the existing PCA app.

## 1. Get the code

Clone or update the `frontend/tastebridge-ui` branch from the team repository. For an existing checkout:

```bash
git fetch origin
git switch frontend/tastebridge-ui
git pull --ff-only
```

Requirements: Python 3.11 or newer, `uv`, Node.js 20 or newer, and npm.

## 2. Provide the local data files

Large/private data files are intentionally not tracked by Git. The app serves the 25k-track fma_medium catalogue and needs both embedding files:

```text
<repository>/data/effnet_embeddings_medium.npz   # Discogs-EffNet, recommendations
<repository>/data/clap_embeddings_medium.npz     # CLAP, text search
```

They live in the Hugging Face dataset `mrln77/tastebridge-audio` (folder `embeddings/`), along with the 128 kbps web clips. With access to it:

```bash
uv run hf download mrln77/tastebridge-audio --repo-type dataset \
    --include "embeddings/*" --local-dir /tmp/tb && mv /tmp/tb/embeddings/*.npz data/
```

Both are rebuilt from the audio by `modeling/essentia_embed.py` and `modeling/clap_embed.py` (about 1.5 hours together). The list of re-uploads and silent tracks the catalogue removes, `data/processed/duplicates_medium.csv`, is in Git.

Each archive contains `ids` and `embeddings`: 24,982 x 1,280 for EffNet, 24,980 x 512 for CLAP. The app uses the tracks present in both, minus duplicates: 24,896.

FMA metadata, and either FMA Medium audio or the web clips in `data/web_audio/`, must also be available locally. The directory supplied to `FMA_DATA_DIR` must have this layout:

```text
<FMA_DATA_DIR>/
  fma_medium/         # 000/000002.mp3, ... (not needed if data/web_audio/ exists)
  fma_metadata/
    tracks.csv
```

FMA files are not in this Git branch. Use the team's approved shared storage or the copy already on your machine. Do not commit the dataset or embedding archive to Git.

## 3. Start the EffNet API

From the repository root, set the paths for your machine. Keep `FMA_DATA_DIR` pointed at the `data` directory that contains both `fma_medium` and `fma_metadata`:

```bash
export FMA_DATA_DIR="/absolute/path/to/capstone-matchas/data"
export API_PUBLIC_BASE="http://localhost:8001"
uv sync
uv run uvicorn app_effnet:app --host 127.0.0.1 --port 8001
```

Wait for Uvicorn to report that application startup completed. In another terminal, verify:

```bash
curl http://localhost:8001/health
```

Expected fields include `"status":"ok"`, `"model":"Discogs-EffNet"`, `"tracks":7994`, and `"embedding_dimensions":1280`.

The API provides `GET /tracks`, `POST /recommendations`, and `GET /audio/{track_id}`.

## 4. Start the EffNet frontend

In a second terminal:

```bash
cd frontend
npm ci
cp .env.effnet.example .env.effnet
npm run dev:effnet
```

Open <http://localhost:5174>. The `.env.effnet` file is local-only and may be edited if the API runs on a different host or port.

The PCA app remains separate on ports 8000 and 5173.

## What uses EffNet

Recommendations use the saved 1,280-dimensional Discogs-EffNet embeddings and cosine nearest-neighbour search. No Essentia install or `.pb` model weights are needed when the precomputed embedding archive is available.
