# Run Tastey locally

The app is two processes: the FastAPI server `app_effnet.py` (recommendations, search, audio) and the React front end in `frontend/`. Allow about 15 minutes the first time.

**You need:** [uv](https://docs.astral.sh/uv/) (see [UV_SETUP.md](UV_SETUP.md)), Node.js 20 or newer, and read access to the private Hugging Face dataset `mrln77/tastebridge-audio` (ask Tim).

## 1. Install

```bash
uv sync
cd frontend && npm ci && cd ..
```

## 2. Get the data

Large files are not in Git. Three things go into `data/`:

**FMA metadata** (`tracks.csv`, `genres.csv`): download [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) (342 MB) and unzip it so the files sit in `data/fma_metadata/`.

**Embeddings** (EffNet for recommendations, CLAP for text search), from the private dataset:

```bash
uv run hf auth login        # a token with read access to mrln77/tastebridge-audio
uv run hf download mrln77/tastebridge-audio --repo-type dataset \
    --include "embeddings/*" --local-dir /tmp/tb && mv /tmp/tb/embeddings/*.npz data/
```

This gives `data/effnet_embeddings_medium.npz` and `data/clap_embeddings_medium.npz`.

**Artist countries** (optional, enables the "same region" connection):

```bash
uv run python -m modeling.geography      # writes data/artist_countries.csv
```

Already in Git: `data/processed/duplicates_medium.csv` (re-uploads and silent clips to drop) and `data/processed/track_tempo.csv` (BPM).

## 3. Start the server

The simplest way streams audio from the dataset, so you don't need the 12 GB of clips:

```bash
export HF_AUDIO_REPO=mrln77/tastebridge-audio HF_AUDIO_PREFIX=""
uv run uvicorn app_effnet:app --port 8001
```

It uses the token from `hf auth login` (or an `HF_TOKEN` environment variable). Startup takes about a minute while it loads the catalogue and the text-search model. Check it:

```bash
curl http://localhost:8001/health
# {"status":"ok","model":"Discogs-EffNet","tracks":24781,"embedding_dimensions":1280}
```

**Audio on disk instead:** leave out the two `export`s and either put the 128 kbps web clips in `data/web_audio/` (`000/000002.mp3`, …) or point `FMA_AUDIO_DIR` at an `fma_medium/` folder of original mp3s.

## 4. Start the front end

In a second terminal:

```bash
cd frontend
cp .env.effnet.example .env.effnet     # points the app at http://localhost:8001
npm run dev:effnet
```

Open http://localhost:5174.

## Paths and settings

| Variable | Default | What it sets |
|---|---|---|
| `FMA_DATA_DIR` | `data/` (or a sibling `../capstone-matchas/data` if it exists) | where `fma_metadata/` lives |
| `EFFNET_EMBEDDINGS_PATH`, `CLAP_EMBEDDINGS_PATH` | `data/*_embeddings_medium.npz` | the two embedding files |
| `HF_AUDIO_REPO`, `HF_AUDIO_PREFIX` | unset | stream audio from the dataset instead of disk |
| `API_PUBLIC_BASE` | `http://localhost:8001` | the base of the audio links the server hands out |

## Rebuilding the data from scratch

The embeddings are recomputed from audio by `modeling/essentia_embed.py` and `modeling/clap_embed.py` (about 1.5 hours together), the web clips by `modeling/transcode_audio.py`, and the duplicates list by `uv run python -m modeling.catalogue`. Notebook 11 checks the result.

## Deploying

See [README.md → Deploy](README.md#deploy).
