"""Assemble the hosted app (a Hugging Face Docker Space) and optionally push it.

    uv run python deploy/build_space.py                      # build into build/space/ only
    uv run python deploy/build_space.py --push USER/tastebridge

The Space holds the server code, the built frontend and a slim copy of the FMA
metadata (only the catalogue's tracks and the columns the server reads: ~250 MB
-> a few MB). It never holds the embeddings or audio: `start.py` downloads the
embeddings from the private dataset at startup and `/audio` streams the clips
from it, both with the Space's HF_TOKEN secret.

Pushing needs `uv run hf auth login` first, with write access to the Space's
owner. The HF_TOKEN secret is set by hand in the Space's settings (see the
message printed after a push) -- this script never uploads a token.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "build" / "space"
METADATA = REPO_ROOT / "data" / "fma_metadata"
DATA_REPO = "mrln77/tastebridge-audio"
# every tracks.csv column the server reads (app_effnet, modeling.catalogue,
# connecting_factors, text_search, tiers)
COLUMNS = [("set", "subset"), ("track", "title"), ("track", "genre_top"), ("track", "genres_all"),
           ("track", "duration"), ("track", "listens"), ("track", "date_created"), ("artist", "name"), ("artist", "location"),
           ("album", "date_released"), ("album", "date_created")]
# imported inside functions, so importing app_effnet doesn't reveal them
LAZY_MODULES = {"geography"}


def runtime_modules() -> set:
    """The modeling/ modules the server needs: whatever importing it pulls in, plus LAZY_MODULES."""
    found = subprocess.run(
        [sys.executable, "-c", "import sys, app_effnet; "
                               "print(' '.join(m[9:] for m in sys.modules if m.startswith('modeling.')))"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()
    return set(found) | LAZY_MODULES


def unlicensed_ids() -> list:
    """Track ids the dataset's attribution.csv lists without a licence."""
    from huggingface_hub import hf_hub_download

    attribution = pd.read_csv(hf_hub_download(DATA_REPO, "attribution.csv", repo_type="dataset"))
    return attribution.loc[attribution["license"].isna(), attribution.columns[0]].astype(int).tolist()


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "modeling").mkdir(parents=True)
    (OUT / "data" / "fma_metadata").mkdir(parents=True)
    (OUT / "data" / "processed").mkdir(parents=True)

    for name in ("Dockerfile", "requirements.txt", "start.py", "README.md"):
        shutil.copy(REPO_ROOT / "deploy" / "space" / name, OUT / name)
    shutil.copy(REPO_ROOT / "app_effnet.py", OUT / "app_effnet.py")
    for module in sorted(runtime_modules()):
        shutil.copy(REPO_ROOT / "modeling" / f"{module}.py", OUT / "modeling" / f"{module}.py")
    (OUT / "modeling" / "__init__.py").touch()

    tracks = pd.read_csv(METADATA / "tracks.csv", index_col=0, header=[0, 1])
    tracks = tracks.loc[tracks[("set", "subset")].isin(["small", "medium"]), COLUMNS]
    # the Space is public: tracks whose FMA licence is unknown stay out of it
    unlicensed = unlicensed_ids()
    tracks = tracks.drop(index=unlicensed, errors="ignore")
    print(f"Left out {len(unlicensed)} tracks without a licence")
    tracks.to_csv(OUT / "data" / "fma_metadata" / "tracks.csv")
    shutil.copy(METADATA / "genres.csv", OUT / "data" / "fma_metadata" / "genres.csv")
    shutil.copy(REPO_ROOT / "data" / "artist_countries.csv", OUT / "data" / "artist_countries.csv")
    for name in ("duplicates_medium.csv", "track_tempo.csv"):
        shutil.copy(REPO_ROOT / "data" / "processed" / name, OUT / "data" / "processed" / name)

    # "/" = the API is the same site that serves the page (env beats .env files in Vite)
    env = {**os.environ, "VITE_RECOMMENDER_API_URL": "/"}
    subprocess.run(["npm", "run", "build", "--", "--outDir", str(OUT / "frontend" / "dist"), "--emptyOutDir"],
                   cwd=REPO_ROOT / "frontend", env=env, check=True)

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Built {OUT} ({size / 1e6:.1f} MB, {len(tracks):,} tracks of metadata)")


def push(space_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.upload_folder(repo_id=space_id, repo_type="space", folder_path=OUT,
                      commit_message="Deploy TasteBridge from build/space")
    print(f"Pushed. https://huggingface.co/spaces/{space_id}\n"
          f"If the HF_TOKEN secret isn't set yet: Settings -> Variables and secrets -> New secret, "
          f"name HF_TOKEN, value a read token with access to {DATA_REPO}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", metavar="USER/SPACE", help="also create/update this Space")
    args = parser.parse_args()
    build()
    if args.push:
        push(args.push)
