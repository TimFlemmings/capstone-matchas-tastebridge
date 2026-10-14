"""Discogs-EffNet embeddings for the external seed and its 18 Spotify/YouTube
recommendations -- the other half of the handoff notebooks 08/09 read from
disk (`modeling.build_handoff` does the heard-space half). Runs under
`.venv-essentia` (needs `essentia`), unlike `build_handoff.py`.

    .venv-essentia/bin/python -m modeling.embed_external

Writes data/processed/essentia_handoff/external_effnet_embeddings.npz with
three parallel arrays: names, platforms ("seed"/"spotify_tests"/"youtube_tests"),
embeddings (1280-dim each).
"""

import logging
from pathlib import Path

import numpy as np

from modeling.essentia_embed import embed_file

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "processed" / "essentia_handoff" / "external_effnet_embeddings.npz"
SEED_PATH = REPO_ROOT / "data" / "raw" / "youtube_tests" / "Y2Mate.is - Elyjah Slaps the Space.mp3"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    paths = [SEED_PATH]
    paths += sorted((REPO_ROOT / "data" / "raw" / "spotify_tests").glob("*.mp3"))
    paths += sorted((REPO_ROOT / "data" / "raw" / "youtube_tests").glob("*.mp3"))
    # the seed's own copy sits in youtube_tests too (both platforms were given
    # the same track) -- keep the first occurrence (labelled "seed"), drop the
    # duplicate rather than embedding it twice under a platform label
    seen = set()
    deduped = []
    for p in paths:
        if p.name in seen:
            continue
        seen.add(p.name)
        deduped.append(p)
    paths = deduped

    names, platforms, vecs = [], [], []
    for i, p in enumerate(paths):
        v = embed_file(p)
        if v is None:
            logger.warning("failed to embed %s", p)
            continue
        names.append(p.name)
        platforms.append("seed" if p == SEED_PATH else p.parent.name)
        vecs.append(v)
        logger.info("embedded %s / %s: %s", i + 1, len(paths), p.name)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, names=np.array(names), platforms=np.array(platforms), embeddings=np.vstack(vecs))
    logger.info("wrote %s tracks to %s", len(names), OUT_PATH)


if __name__ == "__main__":
    main()
