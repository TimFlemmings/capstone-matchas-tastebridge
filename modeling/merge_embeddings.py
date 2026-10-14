"""Merge EffNet shards with the fma_small embeddings into one fma_medium file, and check both outputs.

Refuses to write if any check fails: duplicate ids, non-finite values, ids
outside fma_medium, or a dimension mismatch. Tracks with no embedding (FMA
ships a handful of truncated files) are listed, not treated as errors.

    uv run python -m modeling.merge_embeddings
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
TRACK_LIST = DATA / "processed" / "essentia_handoff" / "medium_all_tracks.csv"
EFFNET_SMALL = DATA / "effnet_embeddings_full.npz"
EFFNET_SHARDS = sorted((DATA / "embeddings_partial").glob("effnet_medium_shard*.npz"))
EFFNET_OUT = DATA / "effnet_embeddings_medium.npz"
CLAP = DATA / "clap_embeddings_medium.npz"


def check(name, ids, emb, dim, expected):
    problems = []
    if len(ids) != len(emb):
        problems.append(f"{len(ids)} ids but {len(emb)} rows")
    if len(set(ids.tolist())) != len(ids):
        problems.append(f"{len(ids) - len(set(ids.tolist()))} duplicate ids")
    if emb.shape[1] != dim:
        problems.append(f"dim {emb.shape[1]}, expected {dim}")
    if not np.isfinite(emb).all():
        problems.append("non-finite values")
    if (np.linalg.norm(emb, axis=1) == 0).any():
        problems.append("all-zero rows")
    stray = set(ids.tolist()) - expected
    if stray:
        problems.append(f"{len(stray)} ids not in fma_medium")
    missing = sorted(expected - set(ids.tolist()))
    logger.info("%s: %s tracks x %s, %s missing %s", name, len(ids), emb.shape[1], len(missing), missing[:20])
    for p in problems:
        logger.error("%s: %s", name, p)
    return not problems


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    expected = set(pd.read_csv(TRACK_LIST, index_col=0).index.tolist())
    if len(EFFNET_SHARDS) != 4:
        sys.exit(f"expected 4 EffNet shards, found {len(EFFNET_SHARDS)}")

    parts = [np.load(EFFNET_SMALL)] + [np.load(p) for p in EFFNET_SHARDS]
    ids = np.concatenate([p["ids"] for p in parts]).astype(np.int64)
    emb = np.vstack([p["embeddings"] for p in parts]).astype(np.float32)
    order = np.argsort(ids)
    ids, emb = ids[order], emb[order]

    ok = check("effnet", ids, emb, 1280, expected)
    clap = np.load(CLAP)
    ok &= check("clap", clap["ids"], clap["embeddings"], 512, expected)
    if not ok:
        sys.exit("checks failed -- nothing written")

    tmp = EFFNET_OUT.with_name(EFFNET_OUT.stem + ".tmp.npz")
    np.savez(tmp, ids=ids, embeddings=emb)
    tmp.replace(EFFNET_OUT)
    logger.info("wrote %s", EFFNET_OUT)


if __name__ == "__main__":
    main()
