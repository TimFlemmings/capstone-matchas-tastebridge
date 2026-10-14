"""CLAP embeddings (512-dim) for FMA tracks, so tracks can be searched with text.

CLAP puts audio and text in one space: "jazz with saxophone and swing drums"
lands near tracks that sound like that. We store one vector per track; at query
time only the text side runs, and search is a dot product against these.

Checkpoint: laion/larger_clap_general. Benchmarked against the alternatives on
300 fma_medium tracks across 6 genres (zero-shot genre accuracy from a text
prompt): larger_clap_general 0.79, clap-htsat-unfused 0.76, and
larger_clap_music 0.09 -- below chance. The "music" checkpoint collapses every
track to nearly the same vector under transformers 5, silently, so don't
switch to it without re-running that check.

Runs in the main environment (torch + transformers), on Apple's GPU if present:

    uv run python -m modeling.clap_embed \\
        --track-list data/processed/essentia_handoff/medium_all_tracks.csv \\
        --out data/clap_embeddings_medium.npz
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "laion/larger_clap_general"
AUDIO_DIR = REPO_ROOT / "data" / "fma_medium"
SAMPLE_RATE = 48000  # what CLAP was trained at
WINDOW = 10 * SAMPLE_RATE  # CLAP reads 10s at a time; a 30s clip gives 3 windows
DIM = 512

_model = None


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_model():
    """(model, processor), loaded once per process."""
    global _model
    if _model is None:
        from transformers import ClapModel, ClapProcessor
        _model = (ClapModel.from_pretrained(MODEL_NAME).to(_device()).eval(), ClapProcessor.from_pretrained(MODEL_NAME))
    return _model


def audio_path(track_id: int) -> Path:
    """FMA stores 000123.mp3 under a directory named after its first 3 digits."""
    name = f"{int(track_id):06d}"
    return AUDIO_DIR / name[:3] / f"{name}.mp3"


def embed_file(path) -> np.ndarray | None:
    """One unit-length 512-dim vector: the mean of the clip's 10s windows, renormalised."""
    import librosa
    import torch

    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 -- one bad file shouldn't end the run
        logger.warning("%s did not decode: %s", path, exc)
        return None
    if len(audio) < SAMPLE_RATE:
        logger.warning("%s is too short (%.2fs)", path, len(audio) / SAMPLE_RATE)
        return None

    windows = [audio[i:i + WINDOW] for i in range(0, max(len(audio) - WINDOW, 0) + 1, WINDOW)][:3]
    model, processor = get_model()
    try:
        with torch.no_grad():
            inputs = processor(audio=windows, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(_device())
            frames = model.get_audio_features(**inputs).pooler_output.cpu().numpy()
    except Exception as exc:  # noqa: BLE001 -- inference failing on one track shouldn't end the run
        logger.warning("%s: inference failed: %s", path, exc)
        return None
    v = frames.mean(axis=0)
    if not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        logger.warning("%s produced no usable embedding", path)
        return None
    return (v / np.linalg.norm(v)).astype(np.float32)


def embed_text(queries: list[str]) -> np.ndarray:
    """Unit-length text vectors; score tracks with `embed_text([q]) @ embeddings.T`."""
    import torch

    model, processor = get_model()
    with torch.no_grad():
        inputs = processor(text=queries, return_tensors="pt", padding=True).to(_device())
        return model.get_text_features(**inputs).pooler_output.cpu().numpy().astype(np.float32)


def _save_checkpoint(path: Path, ids: list, embeds: list):
    """Write to a temp file, then rename over `path` -- a crash mid-write leaves the old checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(ids, dtype=np.int64),
             embeddings=np.vstack(embeds) if embeds else np.empty((0, DIM), dtype=np.float32))
    os.replace(tmp, path)


def _main():
    global AUDIO_DIR
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-list", type=Path, required=True,
                        help="csv with an index column of FMA track ids")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "clap_embeddings_medium.npz")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--limit", type=int, help="only the first N tracks")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing --out and start over, instead of resuming")
    args = parser.parse_args()
    AUDIO_DIR = args.audio_dir

    track_ids = pd.read_csv(args.track_list, index_col=0).index.tolist()
    if args.limit:
        track_ids = track_ids[: args.limit]

    ids, embeds = [], []
    if args.out.exists() and not args.no_resume:
        prior = np.load(args.out)
        ids, embeds = list(prior["ids"]), list(prior["embeddings"])
        done = set(int(t) for t in ids)
        track_ids = [t for t in track_ids if int(t) not in done]
        logger.info("resuming from %s: %s done, %s remaining", args.out, len(done), len(track_ids))

    logger.info("Embedding %s tracks with %s on %s", len(track_ids), MODEL_NAME, _device())
    failed = 0
    for i, tid in enumerate(track_ids, 1):
        v = embed_file(audio_path(tid))
        if v is None:
            failed += 1
        else:
            ids.append(int(tid))
            embeds.append(v)
        if i % 100 == 0:
            logger.info("%s / %s tracks", i, len(track_ids))
        if i % args.checkpoint_every == 0:
            _save_checkpoint(args.out, ids, embeds)

    if failed:
        logger.warning("%s tracks failed to embed", failed)
    _save_checkpoint(args.out, ids, embeds)
    logger.info("Wrote %s tracks x %s dims to %s", len(ids), DIM, args.out)


if __name__ == "__main__":
    _main()
