"""Discogs-EffNet embeddings (1280-dim) for FMA tracks and outside audio.

A pretrained style-classification model repurposed for similarity -- the same
approach cosine.club is built on. Its last pooling layer, before the
classification head, turns out to encode enough about how a track sounds that
cosine distance between two tracks' vectors there is a reasonable similarity
score, even though nothing was ever trained for that.

This runs under the separate `.venv-essentia` environment (Python 3.11 --
essentia-tensorflow has no 3.13 wheel as of this writing) and needs no other
project dependency, just essentia + numpy + pandas:

    .venv-essentia/bin/python -m modeling.essentia_embed \\
        --track-list data/processed/essentia_handoff/sample_tracks.csv \\
        --out data/effnet_embeddings.npz

Model: discogs-effnet-bs64-1.pb (essentia.upf.edu/models), CC BY-NC-SA 4.0 --
non-commercial, fine for a capstone, not fine to ship in a commercial product
without a separate license from UPF.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PB = REPO_ROOT / "models" / "essentia" / "discogs-effnet-bs64-1.pb"
AUDIO_DIR = REPO_ROOT / "data" / "fma_small"
SAMPLE_RATE = 16000  # what this model was trained at; not negotiable

_model = None


def _get_model():
    """Loaded once per process -- the graph load is the slow part."""
    global _model
    if _model is None:
        import essentia.standard as es
        _model = es.TensorflowPredictEffnetDiscogs(graphFilename=str(MODEL_PB), output="PartitionedCall:1")
    return _model


def audio_path(track_id: int) -> Path:
    """FMA stores 000123.mp3 under a directory named after its first 3 digits."""
    name = f"{int(track_id):06d}"
    return AUDIO_DIR / name[:3] / f"{name}.mp3"


def embed_file(path) -> np.ndarray | None:
    """One 1280-dim vector for a file, mean-pooled over its time frames.

    The model reads ~3s windows internally and returns one embedding per
    window; mean-pooling to a single vector per track is what cosine.club
    itself does, per the creator's own description of the pipeline.
    """
    import essentia.standard as es

    try:
        audio = es.MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE)()
    except Exception as exc:  # noqa: BLE001 -- one bad file shouldn't end the run
        logger.warning("%s did not decode: %s", path, exc)
        return None
    if len(audio) < SAMPLE_RATE:
        logger.warning("%s is too short (%.2fs)", path, len(audio) / SAMPLE_RATE)
        return None

    try:
        frames = np.asarray(_get_model()(audio))
    except Exception as exc:  # noqa: BLE001 -- inference failing on one track shouldn't end the run
        logger.warning("%s: inference failed: %s", path, exc)
        return None
    if frames.size == 0 or not np.isfinite(frames).all():
        logger.warning("%s produced no usable embedding", path)
        return None
    return frames.mean(axis=0).astype(np.float32)


def _save_checkpoint(path: Path, ids: list, embeds: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=np.int64),
              embeddings=np.vstack(embeds) if embeds else np.empty((0, 1280), dtype=np.float32))


def embed_catalogue(track_ids, log_every: int = 50, checkpoint_path: Path = None, checkpoint_every: int = 250,
                     resume_ids: list = None, resume_embeds: list = None):
    """Every track in track_ids, in order. Tracks that fail are dropped.

    Writes to `checkpoint_path` every `checkpoint_every` tracks (when given),
    so a crash partway through a multi-hour run (like the one that motivated
    this -- an unhandled type from one track killed a ~55%-complete run with
    nothing saved) loses at most `checkpoint_every` tracks of work, not all of
    it. `resume_ids`/`resume_embeds` seed the result with a previous partial
    run (see `_main`'s resume-from-`--out` handling), so re-running after a
    crash only redoes the tracks that were never finished.
    """
    ids, embeds = list(resume_ids or []), list(resume_embeds or [])
    for i, tid in enumerate(track_ids, 1):
        v = embed_file(audio_path(tid))
        if v is not None:
            ids.append(int(tid))
            embeds.append(v)
        if i % log_every == 0:
            logger.info("%s / %s tracks", i, len(track_ids))
        if checkpoint_path is not None and i % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, ids, embeds)
            logger.info("checkpointed %s tracks to %s", len(ids), checkpoint_path)
    return np.array(ids), np.vstack(embeds) if embeds else np.empty((0, 1280), dtype=np.float32)


def _main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-list", type=Path, required=True,
                         help="csv with an index column of FMA track ids")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "effnet_embeddings.npz")
    parser.add_argument("--limit", type=int, help="only the first N tracks")
    parser.add_argument("--checkpoint-every", type=int, default=250,
                         help="write partial progress to --out this often, in tracks")
    parser.add_argument("--no-resume", action="store_true",
                         help="ignore an existing --out and start over, instead of resuming")
    args = parser.parse_args()

    track_ids = pd.read_csv(args.track_list, index_col=0).index.tolist()
    if args.limit:
        track_ids = track_ids[: args.limit]

    resume_ids, resume_embeds = [], []
    if args.out.exists() and not args.no_resume:
        prior = np.load(args.out)
        resume_ids = list(prior["ids"])
        resume_embeds = list(prior["embeddings"])
        done = set(int(t) for t in resume_ids)
        before = len(track_ids)
        track_ids = [t for t in track_ids if int(t) not in done]
        logger.info("resuming from %s: %s tracks already embedded, %s remaining (of %s)",
                    args.out, len(done), len(track_ids), before)

    logger.info("Embedding %s tracks with Discogs-EffNet", len(track_ids))
    ids, embeds = embed_catalogue(track_ids, checkpoint_path=args.out, checkpoint_every=args.checkpoint_every,
                                   resume_ids=resume_ids, resume_embeds=resume_embeds)

    failed = (len(track_ids) + len(resume_ids)) - len(ids)
    if failed:
        logger.warning("%s tracks failed to embed", failed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=ids, embeddings=embeds)
    logger.info("Wrote %s tracks x %s dims to %s", embeds.shape[0], embeds.shape[1], args.out)


if __name__ == "__main__":
    _main()
