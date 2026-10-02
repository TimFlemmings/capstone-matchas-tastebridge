"""Smaller, loudness-matched copies of the FMA clips, for serving from the cloud.

The originals are ~0.93 MB each (mostly 320 kbps), 23 GB for fma_medium. For
30-second previews played in a browser, 128 kbps stereo MP3 is indistinguishable
on laptop speakers and earbuds and cuts that to ~0.47 MB a clip (~12 GB total).

Each clip is also loudness-normalised (EBU R128, -16 LUFS, the usual target for
web/podcast audio). FMA tracks vary wildly in level; without this, a listener
clicking through recommendations keeps reaching for the volume, which reads as
"the recommendations are jarring" when it's really just mastering.

Embeddings are always computed from the originals, never from these files.

Resumable and crash-safe: existing outputs are skipped, and every file is
written under a temp name and renamed into place only once ffmpeg succeeds.

    uv run python -m modeling.transcode_audio --track-list data/processed/essentia_handoff/medium_all_tracks.csv
"""

import argparse
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "data" / "fma_medium"
OUT_DIR = REPO_ROOT / "data" / "web_audio"
BITRATE = "128k"


def _rel(track_id: int) -> Path:
    name = f"{int(track_id):06d}"
    return Path(name[:3]) / f"{name}.mp3"


def transcode(track_id: int, src_dir: Path = SRC_DIR, out_dir: Path = OUT_DIR) -> str:
    """'done', 'skipped' or 'failed: ...' for one track."""
    src, dst = src_dir / _rel(track_id), out_dir / _rel(track_id)
    if dst.exists():
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp3")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map_metadata", "-1",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "2", "-b:a", BITRATE, str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 10_000:
        tmp.unlink(missing_ok=True)
        return f"failed: {result.stderr.strip()[-200:]}"
    os.replace(tmp, dst)
    return "done"


def _main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-list", type=Path, required=True)
    parser.add_argument("--src-dir", type=Path, default=SRC_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    track_ids = pd.read_csv(args.track_list, index_col=0).index.tolist()[: args.limit]
    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(transcode, track_ids, [args.src_dir] * len(track_ids), [args.out_dir] * len(track_ids),
                           chunksize=16)
        for i, (tid, status) in enumerate(zip(track_ids, results), 1):
            counts[status.split(":")[0]] += 1
            if status.startswith("failed"):
                logger.warning("%s %s", tid, status)
            if i % 1000 == 0:
                logger.info("%s / %s  %s", i, len(track_ids), counts)
    logger.info("finished: %s", counts)


if __name__ == "__main__":
    _main()
