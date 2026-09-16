import logging
from pathlib import Path

import parsenvy

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# mlflow
# sometimes when saving links in text.. there is a new line .. strip removes that
# the path is resolved from this file, so it works from notebooks/ as well as the repo root
try:
    TRACKING_URI = (REPO_ROOT / ".mlflow_uri").read_text().strip()
except OSError:
    TRACKING_URI = parsenvy.str("MLFLOW_URI")

# experiment of the neuefische coffee template (modeling/train.py, modeling/predict.py)
EXPERIMENT_NAME = "0-template-ds-modeling"

# experiment of this capstone
MUSIC_EXPERIMENT_NAME = "Music_Recommendation_PCA"
