"""Space entrypoint: fetch the embeddings from the private dataset, then serve.

The embeddings are downloaded at startup rather than baked into the image: the
image of a public Space can be pulled by anyone, the dataset is private.
Needs the HF_TOKEN secret (read access to HF_DATA_REPO).
"""

import os

from huggingface_hub import hf_hub_download

REPO = os.environ["HF_DATA_REPO"]
for env, filename in (("EFFNET_EMBEDDINGS_PATH", "embeddings/effnet_embeddings_medium.npz"),
                      ("CLAP_EMBEDDINGS_PATH", "embeddings/clap_embeddings_medium.npz")):
    os.environ[env] = hf_hub_download(REPO, filename, repo_type="dataset")

os.execvp("uvicorn", ["uvicorn", "app_effnet:app", "--host", "0.0.0.0", "--port", "7860"])
