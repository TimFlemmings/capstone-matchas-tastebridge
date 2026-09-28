# Getting notebooks 08 and 09 running on a fresh machine

Notebooks 06 and 07 need nothing beyond the project's usual `uv sync` and the
FMA data you already have. **08 and 09 are different** — they use Discogs-EffNet
via Essentia, which has no Python 3.13 wheel yet, so they run in a second,
separate environment alongside the main one. Nothing here touches or breaks
your existing setup; it's purely additive.

## 1. A second environment, Python 3.11

```bash
uv venv --python 3.11.16 .venv-essentia
uv pip install --python .venv-essentia/bin/python \
    essentia-tensorflow numpy pandas scikit-learn joblib soundfile pycountry "ipykernel<7"
```

The `ipykernel<7` pin matters, not just a version preference: with ipykernel
7.x, loading the TensorFlow model for the first time *inside a live Jupyter
kernel* hangs indefinitely — reproduced twice, confirmed by running the exact
same code as a plain script (works in 1-3 seconds every time). Downgrading is
the fix that actually worked; nothing else we tried did.

Register it as a Jupyter kernel so the notebooks can find it:

```bash
.venv-essentia/bin/python -m ipykernel install --user --name essentia311 \
    --display-name "Python 3.11 (essentia)"
```

## 2. The model weights (not in git — 18MB, license restricts redistribution)

```bash
mkdir -p models/essentia
curl -fL -o models/essentia/discogs-effnet-bs64-1.pb \
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
curl -fL -o models/essentia/discogs-effnet-bs64-1.json \
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json"
```

Essentia's server is slow and occasionally drops the connection partway
through — if `curl` reports a short/partial file, resume rather than restart:

```bash
curl -fL -C - -o models/essentia/discogs-effnet-bs64-1.pb \
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
```

Verify the size before trusting it — a truncated file loads *without an
error* and then fails confusingly later on first use:

```bash
ls -la models/essentia/discogs-effnet-bs64-1.pb   # should be 18,366,619 bytes
```

License: CC BY-NC-SA 4.0 — non-commercial, fine for the capstone, not fine to
ship in a commercial product without a separate license from UPF.

## 3. Data layout Essentia's code expects

Same FMA data you already have, just needs to also be reachable at these
paths (symlinks, not copies — these are gitignored, recreate them locally):

```bash
ln -s raw/fma_metadata data/fma_metadata
ln -s raw/fma_small data/fma_small
```

## 4. Precomputed feature files (all gitignored, regenerate locally)

If you don't already have these from working on 06/07:

```bash
uv run python -m modeling.rhythm         # ~90s
uv run python -m modeling.separation     # ~15-20 min, the slow one
uv run python -m modeling.geography      # <1 min, deterministic
```

## 5. The handoff data notebooks 08/09 read instead of recomputing live

Two scripts, two environments — run both:

```bash
# main environment (3.13) -- heard-space side of the handoff
uv run python -m modeling.build_handoff

# essentia environment (3.11) -- EffNet side of the handoff
.venv-essentia/bin/python -m modeling.embed_external
```

## 6. The actual embeddings

```bash
# ~2,001-track stratified sample, used early in notebook 08 -- ~25 min
.venv-essentia/bin/python -m modeling.essentia_embed \
    --track-list data/processed/essentia_handoff/sample_tracks.csv \
    --out data/effnet_embeddings.npz

# the full 7,994-track catalogue, used everywhere after that -- ~1h45m
.venv-essentia/bin/python -m modeling.essentia_embed \
    --track-list data/processed/essentia_handoff/all_small_tracks.csv \
    --out data/effnet_embeddings_full.npz
```

Both checkpoint progress to `--out` every 250 tracks and resume automatically
if interrupted — safe to Ctrl-C and rerun the same command later; it picks up
where it left off rather than starting over. (`--no-resume` forces a clean
restart if you ever want one.)

## 7. Then just open the notebooks

VS Code should offer the "Python 3.11 (essentia)" kernel for 08/09
automatically. **One thing that bit us**: if a notebook tab is left open
across multiple external re-executions (e.g. from the CLI, or from someone
else's `uv run jupyter nbconvert`), the editor can end up holding a stale
in-memory copy and silently overwrite a freshly-rebuilt file with that stale
one on save — including corrupting the kernelspec metadata, which then makes
the notebook silently run under the *wrong* Python and fail with
`ModuleNotFoundError: No module named 'essentia'`. If a notebook stops showing
outputs or throws that error after re-running it externally, close the tab
completely (don't save) and reopen fresh before assuming something's broken.

## Everything above, roughly in order, total time

| step | time |
|---|---:|
| env + model download | ~5-20 min (model server is slow) |
| rhythm + separation extraction | ~20 min |
| handoff scripts | ~2 min |
| sample embeddings (2,001 tracks) | ~25 min |
| full embeddings (7,994 tracks) | ~1h45m |

The full embeddings are the only genuinely long step — everything else is
comfortably a coffee break.
