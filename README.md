# Matcha — a music recommender you can steer

Most recommenders keep you where you already are. Matcha lets you decide how far
from your taste you want to travel, and shows you *why* an unfamiliar track
still connects to where you started.

This repository holds the recommendation engine: the feature pipeline, the
model, and the evaluation setup that every later idea is measured against.

---

## Where the project stands

**Milestone 1 — project setup: done.** The baseline works end to end: give it one
track, get a ranked list of similar ones.

```bash
uv run python -m modeling.recommender --track-id 2
```

```
seed track : AWOL — Food [Hip-Hop]
space      : 146 components, cosine distance

          distance               artist                          title          genre
7492        0.4237              BrokeMC                    In the Dark        Hip-Hop
149953      0.4253            K. Sparks                 Blue Real Talk        Hip-Hop
98584       0.4481               Hi-Rez                          Relax        Hip-Hop
115723      0.4766            Martin R.                      Acidamine     Electronic
...
```

### Baseline scorecard

Measured on held-out tracks with 5-fold cross-validation — the pipeline is
refitted per fold, so these numbers say how it behaves on music it has never
seen.

| Metric | Held out | What it asks |
|---|---|---|
| `precision@10` | **0.400** ± 0.006 | how many of the ten share the seed's genre |
| `nDCG@10` | 0.420 ± 0.006 | are the good ones near the *top* |
| `genre_overlap@10` | 0.262 ± 0.006 | how much the full genre lists overlap |
| `coverage` | 0.780 ± 0.004 | how much of the catalogue ever gets surfaced |
| random guess | 0.125 | eight balanced genres, so 1-in-8 |

**3.2× better than chance**, and the model never sees a genre label — it matches
on sound alone. Genre is only used afterwards, to mark the paper.

---

## The data

[FMA](https://github.com/mdeff/fma) `small` subset:

- **8,000 tracks**, ~229 seconds each
- **518 pre-extracted audio features** — mfcc, chroma, spectral, tonnetz, zcr, rmse
- **8 genres**, exactly 1,000 tracks each — perfectly balanced, which is why the
  random baseline is a clean `1/8`
- No missing values; 17 duplicate rows

Audio and metadata live under `data/` and are **not** in this repository — they
are 32 GB. Download them from the FMA project.

### Three things the EDA turned up

**The features are lopsided.** Three `chroma_*` families carry 252 of the 518
columns — 49% — while measuring much the same thing. Rhythm and texture (`zcr`,
`rmse`) get 7 columns each. Since PCA picks directions by variance and variance
grows with column count, the similarity space leans heavily on harmony.

**The tails are heavy.** 86 of 518 columns have kurtosis above 100, one reaches
7,599, and after standardising the most extreme value still sits 88 standard
deviations from its mean. `RobustScaler` is the obvious fix — we tried it, and
it scored *worse* (0.336 vs 0.413). Those outliers carry real signal.

**It compresses a long way.** 146 components hold 90% of the variance, 207 hold
95%, 334 hold 99%. No single component dominates: the strongest explains only
11.8%.

---

## How it works

```
518 audio features  →  StandardScaler  →  PCA (146)  →  nearest neighbours
                                                         cosine distance
```

PCA on its own recommends nothing — it only reshapes the features. The model is
the last step: index every track by its position in the reduced space, and
answer *"what sounds like this?"* with the closest tracks to it.

The baseline is deliberately plain — no genre filtering, no popularity
weighting, no learned similarity — so that every later idea has something honest
to beat.

**Cosine rather than euclidean** because it compares the *direction* of two
feature vectors and ignores their magnitude: a loud mix and a quiet one with the
same character stay close. Worth 0.413 against 0.378 for euclidean.

---

## Repository layout

```
modeling/
├── config.py         MLflow tracking URI and experiment name
├── data.py           load the FMA subset and its metadata
├── features.py       scaling, PCA, artifact persistence
├── recommender.py    the model — nearest neighbours in PCA space
├── evaluation.py     the four metrics and k-fold validation
└── train_pca.py      CLI: fit a feature space and log the run

notebooks/
└── 01_eda_baseline.ipynb    the analysis, step by step
```

---

## Running it

### Setup

```bash
uv sync --all-groups
```

[uv](https://docs.astral.sh/uv/) installs the right Python for you — no pyenv,
conda or pip needed. Install it once per machine:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### MLflow

Save the tracking URI locally — it is gitignored, so it never reaches the repo:

```bash
echo http://127.0.0.1:5000/ > .mlflow_uri
```

Start the server in its own terminal and leave it running:

```bash
make mlflow      # uv run mlflow ui
```

> **The trap that costs an afternoon:** notebooks run with `notebooks/` as the
> working directory, and MLflow silently defaults to a local store there when no
> tracking URI is set. Runs vanish from the UI without an error. Always import
> `TRACKING_URI` from `modeling/config.py`, which resolves the path from the
> module rather than the working directory. And never run `!mlflow ui` inside a
> notebook cell — it blocks the kernel.

### Recommend

```bash
uv run python -m modeling.recommender --track-id 2          # ten similar tracks
uv run python -m modeling.recommender --evaluate            # score it
uv run python -m modeling.recommender --metric euclidean    # try another distance
```

### Fit a feature space

Every flag maps to an experiment we wanted to run, so a proposal is a flag
rather than a notebook edit. Runs land in the `Music_Recommendation_PCA`
experiment carrying all four metrics.

```bash
uv run python -m modeling.train_pca                              # baseline: 90% variance
uv run python -m modeling.train_pca --n-components 60            # a fixed, smaller space
uv run python -m modeling.train_pca --n-components 60 --scaler robust
uv run python -m modeling.train_pca --n-components 60 --drop-chroma
```

`--n-components` takes either form: `0.90` is a variance target, `60` a fixed
component count.

### Reproduce the audit

```bash
uv run python -m modeling.evaluation
```

Prints the feature-family breakdown, the tail statistics, and the full
comparison table behind every number in this README.

---

## What we measured, and what we left alone

Logged in MLflow, none applied to the baseline — it stays as the reference point
for the final comparison.

| Change | Features | Components | precision@10 |
|---|---:|---:|---:|
| Drop the two redundant chroma families | 350 | 119 | **0.429** |
| Drop chroma + kurtosis/skew columns | 250 | 78 | 0.423 |
| **Baseline** — all features | 518 | 146 | 0.413 |
| Fixed 60 components | 518 | 60 | 0.404 |
| RobustScaler instead of StandardScaler | 518 | 60 | 0.336 |

Dropping two chroma families helps *and* shrinks the space. Applying it is one
flag, whenever we decide the baseline has served its purpose.

---

## A note on the metrics

Genre agreement is a **proxy, not ground truth**. Two tracks can be similar and
carry different labels, and FMA's eight genres are coarse. It is reliable for
comparing configurations against each other — which is what we need — but it is
not a measure of how good the recommender feels to a listener.

This is also why `genre_overlap@10` exists. Exact genre matching scores a
cross-genre recommendation as simply wrong, which would mark the planned
Familiar → Adventurous control as a regression the moment it works. Overlap
grades the distance instead, so we can tell smart exploration from random noise.

The four metrics are **frozen**. Swapping rulers halfway would invalidate every
comparison behind them.

---

## Development notes

### Merge conflicts in notebooks

Notebooks are stored as JSON, so conflicts are ugly. This project uses
**nbdime** (already in the `dev` dependency group).

Enable it once:

```bash
nbdime config-git --enable
```

When a conflict occurs:

```bash
nbdime mergetool
```

A browser window opens showing both versions side by side. Select the correct
cells, save, then finish the merge:

```bash
git add <notebook>.ipynb
git commit -m "Resolve notebook merge conflict"
```
