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

**Milestone 2 — the Familiar → Adventurous control: working.** One dial, and no
second model behind it.

```bash
uv run python -m modeling.recommender --track-id 2 --adventure 0.7
```

---

## The dial

The baseline answers with the ten nearest tracks — but it ranks *all 7,999*
others. We simply never look past the first ten.

```
nearest ──────────────────────────────────────────▶ furthest
[1][2][3][4][5][6][7][8][9][10] [11] ... [500] ... [7999]
└──── what the baseline shows ──┘
```

So the control slides a window of ten down a ranking that already exists.
`adventure` runs 0 → 1 and maps **geometrically** onto the starting rank,
because distance grows slowly with rank: ranks 1-5 sit at cosine 0.42-0.48,
ranks 31-35 at 0.55, ranks 201-205 at 0.66. A linear dial would spend most of
its travel inside a band nobody can hear apart.

| `adventure` | rank | `novelty` | `precision@10` | `genre_overlap@10` | `intra_list_div` | `coverage` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1 | 0.412 | 0.406 | 0.269 | 0.508 | 0.603 |
| 0.22 | 5 | 0.438 | 0.374 | 0.233 | 0.533 | 0.601 |
| 0.50 | 22 | 0.494 | 0.316 | 0.184 | 0.598 | 0.610 |
| 0.75 | 106 | 0.575 | 0.270 | 0.151 | 0.688 | 0.643 |
| 1.00 | 500 | 0.692 | 0.214 | 0.113 | 0.808 | 0.683 |
| | | **+68%** | **−47%** | **−58%** | **+59%** | **+13%** |

```bash
uv run python -m modeling.recommender --depth-profile   # reproduces this table
```

The last row is the point: the two columns that *should* fall do fall, and the
three that should rise do rise. A dial that moved neither would be decoration.

Three things this settles:

**Relevance decays smoothly — there is no cliff.** So a continuous slider is
defensible; we do not have to hide the far end behind a safety cap.

**Even at full adventure, `precision@10` is 0.214 — still 1.7× random.** Rank
500 of 8,000 is the top 6% by similarity. This is *steering*, not shuffling: far
away, but chosen. A random track is noise, and listeners learn to distrust
noise.

**`intra_list_diversity` moves furthest of the three.** The ten tracks become
0.508 → 0.808 apart *from each other*, which is what a listener actually
experiences as variety. `coverage` barely notices (+13%) because it is a
system-level count — see the note on metrics below.

### Why the dial stops at rank 500

Because past it the feature stops being a recommender. Pushing the profile out
across the whole catalogue:

| starting rank | share of catalogue | mean distance | `precision@10` | vs random | `coverage` |
|---:|---:|---:|---:|---:|---:|
| 500 | 6% | 0.699 | 0.217 | **1.73×** | 0.386 |
| 1,000 | 13% | 0.770 | 0.194 | 1.55× | 0.389 |
| 2,000 | 25% | 0.865 | 0.153 | 1.22× | 0.393 |
| 4,000 | 50% | 1.005 | 0.103 | **0.82×** | 0.384 |
| 7,900 | 99% | 1.369 | 0.063 | 0.50× | 0.353 |

Around rank 3,000 the recommender falls *below* chance — not neutral but
actively wrong, and a plain `random.choice()` would beat it. The geometry says
why: mean cosine distance crosses **1.0** there, the value at which two vectors
are orthogonal, and keeps climbing into negative correlation. The far half of
the catalogue is not distant music; it is opposite music.

The other half of the argument is that going deeper buys nothing. Coverage — the
only reason to reach past the top ranks at all — flattens after 500, gaining
0.009 across the remaining 7,500 ranks. So 500 is where relevance is still
1.73× random and the coverage curve has already levelled off. Rank 1,000 is
defensible if we want a bolder far end; 2,000 is not.

A falling `precision@10` here is the feature *working*, not a regression — which
is exactly why `genre_overlap@10` was frozen into the scorecard before this
existed. It grades how far a cross-genre jump landed instead of marking it
simply wrong.

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
├── recommender.py    the model — nearest neighbours, and the adventure dial
├── evaluation.py     the four metrics, k-fold validation, the depth profile
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
uv run python -m modeling.recommender --adventure 0.7       # ten further-out ones
uv run python -m modeling.recommender --depth-profile       # what adventure costs
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

These are **experiments, not the current pipeline.** Each was run once and
logged in MLflow; none of them has been applied. The baseline below is what the
code actually does today, and it stays that way — it is the reference point the
final Baseline vs TasteBridge comparison is measured against.

| Experiment (not applied) | Features | Components | precision@10 |
|---|---:|---:|---:|
| Without the two redundant chroma families | 350 | 119 | **0.429** |
| Without chroma + the kurtosis/skew columns | 250 | 78 | 0.423 |
| **← the pipeline as it runs today** | **518** | **146** | **0.413** |
| Fixed 60 components | 518 | 60 | 0.404 |
| RobustScaler instead of StandardScaler | 518 | 60 | 0.336 |

Two things to take from this. Dropping two chroma families would help *and*
shrink the space, so it is the strongest lead we have. And `RobustScaler` —
the obvious answer to those heavy tails — made things clearly worse, so the
outliers are carrying real signal.

Whenever we decide the baseline has served its purpose, applying any of these
is one flag:

```bash
uv run python -m modeling.train_pca --n-components 60 --drop-chroma
```

---

## A note on the metrics

### Which ruler measures what

Three of the four scorecard metrics are built on genre, and two of those are
built on the *same* question:

| Metric | Built on | Judges | Moves with adventure |
|---|---|---|---|
| `precision@10` | genre, hard (`genre_top`, right/wrong) | familiarity | ↓ −47% |
| `nDCG@10` | genre, hard + position | familiarity, ordering | ↓ |
| `genre_overlap@10` | genre, **graded** (`genres_all`, Jaccard) | how far a jump landed | ↓ −58% |
| `coverage` | no genre at all | catalogue breadth, **system-wide** | ↑ +13% |

`precision` and `nDCG` ask one question with two eyes: *is it the same genre?*
Only `nDCG` also cares whether the right answers sit near the top.

`genre_overlap` is the only genre metric that is fair to an adventurous
recommendation, because it scores the distance rather than marking a cross-genre
jump simply wrong.

### The hole these left, and what fills it

None of the four notices when **one listener's ten tracks are near-copies of
each other**. `coverage` cannot see it: it counts what the whole catalogue
surfaces across every seed, so a recommender can reach 68% of the catalogue
while handing each individual listener ten interchangeable tracks.

For a feature named *adventurous*, that is the wrong blind spot to keep. Two
additions, neither of which touches genre:

- **`intra_list_diversity`** — mean distance between the ten recommendations
  *themselves*. The per-listener version of coverage, and the metric that moves
  most across the dial (0.508 → 0.808).
- **`novelty`** — mean distance from the seed. The plainest reading of "how far
  did we travel". Trivially gamed on its own (recommend the furthest tracks in
  the catalogue and it maxes out), so it is only meaningful read against
  `precision`, which that strategy destroys.

The original four stay **frozen** — every comparison in this README is built on
them. These two are added alongside for the adventure control, not swapped in.

### And the proxy caveat still holds

Genre agreement is a **proxy, not ground truth**. Two tracks can be similar and
carry different labels, and FMA's eight genres are coarse. It is reliable for
comparing configurations against each other — which is what we need — but it is
not a measure of how good the recommender feels to a listener.

This is also why `genre_overlap@10` exists. Exact genre matching scores a
cross-genre recommendation as simply wrong, which would have marked the
Familiar → Adventurous control as a regression the moment it started working.
Overlap grades the distance instead, so we can tell smart exploration from
random noise — and the depth profile above is the first place that paid off.

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
