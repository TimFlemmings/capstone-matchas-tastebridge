# Matcha — a music recommender you can steer

Most recommenders keep you where you already are. Matcha lets you decide how far
from your taste you want to travel, and shows you *why* an unfamiliar track
still connects to where you started.

This repository holds the recommendation engine: the feature pipeline, the
model, and the evaluation setup that every later idea is measured against.

Six people rated eighty pairs of tracks blind, and their answers changed what
the model runs on. It beats a random pair to a human ear (p = 0.002), the far
end of the dial is heard as related as the near end (p = 0.60), and two of the
features it now uses did not exist in the dataset — listeners named them, so we
read them off the audio. [What the listeners changed](#what-the-listeners-changed)
is the short version.

---

## Where the project stands

**Milestone 1 — project setup: done.** The baseline works end to end: give it one
track, get a ranked list of similar ones.

```bash
uv run python -m modeling.recommender --track-id 2
```

```
seed track : AWOL — Food [Hip-Hop]
space      : 57 components, cosine distance

          rank  distance                   artist                      title    genre
108495       1    0.3152  Sun Sunych ft. DanBrown        Toys (ft. DanBrown)  Hip-Hop
22094        2    0.3800                    Gates            I Remember When  Hip-Hop
142092       3    0.3822                       SK    woke up feeling blessed  Hip-Hop
83717        5    0.3876                   Cullah                 Ninety Six  Hip-Hop
...
```

### Why the space changed

The reason is not on a genre scorecard. It is this:

| space | matches what listeners said |
|---|---:|
| all 518 FMA features | +0.05 — no relation at all |
| **heard, 154 columns** | **−0.35** |

Distance in each space, correlated against 124 blind human ratings. Negative is
the good direction: closer in that space, rated more alike. The old space has no
relationship to what people hear; the new one has a strong one, and that
ordering held five times as ratings arrived, at 42, 63, 84, 103 and 124.

That is the argument. Everything below is a check that the change did not cost
anything, not a reason for making it.

### Baseline scorecard

Measured on held-out tracks with 5-fold cross-validation — the pipeline is
refitted per fold, so these numbers say how it behaves on music it has never
seen.

| Metric | all 518 | **heard, 154** | What it asks |
|---|---|---|---|
| `precision@10` | 0.400 ± 0.006 | **0.422** ± 0.002 | how many of the ten share the seed's genre |
| `nDCG@10` | 0.420 ± 0.006 | **0.441** ± 0.002 | are the good ones near the *top* |
| `genre_overlap@10` | 0.262 ± 0.006 | **0.280** ± 0.003 | how much the full genre lists overlap |
| `coverage` | 0.780 ± 0.004 | 0.757 ± 0.005 | how much of the catalogue ever gets surfaced |
| components | 144 | **57** | how wide the search space is |
| random guess | 0.125 | 0.125 | eight balanced genres, so 1-in-8 |

Three of the four rose and one fell, on a third of the dimensions. **Read that
as "nothing broke", not as the result.**

All four are built on genre, and two things make them weak evidence here. The
rise is close to mechanical: 294 of the removed columns were the ones dragging
genre agreement down, so dropping them had to lift it. And a higher genre score
is not obviously a better recommender — a system that returned nothing but the
seed's own genre would score beautifully and be exactly the bubble this project
exists to get out of.

`coverage` is the row that fell. It counts how much of the catalogue is ever
surfaced across every seed, and a tighter space surfaces a slightly narrower
slice.

The new space is smaller than the old one and contains columns the old one never
had. Both halves of that came out of the listening test — see
[what the listeners changed](#what-the-listeners-changed).

**Do not take the comparison on trust.** Every space is still here, still
fitted from the raw data, and one command scores all of them side by side:

```bash
uv run python -m modeling.recommender --compare
```

```
     space  columns  components  precision@k  genre_overlap  novelty 0->1  precision 0->1
       all      518         146        0.414          0.270          +67%            -47%
heard_wide      238          81        0.469          0.325          +79%            -51%
     heard      154          57        0.433          0.287          +86%            -51%
```

Those are in-sample over a seed sample, so they sit a little below the held-out
table above and are not the same measurement — what they are good for is the
comparison between the rows. Note that this output does **not** agree with the
choice this repository made: on genre agreement `heard_wide` wins. The reason we
ship `heard` anyway is the listening test, and
[what is still open](#what-is-still-open) says how we would settle it properly.

**Milestone 2 — the Familiar → Adventurous control: working.** One dial, and no
second model behind it.

```bash
uv run python -m modeling.recommender --track-id 2 --adventure 0.7
```

**Milestone 3 — measured against human ears, not genre labels: done.** The
recommender beats a random pair to a listener (p = 0.002), and the far end of
the dial is heard as related as the near end (p = 0.60).

One number from that worth sitting with: listeners rated the model's own top
five at **2.96 out of 5**, which on the rubric they were given is "a
recognisable connection, with substantial differences". Clearly above the
random floor of 1.87, and clearly not "strongly related". The recommender
works; it is not yet good.

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
| 0.00 | 1 | 0.338 | 0.429 | 0.292 | 0.411 | 0.590 |
| 0.22 | 4 | 0.361 | 0.393 | 0.252 | 0.436 | 0.585 |
| 0.50 | 22 | 0.414 | 0.333 | 0.198 | 0.496 | 0.598 |
| 0.75 | 106 | 0.494 | 0.290 | 0.164 | 0.591 | 0.629 |
| 1.00 | 500 | 0.621 | 0.232 | 0.125 | 0.735 | 0.674 |
| | | **+84%** | **−46%** | **−57%** | **+79%** | **+14%** |

```bash
uv run python -m modeling.recommender --depth-profile   # reproduces this table
```

The last row is the point: the two columns that *should* fall do fall, and the
three that should rise do rise. A dial that moved neither would be decoration.

The same table on the old 518-column space, for comparison — the shape survived
the change, and two of the five ends improved:

| `adventure` | `novelty` | `precision@10` | `genre_overlap@10` | `intra_list_div` | `coverage` |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.412 | 0.406 | 0.269 | 0.508 | 0.603 |
| 1.00 | 0.692 | 0.214 | 0.113 | 0.808 | 0.683 |
| | **+68%** | **−47%** | **−58%** | **+59%** | **+13%** |

`novelty` and `intra_list_diversity` now travel further across the dial — +84%
and +79%, against +68% and +59% — because the new space starts tighter. At
`adventure = 0` its ten recommendations sit closer to the seed and to each
other, which is what "familiar" should look like, and it has further to go.

Three things this settles:

**Relevance decays smoothly — there is no cliff.** So a continuous slider is
defensible; we do not have to hide the far end behind a safety cap.

**Even at full adventure, `precision@10` is 0.232 — still 1.9× random.** Rank
500 of 8,000 is the top 6% by similarity. This is *steering*, not shuffling: far
away, but chosen. A random track is noise, and listeners learn to distrust
noise.

Listeners confirmed that directly, which is the part no metric could settle:
pairs drawn from rank 50 and 200 were rated **as related as the model's own top
five** (p = 0.60), and both beat a random pair (p = 0.003 and p = 0.002).

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

## What the listeners changed

Every number above this line is built on genre labels, and genre is a proxy for
the thing we actually care about. So we asked people.

Eighty pairs of thirty-second clips, six raters, blind: a pair number and two
clips, no artist, no title, no genre, no cosine score, volumes matched from
`features.csv` so loudness could not read as similarity. 124 scored ratings came
back, covering every pair. Built in `02_listening_test.ipynb`, read back in
`03_listening_test_results.ipynb`.

Three kinds of pair, because a score with nothing to read it against says
nothing:

| kind | what it is | why |
|---|---|---|
| `model` | the seed and one of its top five | the thing being tested |
| `random` | the seed and a track drawn at random | the floor |
| `deep` | the seed and its rank 50 or rank 200 | the far end of the dial |

### It works, to a person

| kind | ratings | mean |
|---|---:|---:|
| `deep` | 29 | 3.03 |
| `model` | 80 | 2.96 |
| `random` | 15 | 1.87 |

`model` beats `random` at **p = 0.002**, `deep` beats it at **p = 0.003**, and
`model` and `deep` are indistinguishable at **p = 0.60**.

That last one is the result for the dial. Rank 200 is heard as related as rank
1, and both are heard as more related than chance. Far away, but chosen.

Raters agree with each other at r = 0.40 — some pairs split four points of five —
so the average of six is reliable at 0.80, which caps any model near 0.90. The
shipped space reaches −0.77 against that average on the nine shared pairs.

### Most of the 518 columns have nothing to do with what people hear

Distance in each space, correlated against the human score. Negative is the good
direction: closer in that space, rated more alike.

| space | columns | genre agreement | matches the ear |
|---|---:|---:|---:|
| all 518 | 518 | 0.414 | +0.047 |
| harmony | 294 | 0.278 | +0.040 |
| timbre | 224 | 0.458 | −0.179 |
| **mfcc** | 140 | 0.412 | **−0.314** |

`all 518` and `mfcc` are within 0.002 of each other on genre agreement, and a
listener separates them immediately. The full space carries 294 harmony columns
that pull the wrong way and drown the signal the 140 mfcc columns carry. No
amount of tuning against the frozen metrics would have found that; they cannot
see it.

### The second question is the one that turned into work

The form asks what connects a pair, from a fixed vocabulary. Those words, lined
up against the feature families that could carry them:

| word | mentions | mean score | families in FMA |
|---|---:|---:|---|
| `mood` | 50 | 3.42 | none — and deliberately not built |
| `energy` | 48 | 3.67 | rmse, zcr |
| `harmony` | 33 | 3.61 | chroma × 3, tonnetz |
| **`rhythm`** | 29 | 3.83 | **none** |
| **`instrumentation`** | 25 | 4.00 | **none** |
| `timbre` | 18 | 3.44 | mfcc, spectral × 4 |
| `nothing` | 19 | 1.11 | the control word, behaving |

```bash
uv run python -m modeling.listening    # rebuilds this table from responses/
```

FMA's eleven families are pitch, texture or loudness. Nothing measures time, and
nothing says what is producing the sound. So the model could not tell whether two
tracks share a beat, and the raters kept saying that is what connects them.

Neither is recoverable from `features.csv`, so both were read back off the mp3s.

### Two features, and what survived

| module | columns | kept | why |
|---|---:|---:|---|
| `rhythm.py` | 13 | 13 | tempo, note density, how hard it hits, how firmly it keeps the beat |
| `separation.py` | 81 | 1 | the drum-to-instrument energy ratio |

The other 80 were the timbre of the drums and the instruments described
separately, after splitting the audio. They produced **the best genre agreement
in the project, 0.488** — and left the supervised ceiling flat at 0.546, which is
the signature of information rearranged rather than added. They also made the
match with human ratings worse. Judged on agreement alone we would have kept
them.

Three other ideas were built and rejected the same way: a space made only of the
families the ear approved, key-invariant chroma (rotating each track's pitch
histogram so its strongest note sits first, to test whether absolute key was why
harmony fails — it was not), and the split-timbre block above.
`04_new_features.ipynb` walks the surviving columns one at a time.

---

## The average hides a wide spread — and the spread is about the labels

`precision@10` of 0.433 is a mean over 8,000 seeds, and a listener meets one
seed at a time. Per seed it runs the whole range:

| | per-seed `precision@10` |
|---|---:|
| 10th percentile | 0.0 |
| 25th | 0.2 |
| median | 0.4 |
| 75th | 0.7 |
| 90th | 0.9 |

**10.8% of tracks score zero** — the recommender finds no same-genre neighbour
at all for one track in nine. Another 6.1% score a perfect ten.

By the seed's own genre:

| genre | mean | |
|---|---:|---|
| International | 0.541 | 4.3× chance |
| Hip-Hop | 0.538 | |
| Folk | 0.505 | |
| Rock | 0.501 | |
| Instrumental | 0.492 | |
| Electronic | 0.403 | |
| Experimental | 0.290 | |
| **Pop** | **0.199** | 1.6× chance |

Pop and Experimental are the two genres that are not a sound. Pop is whatever
charts and Experimental is defined by not fitting elsewhere, so their tracks do
not resemble each other — and this model matches on sound. Read that as the
label failing rather than the model: it is the same proxy problem as everywhere
else in this file, showing up per genre instead of in aggregate.

### And the spread means less than it looks like

The obvious next move is to predict which seeds will score badly and warn about
them. There is even a signal to do it with: how crowded a track's corner of the
space is — the mean distance to its ten nearest neighbours, no label involved.
Against the genre metric it works well, correlating −0.36 across the catalogue:

| | mean distance | `precision@10` | scores zero |
|---|---:|---:|---:|
| densest fifth | 0.215 | 0.595 | 5.7% |
| loneliest fifth | 0.459 | 0.285 | 14.3% |

So we checked it against the listening test, and it does not survive.

Across the ten seeds the test used, crowding correlates **+0.11** with the
scores the model's own pairs received — the wrong sign, and nowhere near
significant (p = 0.24). The clearest case is the loneliest seed of the ten:

| seed | crowding | same-genre | rated by listeners |
|---|---:|---:|---:|
| Zlatne Uste Balkan Brass Band | 0.172 | 8 / 10 | 3.00 |
| Myo | 0.299 | — | **1.85** |
| Mutilation Rites | 0.369 | 8 / 10 | 3.50 |
| **Alec K. Redfearn & the Eyesores** | **0.444** | **1 / 10** | **3.50** |

Redfearn's accordion folk sits in the emptiest corner of the space and scores
worst on genre agreement of any seed in the test. Listeners gave its pairs the
joint highest score of the ten. Its neighbours are not in its genre and they do
sound like it.

So the per-seed spread is a fact about the *labels*, not a map of where the
recommender fails. `10.8% score zero` means those tracks have no same-genre
neighbour — which is what it says, and not what it would be tempting to read it
as.

Ten seeds is far too few to conclude that crowding is irrelevant to what a
listener hears; it is enough to say we have no evidence that it is relevant. So
the recommender prints it as what it is:

```
seed track : Alec K. Redfearn & the Eyesores — Queen Of The Wires [Folk]
crowding   : 0.444 — a thinly populated corner, mean distance to the nearest 10
```

A geometric fact about where the seed sits, one extra neighbour query, and no
claim about quality attached to it. Dressing it up as a confidence score would
be this project's own mistake repeated in miniature: trusting a genre-shaped
measurement about something genre has already been shown not to see.

A second use for the same number is open and untried: the dial's depth is a
fixed 500 ranks for every seed, and a thinly populated corner is already far
away at rank 50.

---

### What is still open

The listening test leaves two spaces standing, and the two rulers disagree about
them:

| | `heard` (shipped) | `heard_wide` |
|---|---:|---:|
| columns | 154 | 238 |
| genre agreement | 0.438 | **0.474** |
| supervised ceiling | 0.533 | **0.546** |
| matches the ear | **−0.351** | −0.217 |

They differ only in whether the 84 non-mfcc timbre columns come along, and
across 500 seeds they share 4.5 of their top 10 — so half of what a listener
would hear is different.

`heard` ships because the ear is the ruler this project exists to serve, and
because that ordering held five times as returns arrived, at 42 ratings, then
63, 84, 103 and 124. **But the data that chose it cannot also validate it.** A
second round, put as a forced choice between the two on fresh seeds, is the
honest way to settle it and has not been run.

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
├── features.py       the named spaces, scaling, PCA, artifact persistence
├── recommender.py    the model — nearest neighbours, and the adventure dial
├── evaluation.py     the four metrics, k-fold validation, the depth profile
├── interpretation.py reading a PCA space back in terms of the original columns
├── train_pca.py      CLI: fit a feature space and log the run
│
├── rhythm.py         13 columns FMA does not ship, read off the mp3s
├── separation.py     the harmonic-percussive split and what came out of it
└── listening.py      read the listening test back and name the feature gaps

notebooks/
├── 01_eda_baseline.ipynb            the analysis, step by step
├── 02_listening_test.ipynb          building the blind test
├── 03_listening_test_results.ipynb  reading it back, and what it decided
└── 04_new_features.ipynb            the columns we added, one at a time
```

The bottom three modules did not exist until listeners asked for them. They
write to `data/`, which is gitignored, so a fresh checkout rebuilds them:

```bash
uv run python -m modeling.rhythm       # ~90 seconds on eight cores
uv run python -m modeling.separation   # ~20 minutes; splitting audio is the slow part
```

The listening test's own paperwork *is* tracked — `data/listening_test/` holds
the 124 ratings, the answer key and the assignments. Everything else under
`data/` regenerates; six people rating eighty pairs does not.

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

### Pick a feature space

`--space` selects which columns the recommender runs on. `heard` is the default
and the one the listening test chose; the others are kept so any number in this
file can be reproduced.

```bash
uv run python -m modeling.recommender --track-id 2                    # heard, 154 columns
uv run python -m modeling.recommender --track-id 2 --space all        # the original 518
uv run python -m modeling.recommender --track-id 2 --space heard_wide # the runner-up
```

Each space has its own fitted `scaler`/`pca` pair in `models/`, because a scaler
fitted on 154 columns cannot be pointed at 518.

```bash
uv run python -m modeling.recommender --compare
```

scores all three from the raw data in one run, fitting each rather than loading
it, so nothing in this file has to be taken on trust. The old space was not
deleted when the default moved; every claim about the change can be re-derived.

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

### The proxy caveat was right, and we can now say how badly

Genre agreement is a **proxy, not ground truth**. Two tracks can be similar and
carry different labels, and FMA's eight genres are coarse. This file has said so
since the first milestone. The listening test turned it from a caveat into a
measurement:

> `all 518` scores 0.414 on genre agreement and `mfcc` scores 0.412 — a gap of
> 0.002. Against human ratings they score +0.047 and −0.314. **The proxy could
> not see a difference listeners heard immediately.**

So the honest reading of the frozen metrics is narrower than we used to write.
They are good at one thing:

| question | genre agreement |
|---|---|
| is this space broken? (0.13 is chance) | **yes, reliably** |
| did new columns add information, or only reweight it? | yes, via the supervised ceiling |
| is this space *better* than that one? | **no** |

The third row is the one this project got wrong for a while, and the one the
listening test corrects.

The supervised ceiling deserves its own caveat. It asks how much genre
information a set of columns holds by handing a classifier the labels and
letting it train on them, so a column that carries something real but
genre-irrelevant leaves it flat and looks worthless. Where that mattered, we
checked redundancy directly instead — predicting the new columns from the old
ones, no labels anywhere.

This is also why `genre_overlap@10` exists. Exact genre matching scores a
cross-genre recommendation as simply wrong, which would have marked the
Familiar → Adventurous control as a regression the moment it started working.
Overlap grades the distance instead, so we can tell smart exploration from
random noise — and the depth profile above is the first place that paid off.

The four metrics are **frozen**. Swapping rulers halfway would invalidate every
comparison behind them — which is why the listening test was added *alongside*
them rather than replacing them, and why both feature spaces are scored on all
four in this file.

### One ruler the two baselines do not share

Tim's parallel baseline excludes same-artist neighbours when scoring; this one
does not. That single difference accounts for the whole gap between his 0.343
and this file's 0.414 — same data, same features, same metric. Until the two
protocols match, the two baselines cannot be compared, and roughly 0.07 of the
apparent quality here is the model finding other tracks by the same artist.

Adding the exclusion would move every number in this file. That makes it a
decision about the frozen metrics rather than a bug fix, and it is still open.

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
