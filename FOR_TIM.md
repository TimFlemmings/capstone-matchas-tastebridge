# The `heard` model — how to run it and what to evaluate

Tim, this answers your three questions in order. The short version:

1. **Is the `heard` model in GitHub?** Yes. It is the default — you do not have
   to pass anything to get it.
2. **Is it the latest baseline?** It is the shipped recommender. It replaced the
   518-column space.
3. **Does it still need evaluating?** Yes, and the part that is missing is
   exactly the part you offered to do. See *What is actually missing* below.

---

## Get it running

```bash
git checkout feature/pca-recommendation
uv sync
```

Two feature files are gitignored because they are large and rebuildable. Without
them nothing will run, and the error message will tell you this same thing:

```bash
uv run python -m modeling.rhythm       # ~90 seconds on 8 cores
uv run python -m modeling.separation   # ~20 minutes
```

They read the mp3s in `data/fma_small/` and write `data/rhythm_features.csv` and
`data/separation_features.csv`. You need the FMA small audio and
`data/fma_metadata/` in place first.

Then:

```bash
uv run python -m modeling.recommender --track-id 2
uv run python -m modeling.recommender --track-id 2 --adventure 1.0
uv run python -m modeling.recommender --compare     # slow, scores three spaces
```

---

## What the model is

154 columns, not 518:

| block | columns | where from |
|---|---:|---|
| `mfcc` | 140 | FMA's `features.csv` |
| rhythm — tempo, onset rate, onset strength, pulse clarity | 13 | `modeling/rhythm.py` |
| drum-to-instrument energy ratio | 1 | `modeling/separation.py` |

StandardScaler → PCA at 90% variance (57 components) → NearestNeighbors on
cosine. Nothing is trained and no genre label is ever seen.

In code it is `modeling.features.FEATURE_SETS["heard"]`. `--space all` gives you
the old 518-column space, and `--space heard_wide` the 238-column variant we
rejected. `full_table()` in `modeling/features.py` is what joins FMA's table to
the two files we extracted.

---

## Why it changed — read this before evaluating

**The genre scorecard is not the argument.** It went up (`precision@10` 0.400 →
0.422, `nDCG@10` 0.420 → 0.441), but 294 of the columns removed were the chroma
families that had been dragging genre agreement down, so that rise was close to
mechanical.

The reason is the listening test. Correlating each space's distances against the
125 human ratings:

| space | columns | matches the ear | p |
|---|---:|---:|---:|
| all 518 | 518 | +0.05 | 0.58 — nothing |
| **`heard`** | **154** | **−0.358** | **0.0000** |
| `heard_wide` | 238 | −0.213 | 0.017 |

Negative is the good direction: a short distance should come with a high score.

One cost worth knowing about: `coverage` fell 0.780 → 0.757, so slightly fewer
tracks in the library ever get recommended.

---

## What is actually missing — where you can help most

**The eighty pairs in the listening test were generated from the OLD 518-column
space.** The new columns did not exist when the test was built. So the test
*chose* `heard`; it has not *tested* it. Nobody has heard what this model
recommends.

That is the single biggest hole in the project, and it is the thing worth
evaluating.

The design that would close it, if you want to run it:

- **Forced choice, not rating.** Show a seed plus one recommendation from `all`
  and one from `heard`, ask which is closer. Counting wins is a binomial test,
  which needs far less data than comparing two correlations. Roughly 85
  comparisons detects a 65/35 split; ~47 detects 70/30.
- **Fresh seeds**, not the ten already used.
- **Coin-flip which side is A**, and keep it blind — no artist, title or genre.
- **Write the failure threshold down before running it.** Something like "if
  `heard` wins fewer than 55% of at least 100 comparisons, we do not consider
  the change validated", committed with a date.

Expect `heard` to win, but expect a smaller margin than the table above shows —
the same data chose it, and a selected effect is always larger than the real
one.

`notebooks/02_listening_test.ipynb` builds pairs, clips, volume matching and
rater assignments. Most of it is reusable.

---

## About your YouTube and Spotify test

Two things that will bite you:

**Volume.** FMA is thousands of independent uploads and the loudest clip in our
pool carried twenty times the energy of the quietest. Two clips at the same
volume are heard as more alike whatever the music is doing. `features.csv`
carries each track's RMS energy and the listening-test notebook uses it to play
every clip at a corrected level. YouTube and Spotify are loudness-normalised
and FMA is not, so anything you pull in needs matching before a human hears it.

**The features have to be extracted the same way.** A new track needs FMA's own
`features.csv` pipeline for its mfcc columns, plus `modeling/rhythm.py` and
`modeling/separation.py` for the other 14. `extract()` in each module takes a
single file, so that part is straightforward — the mfcc side is the work.

One detail that matters if you re-extract: `hpss_ratio` is stored **logged**
(`np.log(percussive / harmonic)`). A ratio is multiplicative, and the raw form
scored eta² 0.038 against 0.143 logged. `modeling/separation.py` takes the log
at extraction so nothing downstream has to remember to.

---

## Where to read what

| file | what it holds |
|---|---|
| `README.md` | the whole story, including everything we rejected |
| `notebooks/03_listening_test_results.ipynb` | the test results and how each candidate was judged |
| `notebooks/04_new_features.ipynb` | the new columns one at a time |
| `data/listening_test/all_ratings.csv` | all 125 ratings with both tracks named |
| `data/listening_test/responses/` | the six raters' original returns |

`uv run python -m modeling.listening` prints what listeners said connected the
pairs, against which feature families could carry it. `--csv` rebuilds
`all_ratings.csv`.

---

## One thing to settle between us

Your baseline excludes same-artist neighbours and mine does not. **That alone is
the entire gap between our numbers** — 0.343 against 0.414 on identical data and
identical features. Neither is wrong, but we should not report both without
saying which convention each uses. Adding the exclusion would move every number
in the README by about 0.07, so it is a decision about the frozen metrics rather
than a bug fix. Worth ten minutes before Tuesday.
