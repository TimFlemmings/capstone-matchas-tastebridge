> Working notes on the `heard` feature model (MFCC + rhythm + harmonic–percussive ratio), written for Tim during the project. Formerly `FOR_TIM.md`, which notebooks 06 and 09 still refer to by that name. `heard` was later superseded by Discogs-EffNet (notebook 08).

# The `heard` model — how to run it and what to evaluate

Tim, this answers your three questions in order.

### "Do you have the MFCC + rhythm + harmonic/percussive ratio (`heard`) model available for me?"

**Yes — but as code, not as a file you can load.** Nothing fitted is in the
repository. There is no `.pkl` to download, because a fitted scaler and PCA are
outputs, not source, and they are gitignored along with the feature tables they
were fitted on.

Everything needed to rebuild it is in GitHub, and it is three commands. They are
below, with what each one produces and how long it takes.

### "This is the latest baseline model, I guess?"

It is the shipped recommender. `heard` is the **default** — `--space` exists
only so you can still get the old one. `--space all` gives you the 518-column
space we used before the listening test, and `--space heard_wide` the
238-column variant we rejected.

### "It still needs evaluating, right?"

Yes, and the missing piece is exactly what you offered to do. Jump to
[What is actually missing](#what-is-actually-missing--where-you-can-help-most)
if you only read one section.

---

## Getting it to run

```bash
git checkout feature/pca-recommendation
uv sync
```

You need two things in place that are not in git, both large:

- `data/fma_small/` — the 8,000 mp3s
- `data/fma_metadata/` — FMA's own `features.csv`, `tracks.csv`, `genres.csv`

Then three commands, in this order. **None of them is optional** — each writes
something the next one needs.

### 1. Read rhythm off the audio — about 90 seconds

```bash
uv run python -m modeling.rhythm
```

Writes `data/rhythm_features.csv`, 13 columns per track. Uses every core. It
will report a handful of tracks it skipped — 6 files FMA ships truncated, and 1
that is silent. That is expected; a partly described track has no business in a
distance calculation, so all 7 are dropped everywhere.

### 2. Split drums from instruments — about 20 minutes

```bash
uv run python -m modeling.separation
```

Writes `data/separation_features.csv`. This is the slow one: it runs HPSS on
every track. Start it and go do something else. Use `--limit 50` first if you
want to check it works before committing twenty minutes.

### 3. Fit the scaler and PCA — a couple of minutes

```bash
uv run python -m modeling.train_pca --feature-set heard --n-components 0.90
```

Writes `models/scaler_90pct_heard.pkl` and `models/pca_90pct_heard.pkl`, and
logs the run to MLflow. Without this step the recommender has nothing to
transform with — and if you skip it, the error message tells you this exact
command, so you cannot get stuck.

### Then it runs

```bash
uv run python -m modeling.recommender --track-id 2
uv run python -m modeling.recommender --track-id 2 --adventure 1.0
```

Track 2 is *Food* by AWOL, Hip-Hop. The first command should give you ten
Hip-Hop tracks at distances around 0.32–0.40. The second sends the window to
about rank 500 of the same ranking, which is what the dial does.

```bash
uv run python -m modeling.recommender --compare
```

Scores all three feature spaces from raw data in one go. Slow — several minutes
— and it prints, honestly, that `heard_wide` beats what we shipped on genre
agreement.

---

## What the model actually is

154 columns instead of 518:

| block | columns | where it comes from |
|---|---:|---|
| `mfcc` | 140 | FMA's `features.csv` |
| tempo, onset rate, onset strength, pulse clarity | 13 | `modeling/rhythm.py` |
| drum-to-instrument energy ratio (`hpss_ratio`) | 1 | `modeling/separation.py` |

The pipeline is `StandardScaler` → `PCA(n_components=0.90)` → `NearestNeighbors`
on cosine distance. 154 columns come out as **57 components**, against 144 for
the old space.

Nothing is trained and **no genre label is ever seen** by the recommender. A
RandomForest does appear in the project, but only as a measuring instrument —
see *How a feature earned its place* below.

Where things live in code:

- `modeling/features.py` — `FEATURE_SETS` defines every space; `full_table()`
  joins FMA's table to the two files you built in steps 1 and 2; `space(name)`
  resolves a name to real columns.
- `modeling/recommender.py` — `TrackRecommender`, the dial (`depth_for`), and
  the CLI.
- `modeling/rhythm.py`, `modeling/separation.py` — extraction. Both have an
  `extract()` that takes one file, which is what you want for outside audio.

### The dial, in one paragraph

It is not a second model. For a seed track the recommender has already ranked
all 7,999 others by distance. `depth_for(a) = round(500 ** a) - 1` slides a
ten-wide window down that existing ranking — `0.0` starts at rank 1, `1.0` at
about rank 500. Geometric, so the familiar end has fine control and the
adventurous end moves in big jumps.

---

## Why it changed — please read this before evaluating

**The genre scorecard is not the argument, and I would rather you heard that
from me than found it yourself.**

Genre numbers did improve — `precision@10` 0.400 → 0.422, `nDCG@10` 0.420 →
0.441, `genre_overlap@10` 0.262 → 0.280. But 294 of the columns removed were the
chroma families that had been dragging genre agreement down, so that rise was
close to mechanical. Read it as "nothing broke", not as the result.

The actual reason is the listening test. Take each of the 80 pairs, compute the
model's distance between the two tracks in a candidate space, and correlate
those 80 distances against what people scored them. Negative is the good
direction — a short distance should come with a high score:

| space | columns | matches the ear | p |
|---|---:|---:|---:|
| all 518 | 518 | +0.05 | 0.58 — no relationship at all |
| **`heard` (shipped)** | **154** | **−0.358** | **0.0000** |
| `heard_wide` | 238 | −0.213 | 0.017 — not significant |

That ordering held all five times we recomputed it as raters returned, at 42,
63, 84, 103 and 125 ratings.

**One cost, stated plainly:** `coverage` fell 0.780 → 0.757. Slightly fewer
tracks in the library ever get recommended to anyone.

### How a feature earned its place

Three checks, in this order, and the order is the point.

1. **Is it new?** Two ways. Label-free: regress the old columns onto the new
   one — if R² is high it is a copy with a new name. And supervised: hand a
   RandomForest the genre labels and cross-validate. Whatever accuracy it
   reaches is roughly the most those columns hold, so if adding columns lifts
   it, real information arrived. We call that the ceiling. It matters because
   our own distance score can rise for a fake reason — distance sums over every
   column, so *removing* bad columns raises it without a single new fact
   entering. A RandomForest does not sum, it picks, so it is immune to that.
2. **Does a listener hear it?** The correlation in the table above. **This is
   the one that decides.**
3. **Did anything break?** The genre metrics, as a sanity check. Never to
   choose.

We had this order wrong at first and let check 3 decide. Every time we did, it
gave the wrong answer.

The clearest case: we built 80 columns describing the timbre of the drums and
the instruments separately. They produced **the best genre agreement in the
whole project, 0.488**. They also nudged the human correlation up, −0.358 →
−0.374. We dropped them anyway, because the ceiling did not move — 0.533 →
0.534. Eighty more columns of mfcc, computed on two halves of a signal whose
unsplit mfcc we already had. 0.016 of correlation is not worth that, and it
comes with nothing new behind it.

**Four of the six candidates we rejected had raised the genre score.** Judged on
genre we would have kept nearly all of them.

The full ledger is in `notebooks/03_listening_test_results.ipynb`, and
`notebooks/04_new_features.ipynb` walks the surviving columns one at a time.

---

## What is actually missing — where you can help most

**The eighty pairs in the listening test were generated from the OLD 518-column
space.** The rhythm and ratio columns did not exist when the test was built. So
the test *chose* `heard`; it has not *tested* it. Nobody has yet heard what this
model recommends.

That is the single biggest hole in the project.

If you want to close it, the design matters, because the obvious version is
statistically hopeless. Comparing two correlations measured on the same pairs
needs enormous amounts of data. Comparing two models head to head does not:

- **Forced choice, not rating.** Show a seed plus one recommendation from `all`
  and one from `heard`. Ask which is closer. Counting wins is a binomial test.
  Roughly **85 comparisons** detects a 65/35 split; **47** detects 70/30. Six
  raters at 20 each is 120 — the same effort as round one, and this time it
  gives a clear answer.
- **Fresh seeds**, not the ten already used. Reusing them reintroduces the same
  problem.
- **Coin-flip which side is A**, or raters learn to always click one side.
- **Blind** — no artist, title, genre or score, same as round one.
- **Write the failure threshold down before running it.** Something like: *"if
  `heard` wins fewer than 55% of at least 100 comparisons, we do not consider
  the change validated."* Commit it with a date. Without that, every possible
  result gets explained away afterwards, and we both know it.

Expect `heard` to win — the old space correlated zero with the ear. But expect a
**smaller margin than the table above shows**, because the same data chose it,
and a selected effect is always larger than the real one.

`notebooks/02_listening_test.ipynb` already builds pairs, cuts clips, matches
volume and assigns raters. Most of it is reusable as is.

---

## About your YouTube and Spotify test

Good idea, and two things will bite you.

**Loudness.** FMA is thousands of independent uploads with no shared mastering —
the loudest clip in our pool carried **twenty times** the RMS energy of the
quietest. Two clips at the same volume are heard as more alike whatever the
music is doing, so this leaks straight into a similarity rating. YouTube and
Spotify are loudness-normalised and FMA is not, so anything you pull in has to
be level-matched before a human hears it. `features.csv` carries each track's
RMS energy and `notebooks/02_listening_test.ipynb` shows how we used it (the
reference is the 5th percentile rather than the true minimum, because one
near-silent clip would otherwise set the level for everything).

**The features have to be extracted the same way, or the distances are
meaningless.** A new track needs:

- its 140 mfcc columns through FMA's own `features.csv` pipeline (this is the
  fiddly part — same window size, hop length and summary statistics, or the
  numbers are not comparable);
- its 13 rhythm columns from `modeling.rhythm.extract()`;
- its `hpss_ratio` from `modeling.separation.extract()`.

Then transform with the **same fitted scaler and PCA** — do not refit on the new
track, or you have changed the space.

**One detail that will silently ruin your numbers if you miss it:**
`hpss_ratio` is stored **logged** — `np.log(percussive / harmonic)`. A ratio is
multiplicative, so four times more percussive and a quarter as much should sit
equally far from the middle, but raw they are 4 and 0.25. On a 320-track sample
the log took its separating power from eta² 0.038 to 0.143. `separation.py`
takes the log at extraction so nothing downstream has to remember, which also
means an externally computed raw ratio will not match.

---

## Where to read what

| file | what it holds |
|---|---|
| `README.md` | the whole story, including everything we rejected and why |
| `notebooks/03_listening_test_results.ipynb` | the test results, and how each candidate was judged |
| `notebooks/04_new_features.ipynb` | the new columns one at a time, with per-genre tables |
| `notebooks/02_listening_test.ipynb` | how the test was built — pairs, clips, volume, assignments |
| `data/listening_test/all_ratings.csv` | all 125 ratings, both tracks named. Start here if you would rather read data than prose |
| `data/listening_test/responses/` | the six raters' original returns, untouched |

```bash
uv run python -m modeling.listening        # what listeners said, against the features that could carry it
uv run python -m modeling.listening --csv  # rebuilds all_ratings.csv
```

### Numbers worth having in your head

| | |
|---|---|
| model pairs, human score | 2.96 / 5 |
| rank 50 and 200 | 3.07 / 5 — indistinguishable from the top five, p = 0.65 |
| random pairs | 1.87 / 5 — the model beats it at p = 0.002 |
| rater agreement | r = 0.53 individually; the six-rater average is reliable at 0.87 |
| ceiling on any model | ~0.93, because listeners disagree with each other |

`2.96` is worth repeating honestly: on the rubric the raters were given, 3 means
*"a recognisable connection, with substantial differences"*. It works. It is not
yet good.

---

## One thing to settle between us

Your baseline excludes same-artist neighbours and mine does not. **That alone is
the entire gap between our numbers** — 0.343 against 0.414, on identical data
with identical features.

Neither convention is wrong, but we should not put both in the same presentation
without saying which is which. Adding the exclusion moves every number in the
README by about 0.07, so it is a decision about the frozen metrics rather than a
bug fix — and whichever we pick, it has to be the same one in both notebooks.

Ten minutes before Tuesday.
