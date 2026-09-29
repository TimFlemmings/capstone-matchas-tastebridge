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

## Round two ran, and the space change did not survive it

The eighty pairs in round one were generated from the OLD 518-column space,
because the rhythm and ratio columns did not exist yet. So that test *chose*
`heard` and never *tested* it. Round two closed that hole: twelve fresh seeds,
one per genre, six pairs each drawn from both spaces plus depth and random, with
the failure threshold committed before any rating came back
(`data/listening_test/ROUND2_PREREGISTRATION.md`). Four of six raters are in.

**The headline replicated and got stronger.** Model picks 3.45 against a random
floor of 1.27, p = 0.00004 — round one's gap was 1.09, this is 2.18. Raters
agree with each other at 0.81 against 0.53.

**The space change did not.** Correlating distance against the human score on
round two's independent pairs gives `all` −0.46 and `heard` −0.43 — tied, and on
the raw pairs the old space is clearly ahead at −0.50 against −0.33. On the
pre-registered measure `heard` leads by 0.36 of a point at p = 0.43. **Not
validated.** `heard` stays because 154 columns and 57 dimensions beat 518 and
144 on simplicity, and we no longer claim it is better to a listener.

**And the finding that matters most to you.** Looking only at top-five pairs —
what a user actually sees — *neither* space correlates with what people heard:
−0.15 and +0.03. The correlation in every table comes from random pairs being
far and scored low. **The model separates related from unrelated; it does not
rank among the related.** If your EffNet embedding fixes that, it is a much
bigger claim than a genre-agreement number, and it is testable.

### So: round three, EffNet against `heard`

This is the thing worth doing together, and the machinery exists.

`modeling/round2.py` builds the packages — clips, volume matching, blind ids,
coin-flipped order, rater assignment, the rating form. Point it at a third space
and it produces the same thing for EffNet. What I need from you is a column
table for the catalogue, one row per track id, the 1,280 dimensions as columns,
registered in `modeling/features.py` as a named space. Then packages go out the
same afternoon and `modeling.listening --blind-spots` reads the returns.

Two things learned the hard way, so round three avoids them:

- **Draw depth pairs from every space under test, not just one.** Round two took
  them only from `heard`, which penalised `heard` in exactly the comparison it
  was meant to settle. My error.
- **Exclude same-artist pairs.** 13 of 72 in round two were the same artist
  twice — trivially similar, and they inflate whichever space drew them. Round
  one happened to have none.

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

| | round one (6 raters, 125) | round two (4 raters so far) |
|---|---|---|
| model pairs, human score | 2.96 / 5 | 3.45 / 5 |
| random pairs | 1.87 / 5 | 1.27 / 5 |
| the gap, which is the comparable part | 1.09, p = 0.002 | **2.18, p = 0.00004** |
| rank 50 and 200 | 3.07 — same as the top five, p = 0.65 | 2.00 clean — lower, n = 9 |
| rater agreement | r = 0.53 | r = 0.81 |

Compare the **gap**, not the raw scores: a different group on a different day
uses the 1–5 scale differently, which is exactly what the random pairs are there
to reveal. Round two's raters were harsher on both ends.

`2.96` is worth repeating honestly: on the rubric the raters were given, 3 means
*"a recognisable connection, with substantial differences"*. It works. It is not
yet good.

The round two deep figure is the one to watch. In round one, depth held up and
that was the dial's whole justification. In `heard`, on nine clean pairs, it did
not. Two raters are still out.

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
