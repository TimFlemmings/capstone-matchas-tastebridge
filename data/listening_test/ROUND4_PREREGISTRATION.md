# Round four — written before the pairs are drawn

**Committed 2026-10-05, before a single seed was drawn, a pair generated or a
rating returned.** As in rounds two and three, the git history is the proof that
the questions and thresholds came first.

## Why round three was set aside

Round three (built 2026-10-01) compares EffNet with `heard` on fma_small, using
plain rank windows. The app that ships no longer looks like that: it runs on
fma_medium (24,757 tracks), its tiers are Tim's tuned bands with genre caps and
connection gates, and `heard` features exist only for fma_small, so `heard`
cannot be compared on the shipped catalogue at all. Its answer key is also on
GitHub. It stays in the repository as built and is never sent.

The question "does EffNet's distance match the ear better than `heard`'s?" is
answered separately, retrospectively, from rounds one and two
(`reports/effnet_vs_heard/`). Round four asks the question that analysis cannot:

> **Do the tracks the live app actually recommends sound related to the seed,
> at each of its three tiers?**

## The design

- **Seeds:** 6, drawn from tracks that are in fma_medium but not fma_small, so
  no seed or pick has been heard in any earlier round. One seed per top-level
  genre, one per artist, artists not used in rounds one to three. Fixed random
  state, recorded in the build script.
- **Source of the pairs:** the live app's own `POST /recommendations`
  (https://mrln77-tasty-demo.hf.space), called with `adventure` 0.1 (Home /
  Familiar), 0.5 (Bridge) and 0.9 (Adventure). This is exactly what a user gets.
- **Pairs per seed (7):**

| pairs | source |
|---|---|
| 2 | Home — drawn at random from the first five the app returns |
| 2 | Bridge — same rule |
| 2 | Adventure — same rule |
| 1 | random track from fma_medium, different artist — the floor |

  42 pairs in all. Same-artist pairs are excluded from every source.
- **Everything else follows the earlier rounds:** blind pair ids (`T01`…), a
  coin flip for which clip plays first, volume matched from RMS energy, the same
  1–5 form, a shared core every rater hears. Raters are `R1d`…`R6d`.
- **Who rates:** not Tim — he can see round three's key and helped tune the
  tiers. Amin's own ratings are a pilot and are never counted.
- **The answer key stays off GitHub** (`data/listening_test/round4/`, which is
  gitignored) until the analysis is done.
- **Deadline:** whatever has arrived by **Wednesday 7 October 2026, 12:00
  Berlin** is analysed. Later returns are reported separately, not merged in.

## The questions and thresholds

Each pair's score is the mean over the raters who heard it.

**Q1 — Does each tier beat random?** For each tier, its 12 pairs against this
round's 6 random pairs, one-sided Mann–Whitney U, Holm-corrected across the
three tiers. Because 6 random pairs is a thin floor, a secondary check compares
each tier's pairs with the pooled random floor of rounds one and two (mean
1.57) by a one-sided one-sample Wilcoxon test. The primary test decides.

- p < 0.05 after correction → that tier is heard as related. Say so.
- otherwise → that tier is *not shown* to beat random. Say that, not that it
  fails.

**Q2 — Does relatedness fall in the order the dial promises
(Home ≥ Bridge ≥ Adventure)?** Spearman between tier (1, 2, 3) and pair score,
one-sided (negative expected). Per-seed means are reported beside it.

- ρ < 0 with p < 0.05 → the dial is heard as a dial.
- otherwise → the ordering is not detectable at this sample size.

**Q3 — Is Adventure still related, or has it crossed the line?** Adventure
counts as "still related" if it passes Q1 *and* its mean score is at least 2.0
("weak: one or two minor characteristics connect them" on the raters' scale).
If it beats random but averages below 2.0, it is "barely related"; if it does
not beat random, "not shown to be related".

## What gets reported

Whatever comes back, including a null. A tier that does not beat random is a
finding about the tiers, not a reason to rerun the round until it does.
