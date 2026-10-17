# Round two — written before the returns arrive

**Committed 2026-09-26, before a single rating came back.** That is the whole
point of the file. A threshold written afterwards is a rationalisation; this one
is a commitment, and the git history is the proof of which it is.

## What round one could not answer

Round one's eighty pairs were generated from the old 518-column space, because
the rhythm and drum columns did not exist yet. The 125 ratings that came back
**chose** the space we ship and have never **tested** it. Nobody has heard what
this model recommends.

Round one settled two other things, and neither depends on which space we use:
the recommender beats a random pair to a human ear (2.96 vs 1.87, p = 0.002),
and rank 50 and 200 are heard as no less related than rank 1 (3.07, p = 0.65).
Those stand whatever round two says.

## What round two asks

On seeds round one never touched, does the shipped space's pick score higher
than the old space's pick, to a listener who cannot tell which is which?

Every seed carries six pairs: two from `heard`'s top five, two from `all`'s top
five, one from `heard` at rank 50 or 200, and one drawn at random. Drawing from
both spaces rather than only the one under test is deliberate — pairs taken from
a space sit close together in it by construction, which would rig the comparison.

## The threshold

> **If `heard`'s pairs do not score at least as high as `all`'s pairs on the
> same seeds, round two does not validate the change.**

Three readings, decided now:

- **`heard` scores higher, and the difference clears a paired test at p < 0.05** —
  the change is validated. Say so.
- **No detectable difference** — the most likely outcome given the sample size.
  The change is *not* validated. `heard` stays, on the grounds that it is 154
  columns instead of 518 and 57 components instead of 144, and we say plainly
  that its advantage to the ear is unconfirmed.
- **`all` scores higher** — the change is not supported. The +0.05 against
  −0.358 that drove it was probably an artefact of selecting on the same data.
  Report it, and reopen the question.

## The scale check, which comes first

The random pairs are not decoration. Different people on a different day use the
1–5 scale differently, so an absolute score from round two cannot be compared
with round one's until the floors are compared.

Round one's random pairs averaged **1.87**. If round two's land near that, the
two rounds share a ruler and the numbers can sit side by side. If they land far
from it, every comparison across rounds has to be made on the *gap* above the
floor rather than on the raw score, and this file is where that was decided
rather than discovered.

## What gets reported

**Whatever comes back.** A positive result earns a slide. A null or negative
result gets a sentence on the caveats slide that already exists. Round two is
mentioned either way, because a round that only gets described when it agrees
with us is not evidence, and anyone who asks "did you validate it?" deserves the
real answer.

## What this cannot settle

`heard` against `heard_wide` — the 84 non-mfcc timbre columns. Both correlations
are measured on the same pairs, and separating two dependent correlations needs
far more data than testing either against zero. That needs a forced choice, not
a rating, and it is not what this round is.

---

Built by `uv run python -m modeling.round2`. Returns go in
`data/listening_test/responses/` beside round one's, and
`uv run python -m modeling.listening --csv` merges both rounds.
