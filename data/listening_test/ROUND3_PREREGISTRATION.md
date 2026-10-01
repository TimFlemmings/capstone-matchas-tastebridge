# Round three — written before the returns arrive

**Committed 2026-10-01, before a single pair was generated or rated.** As with
round two, the git history is the proof that the threshold came first.

## What round three asks

Tim replaced hand-built features with Discogs-EffNet, a pretrained network.
Every piece of evidence for that change is offline: genre agreement@10 (0.49 →
0.77 on Hip-Hop) and where Spotify's and YouTube's picks for one seed land in
our ranking (8.67% → 0.17%). Neither is a human ear. Rounds one and two showed
that genre agreement is the weakest evidence this project has — EffNet was
trained on Discogs style labels, so reproducing genre is close to its
definition — and nobody has listened to what EffNet recommends.

> **On seeds no earlier round touched, do EffNet's picks sound more similar to
> the seed than `heard`'s, to a listener who cannot tell which is which?**

A second question rides along: are Tim's far tiers (Bridge, ranks 30–150;
Adventurous, ranks 500–2000) still heard as related, in either space?

## The design

Twelve seeds, one per genre first and the rest drawn at random, none of them
used as a seed or a pick in rounds one or two. Seven pairs per seed, 84 in all:

| pairs | source | why |
|---|---|---|
| 2 | EffNet top five | the thing under test |
| 2 | `heard` top five | the baseline it has to beat |
| 1 | EffNet far band | Tim's tier, by ear |
| 1 | `heard` far band, same ranks | **symmetry** — round two drew its deep pairs from `heard` only, which penalised it |
| 1 | random | the floor that ties this round's scale to the earlier two |

Six seeds use the Bridge band and six the Adventurous band, so both get heard
without lengthening the round. Both spaces are ranked over the same catalogue
(the tracks present in both), so a rank means the same thing on each side.

**Same-artist pairs are excluded from every source.** Round two had 13 of 72,
and removing them changed its conclusion. "Top five" means the five nearest
after the seed's own artist is removed; the original rank is kept in the key.

Everything else follows round two exactly: blind pair ids (`S01`…), a coin flip
for which clip plays first, volume matched from RMS energy, the same 1–5 form,
a shared core every rater hears for agreement. Raters are `R1c`…`R6c` so their
files cannot overwrite earlier rounds. **Amin's own ratings are a pilot and are
never counted** — he is the author.

## The threshold

Primary measure: each pair's score is the mean over the raters who heard it;
EffNet top pairs (24) are compared with `heard` top pairs (24) by a two-sided
Mann–Whitney U test.

- **EffNet higher, p < 0.05** — the switch to EffNet is validated by ear. Say so.
- **No detectable difference** — the most likely outcome at this sample size.
  EffNet is *not* validated by ear. It may stay in the app on other grounds, but
  the claim that it sounds more alike is withdrawn and we say its advantage is
  unconfirmed.
- **`heard` higher, p < 0.05** — the switch is not supported. The offline
  numbers measured something other than sounding alike. Report it, and the app
  goes back to `heard`.

A per-seed paired comparison (Wilcoxon signed-rank on the 12 seed-level
differences) is reported beside it as a robustness check. It does not override
the primary result in either direction.

## Secondary questions, reported whatever they show

1. **The scale check comes first.** Random pairs averaged 1.87 in round one and
   1.27 in round two. If round three's floor lands far from both, comparisons
   across rounds are made on the gap above the floor, not raw scores.
2. **Do the far tiers hold up?** Far-band pairs against random (one-sided,
   greater), per space and per band. A band that does not beat random is
   reported as failing, not quietly retuned.
3. **Ranking among the related.** Round two found that within the top five
   neither space's distance correlated with what people heard. The same
   correlation is computed for EffNet's top pairs. A clear negative correlation
   would be the first evidence that a space ranks among related tracks, not
   just separates related from unrelated.

## What gets reported

**Whatever comes back.** A positive result earns a slide; a null or negative one
goes on the caveats slide and changes what the defense deck claims about
EffNet. A round that only gets described when it agrees with us is not
evidence.

## What this cannot settle

With roughly six raters and 24 pairs per side, only a large difference will
clear the threshold. A null result means "not shown", not "equal". It also
says nothing about the tier logic in Tim's `modeling/tiers.py` beyond the rank
bands: the same-genre cap and era gate are not part of this round.

---

Built by `uv run python -m modeling.round3`. Returns go in
`data/listening_test/responses/` as `<rater>_ratings.csv`, and
`uv run python -m modeling.listening --csv` merges all three rounds.
