# Handoff — where Matcha stands (2026-10-05)

Reconstructed from the git history (44 commits on `feature/pca-recommendation`)
and the project docs, when work moved from Claude in VS Code to Claude in Cowork.

## What has been done

1. **Sep 14–20 — baseline.** Repo created and moved to `uv` (Mayank). Feature
   scaling, PCA, MLflow logging, baseline recommender + evaluation
   (precision@10, nDCG@10, genre_overlap@10, coverage), the
   Familiar → Adventurous dial (rank-500 ceiling), answers to the mentors' PCA
   questions.
2. **Sep 23–25 — listening test, round one.** 6 raters, 80 blind pairs.
   Listeners' answers led to rhythm features (`modeling/rhythm.py`) and the
   drum/instrument ratio (`modeling/separation.py`). Space reduced from 518 to
   154 columns (`heard`, 57 PCA components). README and `FOR_TIM.md` written.
3. **Sep 26–29 — round two.** On fresh seeds the round-one result did not hold
   (old −0.46 vs heard −0.43). The claim that `heard` sounds better was
   withdrawn; it ships on size grounds only.
4. **Oct 1 — round three built.** Tim moved to Discogs-EffNet. Threshold
   pre-registered in `data/listening_test/ROUND3_PREREGISTRATION.md`
   (EffNet top-5 vs `heard` top-5, two-sided Mann–Whitney U). 84 pairs, clips,
   rating forms and zips for raters `R1c`–`R6c` are ready.

## Open items

- [ ] Collect round-three ratings and run the pre-registered analysis.
- [ ] Commit the untracked files: `data/listening_test/ROUND2_PREREGISTRATION.md`,
      `pairs_key_r2.csv`, `pairs_key_r3.csv`.
- Related worktrees: `../tim-branch` (`baseline-recommender-tim`),
  `../tim-effnet` (`frontend/tastebridge-ui`).

## Running it

See `FOR_TIM.md`: `uv sync` → `modeling.rhythm` → `modeling.separation` →
`modeling.train_pca --feature-set heard --n-components 0.90` →
`modeling.recommender --track-id 2`.
