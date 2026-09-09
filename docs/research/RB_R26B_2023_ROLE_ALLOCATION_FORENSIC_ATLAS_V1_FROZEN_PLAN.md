# RB R26B — 2023 Role-Allocation Forensic Atlas V1

Status: **FROZEN BEFORE NEW R26 V1 OUTCOME SLICING**
Date: 2026-09-09
Parent frozen candidate: `RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1`
Parent head: `9b9d49099ec0e62a168e37679cbd56e6b767131d`
Parent authoritative run: `34356222339`
Parent artifact: `10106271075`
Parent artifact digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
Forensic branch: `research-rb-r26b-2023-role-allocation-forensics-v1`

## Scientific label

This is a **retrospective forensic diagnostic only**. It cannot promote a model, authorize a shadow, change a production parameter, or redefine the frozen R26 V1 result.

R26 V1 remains `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW` because its predeclared gate 15 failed in 2023. The purpose of R26B is to identify the pregame allocation state that distinguishes 2023 from the five seasons where vacancy-gated R9 improved vacancy-incumbent receptions MAE.

No model is fit in R26B. No candidate is re-run. No threshold is optimized against outcome.

## Known facts allowed before this atlas

The parent frozen grader already established:

- R26 improves vacancy-incumbent receptions MAE in 5 of 6 seasons.
- 2023 worsens by about 4.54%.
- In 2023 vacancy RB1 MAE worsens about 6.23% and bias flips from underprojection to slight overprojection.
- In 2023 vacancy RB2+ MAE worsens about 3.32% and becomes more negatively biased.
- Pooled RB1 improves strongly; pooled RB2+ MAE improves slightly but bias/p90 deteriorate.

These known facts may motivate the forensic questions but may not be used to alter bins after inspection.

## Primary forensic question

> In vacancy-active RB rooms, what timing-safe pregame allocation state made the fixed R9 redistribution misassign receiving opportunity in 2023 while improving the other five seasons?

The atlas is designed to distinguish four broad explanations without fitting any of them:

1. **baseline-vs-identity ordering disagreement** — R9 moves mass toward the wrong current player;
2. **room composition / vacancy structure** — 2023 vacancy rooms differ in room size, exits, entrants, or continuity;
3. **identity-confidence mismatch** — large persistent-identity residuals are less reliable in certain transition states;
4. **lagged-role instability** — strictly-prior role evidence indicates the room was changing in a way baseline/R9 did not represent.

## Inputs

Only immutable outputs from the authoritative R26 V1 run and its already-leakage-safe source audit are allowed:

- season-level R26 prediction files for 2020-2025;
- R26 safe-transition player state;
- R26 source disposition / structural audits;
- parent frozen plan and result metadata.

No sportsbook data. No 2026 outcomes. No same-week historical depth chart. No new external source.

## Population

Primary:

- `VACANCY_ACTIVE == True`
- `continuing_same_team == True`
- RB/FB player-games with valid actual receptions and both baseline/candidate reception predictions.

The full vacancy-active population, including newcomers, must also be reported as context and may not be silently dropped.

All six parent seasons 2020-2025 must be reported. 2023 must be compared against both:

- pooled winning seasons: 2020, 2021, 2022, 2024, 2025;
- each individual season separately.

## Frozen diagnostic dimensions

No dimensions may be removed after inspection.

### A. Current baseline role / concentration

1. parent `rb_rank`: `RB1` versus `RB2+`;
2. production `baseline_room_share` bands:
   - `<0.25`
   - `[0.25,0.50)`
   - `[0.50,0.75)`
   - `>=0.75`;
3. current RB room size from safe-transition state:
   - `1`
   - `2`
   - `3`
   - `4+`.

### B. Vacancy composition

4. exits:
   - exactly `1`
   - `2+`;
5. entrants:
   - `0`
   - `1+`;
6. prior-to-current room-size delta:
   - shrank
   - unchanged
   - expanded;
7. target player continuity:
   - same-team incumbent
   - new-to-team veteran
   - no-prior-NFL-roster state.

### C. R9 identity state

8. `prior_rb_room_share` bands using the same fixed cut points as baseline share:
   - `<0.25`
   - `[0.25,0.50)`
   - `[0.50,0.75)`
   - `>=0.75`;
9. raw R8/R9 residual bands:
   - `< -0.25`
   - `[-0.25,0.25]`
   - `> 0.25`;
10. residual sign: negative / zero-near / positive under the same `±0.25` definition.

### D. Baseline-versus-candidate ordering

Within each vacancy-active team-game RB room, derive from prediction-only columns:

11. baseline within-room rank from `baseline_room_share`;
12. candidate within-room rank from `candidate_room_share`;
13. player rank movement:
   - moved up
   - unchanged
   - moved down;
14. top-player agreement:
   - baseline RB1 remains candidate top share
   - candidate changes the top-share player;
15. candidate share delta bands:
   - `< -0.10`
   - `[-0.10,-0.025)`
   - `[-0.025,0.025]`
   - `(0.025,0.10]`
   - `>0.10`;
16. room concentration change using Herfindahl index of within-RB shares:
   - candidate more concentrated
   - approximately unchanged (`|delta HHI| <= 0.02`)
   - candidate less concentrated.

These are diagnostic descriptions of the already-frozen candidate, not tunable gates.

### E. Strictly-prior role evidence

17. `prior_depth_covered`: yes/no;
18. when covered, lagged `prior_depth_team` / `prior_depth_position` must be reported only as source labels and normalized ordinal information if an unambiguous ordinal can be parsed without using outcomes;
19. whether the target player's lagged prior depth identity is compatible with its current production baseline rank, reported descriptively only.

Same-week depth is forbidden.

### F. Calendar phase

20. Week 1 versus Weeks 2+;
21. fixed phase buckets:
   - W1-4
   - W5-9
   - W10-13
   - W14-18.

## Frozen outcome diagnostics

For every sufficiently populated slice (`n >= 20`; smaller slices remain in raw output but are labeled `LOW_N`), report baseline and candidate:

- receptions MAE
- RMSE
- bias
- median absolute error
- p75 absolute error
- p90 absolute error
- Pearson
- Spearman
- candidate minus baseline MAE
- candidate minus baseline absolute bias
- candidate minus baseline p90

Targets must be reported in parallel for:

- MAE
- RMSE
- bias.

No `n < 20` slice may be used as the sole justification for a next candidate.

## Team-room attribution

Produce one row per vacancy-active team-game with:

- season/week/team/event_id
- current/prior room size
- exits_n / entrants_n
- baseline top player and share
- candidate top player and share
- whether top identity changed
- baseline/candidate room HHI
- sum player baseline absolute reception error
- sum player candidate absolute reception error
- room error delta
- actual team RB receptions
- predicted baseline team RB receptions
- predicted candidate team RB receptions.

Because the candidate preserves target mass but changes player allocation/catch identity, this table is necessary to distinguish room-total miss from within-room assignment miss.

## 2023-versus-winning-season comparison

The final atlas must explicitly answer, with descriptive effect sizes and row counts:

1. Was 2023 unusually high in baseline/candidate top-rank disagreement?
2. Was 2023 unusually high in large positive R9 residuals applied to baseline RB1s?
3. Was 2023 unusually high in rooms where candidate concentration increased?
4. Did 2023 have different exits/entrants/current-room-size composition?
5. Did 2023 failure concentrate in Week 1 or persist through later phases?
6. Did strictly-prior depth availability/role compatibility differ materially?
7. Were 2023 losses mostly **within-room misallocation** despite correct room-total direction, or were room totals themselves wrong?
8. Which diagnostic state is replicated as harmful in at least one additional season, even if smaller, versus being a 2023-only anomaly?

## Governance / stopping rules

R26B may identify a next hypothesis but may not instantiate or score it in the same forensic run.

Forbidden after inspection:

- changing R9 reliability from its training-derived value;
- changing the vacancy threshold from `>=1 exit`;
- selecting a favorable baseline-share/residual cutoff and calling it validated;
- dropping 2023;
- dropping RB2+;
- redefining Week 1;
- using same-week historical depth;
- using sportsbook or future outcomes as features.

A next candidate must be **materially different and predeclared in a new frozen plan**. Likely candidate families, only if supported by the atlas, include:

- an agreement/trust router that preserves baseline when persistent identity and current allocation state conflict;
- a role-stability router based solely on timing-safe pregame state;
- a separately modeled conserved vacancy inheritance allocation.

No one of these is authorized merely by writing this plan.

## Required outputs

- frozen plan SHA256
- forensic implementation SHA256
- player-level enriched atlas CSV
- slice metrics CSV
- team-room attribution CSV
- season-level comparison CSV
- 2023-versus-winning-season summary JSON
- source/integrity audit JSON
- exact run/job/commit/artifact lineage

## Disposition labels

- `FORENSIC_MECHANISM_IDENTIFIED`: at least one predeclared timing-safe state shows a coherent 2023 mechanism and has supporting directional evidence outside 2023; this authorizes only a new frozen candidate plan.
- `FORENSIC_2023_IDIOSYNCRATIC_NO_ROUTER`: no timing-safe state provides a coherent replicated explanation; do not build a post-hoc router from 2023.
- `FORENSIC_MECHANICAL_OR_SOURCE_FAILURE`: atlas integrity/source contract fails; no scientific interpretation.

No R26B disposition authorizes production or prospective shadow by itself.