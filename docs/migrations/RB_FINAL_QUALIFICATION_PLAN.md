# RB Final Qualification Plan

## Status

**Frozen before any new final-qualification scoring.**

This is an evaluation/decision package only. It does not fit a model, search features, tune thresholds, tune weights, choose a model family, or use sportsbook information upstream.

## Frozen parents and exact source artifacts

### Football casebook

Use the exact original STACK3 artifact and casebook:

- run `33539468967`
- SHA `9d7ea5d0173569ac9e4633685da7e91eed5fcd3d`
- artifact `9812993290`
- artifact name `rb-stack3-frozen-state-composition`
- file `stack3_2025_casebook.csv`
- expected rows: **1,393**

P3 is frozen as:

- **P3 rushing yards** = `stack_yards` in Week 1, otherwise `arch_enriched_opp_stack_eff_yards`.
- **P3 carries** = `stack_att` in Week 1, otherwise `enriched_att`.

Comparison football arms, where present in the exact casebook:

- `M94C`: `m94c_att`, `m94c_yards`
- `FULL_STACK`: `stack_att`, `stack_yards`
- `STACK2_ENRICHED`: `enriched_att`, `arch_enriched_opp_stack_eff_yards`
- `P3`: Week-1 override defined above

No alternate carry or yard composition may be substituted after scoring.

### Exact downstream market universe

Use the exact original STACK5 artifact:

- run `33540065380`
- SHA `c9087faad1bcb4788dc3ea4281bf637ce3b3af99`
- artifact `9813220944`
- artifact name `rb-stack5-market-gap-forensics`
- file `stack5_899_casebook.csv`
- expected rows: **899**

Sportsbook consensus is grading/benchmark information only. It is not allowed into any football projection calculation.

## Frozen identity checks

The evaluator must fail closed unless all of the following reproduce from the exact inherited artifacts within floating-point tolerance (`1e-9` for scalar identities unless unavoidable CSV roundoff requires `1e-8`):

1. STACK3 football rows = `1393`.
2. P3 all-RB rushing-yard MAE = `19.94952397834036`.
3. P3 all-RB rushing-yard RMSE = `28.86651928636813`.
4. Exact market rows = `899`.
5. P3 market rushing-yard MAE = `24.315798244183124`.
6. Vegas market rushing-yard MAE = `23.701890989988875`.
7. P3 market rushing-yard RMSE = `33.34291378702183`.
8. Vegas market rushing-yard RMSE = `32.493543467503315`.
9. No duplicate `(season, week, team, player_clean_key/join_key)` rows in the football casebook.
10. No duplicate player-game rows in the exact market casebook.

Failure of an identity check is a source/integration failure, not a model failure.

## Football-only player qualification

For each frozen arm, compute carries and rushing yards with:

- n
- MAE
- RMSE
- signed bias = prediction - actual
- Pearson correlation
- median absolute error
- 75th percentile absolute error
- 90th percentile absolute error
- overprojection rate
- underprojection rate
- exact-hit rate where applicable

### Frozen time slices

- ALL
- W1
- W2–5
- W6–12
- W13–18

### Frozen actual-carry workload slices

- actual carries >=10
- >=15
- >=20
- >=25

These are grading slices only; they are never available to the prediction model.

### Frozen actual rushing-yard slices

- actual rushing yards >=50
- >=75
- >=100

These are grading slices only.

### Frozen depth-role slices

The exact STACK3 casebook contains `depth_rank`. If finite depth-rank coverage is at least 80% on the 1,393-row universe, score these predeclared slices and report missing-depth separately:

- RB1 = `depth_rank == 1`
- RB2 = `depth_rank == 2`
- SECONDARY = `depth_rank >= 3`
- DEPTH_UNKNOWN = missing/non-finite depth rank

Do not invent or infer role labels from target-game outcomes.

### Tail/state diagnostics

Report, but do not route the central point estimate through:

- `state_m95f_risk`
- `state_vacancy`
- `state_m95i_tail`

These remain diagnostics only.

## Frozen football production decision

This final package is not allowed to select a new arm. P3 is the only promotion candidate.

Separate the decision into two statuses:

### A. Football production qualification

P3 is `FOOTBALL_QUALIFIED` only if all of the following hold on the exact 1,393-row casebook:

1. P3 rushing-yard MAE improves over M94C by at least **0.50 yards**.
2. P3 rushing-yard RMSE improves over M94C (>0 gain).
3. P3 rushing-yard correlation is not worse than M94C by more than **0.01**.
4. P3 carry MAE is not worse than M94C by more than **0.10 carries**.
5. P3 carry RMSE is not worse than M94C (> `-0.10` allowed tolerance).
6. P3 W13–18 rushing-yard MAE is better than M94C W13–18 MAE.
7. P3 does not worsen both the >=20-carry and >=100-rushing-yard MAE slices versus M94C. At least one may be flat/worse, but both cannot deteriorate simultaneously.

Otherwise status is `FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED` and no promotion is authorized.

These gates judge P3 against its football parent/reference, not against Vegas.

### B. Market competitiveness status

This is downstream benchmarking only and cannot alter the football model.

- `AGGREGATE_VEGAS_CLEARED` if P3 market MAE <= Vegas market MAE on the exact 899 rows.
- otherwise `AGGREGATE_VEGAS_NOT_CLEARED`.

No waiver.

A fixed disagreement-bin result may be reported as a diagnostic but may not be called a validated betting edge from this exposed retrospective sample.

## Exact 899-row downstream sportsbook benchmark

Score P3 and Vegas on:

- MAE
- RMSE
- signed bias
- Pearson correlation
- median absolute error
- P75 absolute error
- P90 absolute error

Also report:

- strict P3-closer count/rate
- Vegas-closer count/rate
- exact tie count/rate
- directional edge accuracy

Directional edge rule is frozen:

- if `P3 > consensus_line`, success = `actual_rush_yards > consensus_line`;
- if `P3 < consensus_line`, success = `actual_rush_yards < consensus_line`;
- exact zero-disagreement rows are excluded from directional-accuracy denominator and reported separately.

### Frozen P3-vs-Vegas disagreement bins

Use absolute `P3 - consensus_line`:

- `<2.5`
- `2.5–5`
- `5–7.5`
- `7.5–10`
- `>=10`

For each bin report:

- n
- P3 MAE
- Vegas MAE
- MAE difference (`Vegas MAE - P3 MAE`; positive favors P3)
- P3 strict closer rate
- tie rate
- directional edge accuracy
- mean signed P3-vs-Vegas disagreement

No bin thresholds may be changed after seeing results.

### Frozen market time slices

- W1
- W2–12
- W13–18

## Final disposition vocabulary

Output exactly one football status and one market status.

Football:

- `FOOTBALL_QUALIFIED`
- `FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED`
- `INTEGRITY_FAILURE`

Market:

- `AGGREGATE_VEGAS_CLEARED`
- `AGGREGATE_VEGAS_NOT_CLEARED`
- `MARKET_INTEGRITY_FAILURE`

The combined decision must explicitly state that market status does not change football qualification and sportsbook data remains downstream only.

## Required outputs

- `rb_final_integrity.csv`
- `rb_final_player_metrics.csv`
- `rb_final_error_distribution.csv`
- `rb_final_depth_metrics.csv`
- `rb_final_state_diagnostics.csv`
- `rb_final_market_overall.csv`
- `rb_final_market_bins.csv`
- `rb_final_market_time_slices.csv`
- `rb_final_gates.csv`
- `rb_final_disposition.csv`
- `rb_final_casebook.csv` (exact inherited player universe plus frozen P3 carry/yard columns only; no sportsbook join)
- `rb_final_market_casebook.csv` (exact inherited 899-row grading universe with derived diagnostics)

## After this run

- If football qualification passes, P3 is the RB production candidate subject to implementation/parity review on the production branch.
- If aggregate Vegas is not cleared, record the remaining market gap honestly. Fixed bins are diagnostics only; no retrospective threshold optimization is authorized.
- Regardless of market result, do not reopen STACK6 team-rush feature slicing without a genuinely new timestamp-safe pregame information source.
- Once the RB final decision is recorded, primary research moves to WR.
