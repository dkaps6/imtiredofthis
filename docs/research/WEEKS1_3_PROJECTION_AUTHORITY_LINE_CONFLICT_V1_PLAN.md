# Weeks 1-3 Projection Authority Line-Conflict V1 — Frozen Plan

Date frozen: 2026-09-29  
Status: **FROZEN BEFORE RESULT INSPECTION — DIAGNOSTIC ONLY**  
Branch: `research-week3-postmortem-execution-v1`

## Question

When the upstream Monte Carlo mean and the final production mean fall on opposite sides of the exact selected sportsbook line, is that pregame authority-conflict state descriptively associated with worse realized reliability?

This is a pipeline-authority diagnostic. It is **not** a new betting rule, probability recalibration, side carveout, threshold search, or football-mean candidate.

## Immutable source

Use only the exact artifact from the already-completed projection-authority audit:
- run `36626121661`;
- artifact `11060101978`;
- name `weeks1-3-postmortem-diagnostics-v1-36626121661`;
- digest `sha256:8833b5611cb979d585932d94ea071b26d9dfbdad3784c196381ff376681d23e3`;
- file `projection_authority_attribution_v1/projection_authority_row_detail.csv`.

Expected source population: 1,240 settled selected bets.

No refetch, repricing, sportsbook acquisition, model rerun, outcome fitting, or row substitution is allowed.

## Required fields

Each scored row must have finite:
- `vegas_line`;
- `mc_proj`;
- `model_proj`;
- `actual`;
- `vegas_odds`;
and nonmissing:
- `event_id`;
- `position`;
- `market`;
- `side`;
- `bet_result`.

Rows failing those requirements are reported and excluded; no imputation.

## Frozen state definitions

Let:
- `mc_gap = mc_proj - vegas_line`;
- `final_gap = model_proj - vegas_line`;
- `authority_move = model_proj - mc_proj`;
- numerical zero tolerance = `1e-12`.

Classify exactly once:

1. `CROSSED_LINE`
   - `mc_gap * final_gap < 0`.

2. `FINAL_ON_LINE`
   - not crossed;
   - `abs(final_gap) <= 1e-12`;
   - `abs(mc_gap) > 1e-12`.

3. `MC_ON_LINE`
   - not crossed / not final-on-line;
   - `abs(mc_gap) <= 1e-12`;
   - `abs(final_gap) > 1e-12`.

4. `UNCHANGED_ON_LINE`
   - both gaps within tolerance.

5. `SAME_SIDE_STRENGTHENED`
   - same nonzero sign;
   - `abs(final_gap) > abs(mc_gap) + 1e-12`.

6. `SAME_SIDE_WEAKENED`
   - same nonzero sign;
   - `abs(final_gap) < abs(mc_gap) - 1e-12`.

7. `SAME_SIDE_NO_MATERIAL_DISTANCE_CHANGE`
   - remaining same-sign rows.

The primary contrast is `CROSSED_LINE` vs all non-crossed finite rows.

## Frozen outcomes / diagnostics

Report for each state and for the primary contrast:
- rows;
- unique game clusters;
- selected-bet win rate;
- flat-stake units and ROI using existing `unit_result`;
- final model MAE;
- MC MAE;
- final model-closer-than-line rate;
- mean paired absolute-error improvement `abs(mc-actual) - abs(final-actual)`;
- selected side agreement with MC mean side and final mean side.

Also report the primary contrast by:
- market;
- position;
only as descriptive cells. No multiple threshold search and no cell-specific rule.

## Cluster-aware uncertainty

For the primary `CROSSED_LINE` vs non-crossed contrast only:
- bootstrap NFL games, not rows;
- 10,000 replicates;
- seed `42029`;
- report 95% percentile intervals for:
  - win-rate difference;
  - model-closer-rate difference;
  - final-MAE difference.

No p-value fishing across subcells.

## Frozen interpretation rule

This Weeks 1-3 outcome-exposed pass can **never** promote a production rule.

It may only set one of two descriptive dispositions:

### `AUTHORITY_LINE_CONFLICT_CURRENT_SEASON_SIGNAL`
Requires all of:
- `CROSSED_LINE` >= 50 rows;
- >= 15 unique game clusters;
- crossed win rate at least 5 percentage points below non-crossed;
- crossed final model-closer rate at least 8 percentage points below non-crossed;
- at least one of those two corresponding cluster-bootstrap 95% intervals lies fully below 0.

### `NO_CLEAR_CURRENT_SEASON_AUTHORITY_LINE_CONFLICT_SIGNAL`
Otherwise.

Even if the first disposition is reached, the only authorized next step is a separately frozen prospective Week-4+ or clean historical confirmation. No 2026 Weeks 1-3 fitted threshold or exclusion rule may be deployed.

## Explicit do-not-do

- no edge threshold tuning;
- no probability rescaling;
- no QB/pass-yards carveout;
- no market/position rescue after reading cells;
- no redefinition of the crossing state;
- no use of Week-3-only outcomes;
- no sportsbook data upstream of football generation;
- no production mutation.
