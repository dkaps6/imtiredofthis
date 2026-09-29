# Weeks 1-3 Projection Authority Line-Conflict V1 — Result

Date: 2026-09-29  
Status: **COMPLETE — DIAGNOSTIC ONLY, NO PRODUCTION CHANGE**  
Branch: `research-week3-postmortem-execution-v1`

Frozen plan:
- `docs/research/WEEKS1_3_PROJECTION_AUTHORITY_LINE_CONFLICT_V1_PLAN.md`
- frozen-plan commit `c6b33188d25d34fad5ac1905ae7fd8891e13e08c`

Implementation:
- `scripts/research/analyze_weeks1_3_projection_authority_line_conflict_v1.py`
- implementation head `db39cc4fc0b57d3232beb0ca617c23df4d0ecab5`

Immutable source:
- run `36626121661`
- artifact `11060101978`
- artifact `weeks1-3-postmortem-diagnostics-v1-36626121661`
- verified artifact digest `sha256:8833b5611cb979d585932d94ea071b26d9dfbdad3784c196381ff376681d23e3`
- source file `projection_authority_attribution_v1/projection_authority_row_detail.csv`
- source rows: 1,240
- eligible rows: 1,240
- excluded rows: 0
- unique NFL game clusters: 45

## Frozen disposition

`NO_CLEAR_CURRENT_SEASON_AUTHORITY_LINE_CONFLICT_SIGNAL`

The primary preregistered hypothesis was **not supported**.

### Primary contrast: MC and final projection cross the sportsbook line

`CROSSED_LINE`:
- rows: 119
- game clusters: 43
- win rate: 50.42%
- units: -4.69u
- ROI: -3.94%
- final MAE: 23.15
- MC MAE: 24.83
- final model-closer-than-line rate: 48.74%
- paired MC -> final absolute-error improvement: +1.68

All non-crossed rows:
- rows: 1,121
- game clusters: 45
- win rate: 50.76%
- ROI: -3.21%
- final MAE: 16.96
- final model-closer-than-line rate: 45.50%

Crossed minus non-crossed:
- win-rate difference: **-0.34 percentage points**
- final model-closer-rate difference: **+3.24 percentage points**
- final-MAE difference: **+6.19**

The frozen signal rule required:
- >=50 crossed rows: PASS
- >=15 crossed game clusters: PASS
- crossed win rate >=5pp worse: **FAIL**
- crossed model-closer rate >=8pp worse: **FAIL**
- at least one corresponding cluster-bootstrap 95% interval fully below zero: **FAIL**

Game-cluster bootstrap, 10,000 replicates, seed 42029:
- win-rate difference 95% interval: **[-8.15pp, +8.00pp]**
- model-closer-rate difference 95% interval: **[-7.55pp, +14.45pp]**
- final-MAE difference 95% interval: **[-0.20, +13.71]**

Therefore simply crossing the sportsbook line is **not** a supported unreliability state in Weeks 1-3.

## Predeclared authority-state summaries

The frozen plan required all authority states to be reported even though only crossed-vs-noncrossed was the primary inference.

### SAME_SIDE_STRENGTHENED

Definition: MC and final stayed on the same side of the line, but final authority moved the mean farther away from the line.

- rows: 250
- game clusters: 44
- win rate: **44.00%**
- units: **-38.15u**
- ROI: **-15.26%**
- final MAE: 17.52
- MC MAE: 16.65
- final model-closer rate: **40.00%**
- paired MC -> final absolute-error improvement: **-0.87** (final worse)

### SAME_SIDE_WEAKENED

Definition: MC and final stayed on the same side, but final authority moved the mean closer to the line.

- rows: 471
- game clusters: 45
- win rate: **55.20%**
- units: **+24.44u**
- ROI: **+5.19%**
- final MAE: 16.72
- MC MAE: 17.86
- final model-closer rate: **49.68%**
- paired MC -> final absolute-error improvement: **+1.14** (final better)

### SAME_SIDE_NO_MATERIAL_DISTANCE_CHANGE

- rows: 400
- game clusters: 43
- win rate: 49.75%
- units: -22.32u
- ROI: -5.58%
- final MAE = MC MAE: 16.87
- final model-closer rate: 44.00%

No rows landed exactly on the line under the frozen 1e-12 tolerance.

## Interpretation boundary

The headline primary hypothesis failed. **Do not create a line-crossing exclusion rule.**

A separate, secondary pattern is nevertheless visible in a state that was defined before result inspection:
- late authority moves **farther from the market line** performed poorly;
- late authority moves **toward the market line** performed materially better.

That pattern is scientifically interesting but is **not promotion evidence**:
- Weeks 1-3 outcomes were already available when this diagnostic was designed;
- the authority states are confounded with market, position, and which production layers can move each mean;
- multiple predeclared state summaries were inspected;
- no cluster-aware inferential gate was preregistered for strengthened-vs-weakened.

Therefore the only defensible use of this secondary finding is to freeze a genuinely prospective confirmation contract before the next board is graded.

## No-change statement

This result authorizes:
- no production filter;
- no probability or edge adjustment;
- no market/position carveout;
- no mean shrink toward Vegas;
- no sportsbook information upstream of football generation.

The current production stack remains unchanged.
