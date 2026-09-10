# QB Team Pass Opportunity Play/Rate Decomposition V1 — Result

## Disposition

`PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`

This is a diagnostic routing result only. It does not promote or alter production.

## Canonical lineage

- Branch: `research-qb-team-pass-opportunity-play-rate-decomp-v1`
- Tested/workflow head: `36f100799c81769d1ade998c7304d932e5914b44`
- Run: `34535405829`
- Job: `103065737323`
- Artifact: `10175200512` (`qb-team-pass-opportunity-play-rate-decomp-v1`)
- Artifact digest: `sha256:f2a79b330c7cc6f24d6b83479806478e5534afd2bd09961d37304a2c247b6bb4`
- Parent opportunity-chain run: `34523313743`
- Parent opportunity-chain artifact: `10170531084`
- D2 predecessor run: `34534830257`
- D2 result commit: `0b965ec9c9373bf37401cc119ffecdac6e4f8aee`

## Integrity

All frozen integrity gates passed.

- exact M89/M90 rows: `884`
- exact shared WR target rows: `440`
- exact shared WR reception rows: `884`
- 2024/2025 PBP loaded: PASS
- target PBP used only as diagnostic labels: PASS
- model fitting: `0`
- sportsbook inputs: `0`
- production changes: `0`
- predicted identity max gap: `7.105427357601002e-15`
- actual identity max gap: `7.105427357601002e-15`
- Shapley identity max gap: `1.0658141036401503e-14`

## Critical production finding

The promoted M89 opportunity path uses a fixed pass-opportunity/dropback rate:

- mean: `0.57`
- min: `0.57`
- max: `0.57`
- standard deviation: effectively `0`
- distinct rounded values: `1`
- share equal to `0.57`: `100%`

This is intentional lineage from the promoted 57/43 opportunity foundation, not a mechanical wiring failure.

## Two-factor decomposition

The corrected M89 team-pass-opportunity quantity was decomposed exactly as:

`TOTAL_OFFENSIVE_PLAYS × PASS_OPPORTUNITY_RATE`.

Pooled 2024-2025 mean-absolute Shapley contributions:

- `PASS_OPPORTUNITY_RATE`: `5.1714506751` opportunities
- `TOTAL_OFFENSIVE_PLAYS`: `4.3065421286` opportunities

`PASS_OPPORTUNITY_RATE` was the larger component under the frozen routing rule.

## Oracle recoverability

Pooled baseline team pass-opportunity MAE:

- baseline: `7.2165003165`
- perfect offensive plays: `5.3465158371`
- perfect pass-opportunity rate: `4.4057018929`

Therefore rate-side oracle recovery (`2.8108` MAE) exceeds play-volume oracle recovery (`1.8700` MAE).

The result is stable in both seasons:

### 2024
- baseline D MAE: `7.4034391293`
- perfect plays D MAE: `5.5027927928`
- perfect rate D MAE: `4.4253322384`

### 2025
- baseline D MAE: `7.0278620599`
- perfect plays D MAE: `5.1888181818`
- perfect rate D MAE: `4.3858930896`

## Shared receiver attribution

Under the frozen routing gates:

### PASS_OPPORTUNITY_RATE
- 2025 Spearman vs WR target-mass residual: `0.5229848576`
- pooled 2024-2025 Spearman vs WR reception-mass residual: `0.3663928852`
- largest pooled mean-absolute contribution: PASS
- season stability: PASS
- WR-target absolute Spearman >= 0.30: PASS
- WR-target lead >= 0.10 over plays component: PASS
- pooled WR-reception absolute Spearman >= 0.25: PASS

### TOTAL_OFFENSIVE_PLAYS
- 2025 Spearman vs WR target-mass residual: `0.4214355446`
- pooled 2024-2025 Spearman vs WR reception-mass residual: `0.3592811684`
- largest pooled contribution: FAIL
- season stability: FAIL
- WR-target lead gate: FAIL

Only `PASS_OPPORTUNITY_RATE` clears all frozen PRIMARY gates.

## Scientific meaning

The previously isolated upstream team-pass-opportunity miss is not primarily a total-play-volume problem. The larger and more receiver-linked internal error is the rate at which offensive plays become pass opportunities/dropbacks.

The current fixed `0.57` rate is therefore the exact next research bottleneck.

This does **not** authorize generic pass-rate history, PROE, score-state, lead/trail, pace, or M64/M65 state-occupancy retesting. Those families remain closed under the M80/M82 no-retest ledger. The promoted 57/43 foundation may only be challenged by materially new independent pregame information or a genuinely untested architecture linkage.

## Next authorized step

Perform source/provenance audits for materially new pregame information that can plausibly predict a directional departure from 57% pass opportunity rate.

A particularly relevant untested architecture family is **directional personnel consequence**: prior opportunity/value lost from unavailable backfield/run personnel versus unavailable receiving personnel. M77/M79 grouped offensive skill personnel together and did not test a run-capacity-versus-pass-capacity imbalance as a pass-opportunity-rate mechanism.

No predictive model is authorized until that source contract is frozen and passes leakage/coverage checks.
