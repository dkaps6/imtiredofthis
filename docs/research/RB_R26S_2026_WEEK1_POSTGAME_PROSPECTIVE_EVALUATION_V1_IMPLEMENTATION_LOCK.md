# RB R26S — 2026 Week 1 Postgame Prospective Evaluation V1 — IMPLEMENTATION LOCK

Status: **LOCKED BEFORE R26S OUTCOME INGESTION / EXECUTION**

## Frozen scientific plan
- path: `docs/research/RB_R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_V1_FROZEN_PLAN.md`
- frozen-plan commit: `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`

## Frozen evaluator
- path: `scripts/backtest/evaluate_rb_r26s_2026_week1_postgame_prospective_v1.py`
- implementation commit: `e42099dbf11cdd510eb64e43a0910444c75ec7f8`

The evaluator was authored after the frozen plan and before R26S read or inspected any 2026 Week 1 outcome values.

## Immutable parent pins

### R26Q
- run `34400524030`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- head `68661da94f03cab2f96182d47636cf55e088b5de`

### R26R
- run `34401814588`
- artifact `10124274040`
- digest `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- head `469aa40c90c738e12a82ee32ccca70c9cdbbc29f`
- frozen pregame result: `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`, 30/30 gates.

## Implementation invariants

The locked evaluator:
1. never reruns R26N/R26O/R26Q football projections;
2. never regenerates baseline or candidate means after outcomes;
3. verifies the 107 sealed R26Q receptions arrays and per-array hashes;
4. loads 2026 Week 1 weekly player stats only at execution time;
5. loads 2026 Week 1 snap counts only at execution time;
6. scores zero receptions only when offensive participation is confirmed (`offense_snaps > 0`);
7. excludes zero-snap/DNP and unresolved participation rows from the primary accuracy cohort;
8. uses the 104 changed sealed rows as the primary candidate universe;
9. uses the 3 unchanged CIN rows as controls;
10. uses 10,000 team-cluster bootstrap resamples with seed 42;
11. applies the frozen `>= 0.50 receptions` large-mover definition;
12. uses the pregame R26R role snapshot descriptively only;
13. uses R26R sportsbook evidence as downstream benchmark only;
14. cannot tune/refit/promote/activate production.

## Change control

After this lock is committed, any change to the frozen plan or evaluator invalidates the R26S execution unless it is a separately documented mechanical/value-neutral compatibility repair that leaves all scientific gates and thresholds unchanged.

Any valid scientific failure is final for R26S and must not be rescued through threshold, cohort, feature, or candidate changes.
