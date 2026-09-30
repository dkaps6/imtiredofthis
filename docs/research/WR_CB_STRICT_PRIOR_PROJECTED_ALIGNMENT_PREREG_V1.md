# WR-CB Strict-Prior Projected-Alignment Effect — Preregistration V1

Date: 2026-09-30  
Status: **FROZEN DESIGN; SOURCE-READINESS GATE NOT MET; NO TARGET OUTCOMES AUTHORIZED YET**

## Research question

Does an explicitly sourced, pregame **projected WR↔CB alignment identity** add predictive information for WR receiving yards beyond the canonical no-WR-CB football baseline when the CB effect is estimated only from that CB's strict-prior, same-season sourced assignments?

This is not a test of FantasyAlarm's editorial grade and not a claim of observed route-by-route shadow coverage.

## Immutable interpretation

The source observable is named:

`PROJECTED_ALIGNMENT_PAIRING`

It means only that the public pregame report projected a WR against a named CB in an explicit left/right/slot alignment bucket.

It must never be renamed or interpreted as:
- actual route responsibility;
- actual man-coverage shadow rate;
- charted defender-on-target responsibility.

## Primary source population

Primary V1 uses **outside alignments only**:
- `LWR_VS_RCB`
- `RWR_VS_LCB`

Slot rows are excluded from the primary test because older source reports describe only selected slot matchups in some seasons. Slot may receive a separately preregistered test only if source completeness is independently established.

Every eligible row must:
1. come from an immutable source lock;
2. have an archive/capture time before that player's applicable kickoff;
3. have exact-week WR identity on the stated offense;
4. have exact-week CB identity on the stated opponent;
5. have schedule-consistent team/opponent identity;
6. use no provider-ID bridge, fuzzy/manual rescue or editorial matchup grade.

Missing source rows are **missing**, never zero exposure.

## Discovery / confirmation split

- discovery/training source years: **2021-2024**
- protected confirmation source year: **2025**
- 2026: prospective/live source collection and, if needed, later untouched confirmation

The split may not move because of outcome performance.

The existing protected 2025W14 source lock does not authorize 2025 outcome access.

## Source-readiness gate before any outcome-based science

Do not join target outcomes or fit parameters until all of the following are true:

1. at least **12 independently verified discovery source weeks** exist across at least 2 discovery seasons;
2. after strict-prior history construction, at least **10 distinct scored target weeks** remain;
3. at least **200 scored outside WR rows** remain;
4. no single week contributes more than 20% of scored rows;
5. every scored target row has at least **3 earlier same-season explicit outside assignments** for its projected CB.

The current two discovery locks (2022W5 + 2024W1) contain 82 strict rows but produce **0 V1 scored rows**, because there is no same-season strict-prior CB assignment history.

If archive acquisition exhausts before this gate is met, V1 stays source-blocked and waits for prospective collection. The threshold may not be lowered after seeing outcomes.

## Frozen baseline

The comparison baseline is the canonical production-eligible WR receiving-yards football model **with no WR-CB feature and with the retired legacy `coverage_penalty()` absent**.

Before the first outcome-based run:
- freeze the exact baseline commit SHA;
- freeze the historical prediction artifact hash;
- freeze the eligible source-lock hashes;
- record all four in the run manifest.

No sportsbook line, price, implied probability or closing line may enter the football feature construction or source screen.

## Strict-prior CB signal

For a target row at season S, week W, projected CB C:

1. take only immutable outside pairing rows for C from the **same season S** and weeks strictly less than W;
2. require at least 3 such prior assignments;
3. for each prior assignment j, compute the canonical baseline residual:

`resid_yards_j = actual_receiving_yards_j - baseline_receiving_yards_j`

4. define the target's CB history signal as the median prior residual:

`cb_prior_resid_median = median(resid_yards_j)`

Positive values mean receivers paired to that CB historically exceeded the canonical baseline; negative values mean they fell short.

No current-week outcome enters the feature.
No prior-season carryover is allowed in V1.
No manually selected CB subgroup, WR tier, team, depth-chart role or evidence-state carveout is allowed.

## Discovery fit

The only fitted model parameter in V1 is one scalar coefficient `beta` applied to the frozen signal:

`adjusted_prediction = baseline_prediction + beta * cb_prior_resid_median`

Discovery fitting rules:
- fit `beta` with **no intercept**;
- constrain `beta` to `[0, 1]`;
- evaluate out of fold using leave-one-week-out discovery folds;
- target rows remain strict-prior inside every fold;
- no alternate transformations, nonlinearities or thresholds may be tried after outcomes are visible.

If the unconstrained relationship would require a negative beta, V1 is considered unsupported rather than inverted.

## Primary discovery endpoint

Primary endpoint: paired change in absolute receiving-yards error:

`delta_AE = abs(adjusted - actual) - abs(baseline - actual)`

Discovery passes only if all are true:
1. mean out-of-fold `delta_AE < 0`;
2. at least 60% of scored weeks have negative mean `delta_AE`;
3. a deterministic week-cluster bootstrap (seed `20260930`, 10,000 resamples) gives a 95% CI whose upper bound for mean `delta_AE` is below 0.

Receiving-yards MAE is the only primary endpoint.

Receptions, longest reception, touchdowns, sportsbook hit rate and ROI are not co-primary endpoints and cannot rescue a failed V1.

## Confirmation freeze and one-shot rule

Only after discovery passes:
1. refit the same one-parameter beta on the full discovery population;
2. freeze beta, code SHA, source hashes, baseline hash, row population and all metrics;
3. verify confirmation-source readiness without opening confirmation outcomes.

Confirmation readiness requires:
- at least **8 distinct scored confirmation weeks**;
- at least **150 scored outside WR rows**;
- the identical same-season, minimum-3-prior-assignment rule.

Then protected confirmation outcomes may be accessed exactly once.

Confirmation passes only if:
1. aggregate paired MAE improves;
2. at least 60% of scored confirmation weeks improve;
3. the week-cluster bootstrap 95% CI upper bound for mean `delta_AE` is below 0.

No retuning after confirmation is opened.

If confirmation fails, V1 closes. Do not create a WR tier carveout, team carveout, alignment-specific rescue, top-N rescue, evidence-state rescue or opposite-sign version from the same confirmation outcomes.

## Production boundary

Even a successful confirmation does not restore the retired static coverage heuristic.

A successful V1 would authorize only a new, explicitly named WR receiving-yards projected-alignment feature under its own production adapter and shadow audit. Other WR markets and other positions remain unchanged unless separately preregistered.

## Current state at freeze

- 2022W5 discovery lock: 36 strict rows
- 2024W1 discovery lock: 46 strict rows
- discovery strict rows total: 82
- current V1 scored rows: **0**
- protected 2025W14 strict source rows: 47
- protected outcomes accessed: **false**
- parameters fit: **0**
- source/model gate: **CLOSED**
