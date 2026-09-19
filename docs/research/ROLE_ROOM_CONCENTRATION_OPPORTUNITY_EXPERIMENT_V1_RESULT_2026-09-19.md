# Role/Room Concentration Opportunity Experiment V1 — Result

**Date:** 2026-09-19  
**Branch:** `research-football-context-execution-v1`  
**Frozen-plan commit:** `ababa341`  
**Evaluator commit:** `3ce0a6bc`  
**Execution commit:** `5fd0d835641d2bfc64f4d5af479b33c5b634c3ae`  
**Actions run:** `35448015251`  
**Job:** `105910213890`  
**Artifact:** `10586007027`  
**Artifact digest:** `sha256:a8746a4789745ffbc12278f19dced86f8656728b99d4ab23f66aad7ab2538749`

## Disposition

`ROLE_ROOM_CONCENTRATION_OPPORTUNITY_V1_FAIL_CLOSED_NO_PRODUCTION_CHANGE`

The preregistered V1 mechanism experiment failed its frozen primary-holdout gates for all three tested families. The candidate family therefore does not advance to production and must not be rescued by post-hoc threshold, feature, subgroup, or model retuning under the same V1 hypothesis.

## Scientific boundary

This experiment tested only the intended opportunity/entitlement mechanism:

- RB room rush concentration -> RB rushing opportunity;
- WR room target concentration -> WR target opportunity;
- TE room target concentration -> TE target opportunity.

It did not use sportsbook data, betting outcomes, target-game context leakage, or final-yard targets as the primary mechanism. The historical base was deterministically rehydrated from the canonical repository pipeline for 2019-2025.

## Integrity / execution

Focused qualification tests: **24 passed**.

Historical base:

- player-games: **37,104**;
- player-game SHA256: `33e9ba1e98e7d9057642cca6707925f4fd3ee3b3e7cd3a246e46ced15c7020e8`;
- context SHA256: `95ef8f324dfc644f6cfd1f530dcc6667dac2b3ec79705075f6c5d87078fc63c6`;
- sportsbook read: **false**.

The preceding qualification evidence remains valid: room concentration is stable, broadly available, identity-safe, and incrementally non-redundant versus canonical PlayerForm opportunity state. This predictive failure does not invalidate that descriptive finding; it means the tested V1 opportunity mechanism did not improve enough under the frozen predictive gates.

## Frozen gate result

| Family | Season | MAE rel delta | RMSE rel delta | p90 rel delta | Primary season pass |
|---|---:|---:|---:|---:|---:|
| RB_RUSH | 2024 | +0.000543 | +0.000124 | +0.000771 | **FAIL** |
| RB_RUSH | 2025 | +0.000453 | +0.000247 | +0.001028 | FAIL |
| WR_TARGET | 2024 | -0.000961 | -0.000926 | +0.003792 | **FAIL** |
| WR_TARGET | 2025 | -0.000583 | +0.000129 | +0.006170 | PASS season-level gate, but family remains closed because 2024 primary failed |
| TE_TARGET | 2024 | +0.001074 | +0.000201 | +0.003909 | **FAIL** |
| TE_TARGET | 2025 | -0.002339 | -0.000896 | -0.008597 | PASS season-level gate, but family remains closed because 2024 primary failed |

Positive relative deltas are worse for error metrics; negative deltas are better.

The decisive frozen failure was the 2024 primary holdout. Replication-year improvement cannot rescue a primary-holdout failure.

Final evaluator dispositions:

- `RB_RUSH = MECHANISM_FAIL_CLOSED_V1`
- `WR_TARGET = MECHANISM_FAIL_CLOSED_V1`
- `TE_TARGET = MECHANISM_FAIL_CLOSED_V1`

## Interpretation

Room concentration is real football context, but under this preregistered V1 formulation it is not a sufficiently useful direct additive predictor of individual next-game opportunity to justify production integration.

The WR and TE replication-year behavior is directionally interesting but cannot be used to retune or rescue V1. It may only motivate a genuinely distinct future hypothesis if that hypothesis has a different football mechanism and is frozen independently before outcomes are inspected.

The correct response is therefore to preserve the negative evidence and move to a different qualified mechanism rather than feature-engineer around the failure.

## Production impact

None.

`main` remains unchanged. No production science, pricing, calibration, or Full Slate behavior is modified by this result.

## No-retest rule

Do not rerun the same V1 hypothesis with:

- different concentration thresholds;
- post-hoc feature subsets;
- alternate regression strength chosen after this result;
- transition-only rescue cohorts;
- 2025-selected variants;
- direct final-yard targets used merely to bypass the failed opportunity mechanism.

Any future use of room concentration must be attached to a genuinely distinct, preregistered football mechanism.

## Next scientific lane

The next highest-value context lane is **event/regime qualification**, not a concentration rescue.

Priority candidates:

1. team-change / new-team cold-start state;
2. joint player-usage + room-transition state;
3. personnel/vacancy events with explicit event semantics;
4. uncertainty/calibration effects in novel-role regimes rather than direct mean boosts.

These event variables should not be forced through adjacent-period persistence gates designed for tendencies. Freeze an event-specific qualification contract first, emphasizing pregame observability, event precision, support, identity integrity, novelty versus production state, and an uncertainty/regime mechanism. Only after qualification should a new predictive plan be frozen.
