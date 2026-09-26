# Availability -> Opportunity Rule-Order Audit V1 — Frozen Plan

Date frozen: 2026-09-26
Status: **DIAGNOSTIC ONLY — FROZEN BEFORE RESULT INSPECTION**

Branch: `research-public-intent-week3-prospective-v1`

## Question

Does the production availability-first architecture remove definitive-unavailable skill players before the existing opportunity-redistribution rules can observe and redistribute their vacated strict-prior opportunity?

This is a systems-integrity audit, not a new predictive feature and not a production candidate.

## Why this is being tested

Current production explicitly resolves availability before opportunity:
- `build_current_player_availability_v1.py` sets `definitive_unavailable=1`;
- `build_reconciled_active_roles_v1.py` removes those players from the active-role artifact;
- PlayerForm consumes the active-role authority;
- PlayerContext is built from PlayerForm;
- `simulation_rules._injury_target_overrides()` can redistribute an injured WR alpha only if that injured WR still exists in PlayerContext.

The existing rule unit test passes an OUT WR directly into PlayerContext. Production availability-first routing appears to make that state unreachable for a definitive OUT/IR/PUP player.

The explicit target-entitlement layer does not recreate missing player opportunity. It conserves the surviving modeled target-share sum up to the 0.95 cap and places the remainder in the residual receiver bucket.

RB has no equivalent production carry-vacancy redistribution rule at all; the separately frozen RB Vacancy Opportunity V1 is already prospectively testing one candidate and must remain untouched.

## Frozen diagnostic scope

Candidate variants scored: **0**
Parameters fit: **0**
Target-game outcomes read: **0**
Sportsbook inputs: **0**
Production mutations: **0**

Audit only:
1. static execution-path reachability;
2. current pregame Week-3 availability state if immutable no-outcome artifacts are available;
3. strict-prior opportunity evidence only;
4. current baseline rule/entitlement state only.

## Primary code-path gates

### Gate A — definitive-unavailable removal
Prove whether `definitive_unavailable==1` players are excluded from:
- reconciled active roles;
- PlayerForm current universe;
- PlayerContext/simulation universe.

### Gate B — legacy WR injury-redistribution reachability
For definitive OUT/IR/PUP WRs, determine whether `_injury_target_overrides()` can ever receive the removed player under the production path.

If the player is absent before PlayerContext, the rule is classified:
`UNREACHABLE_FOR_DEFINITIVE_UNAVAILABLE_PRODUCTION_PLAYER`

This is a code-path classification, not a performance judgment.

### Gate C — target entitlement behavior
Determine whether `TEAM_TARGET_ENTITLEMENT_V1`:
- restores removed strict-prior target mass;
- renormalizes surviving target mass upward;
- or leaves missing mass in residual.

Current code suggests the third behavior; verify exactly.

### Gate D — rushing behavior
Determine whether any production rule transfers a definitive-unavailable RB/FB's strict-prior rush share to surviving rushers before Monte Carlo.

The already-frozen prospective RB Vacancy V1 is excluded from this baseline diagnostic because it is not production.

## Current-event descriptive audit

If Week-3 no-outcome availability + strict-prior artifacts are available, report for every definitive-unavailable RB/FB/WR/TE:
- team / player / position;
- prior opportunity share available to production before removal;
- whether the unavailable player survives each production stage;
- surviving team modeled target/rush-share sum;
- residual probability before/after removal where reconstructible;
- any production redistribution flag reached;
- no actual Week-3 usage/outcome fields.

No player/team is selected based on outcome.

## Interpretation labels

Possible dispositions:
- `NO_RULE_ORDER_GAP`
- `WR_DEFINITIVE_VACANCY_RULE_UNREACHABLE_CONFIRMED`
- `RB_DEFINITIVE_VACANCY_NO_TRANSFER_CONFIRMED`
- `AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`
- `INSUFFICIENT_CURRENT_ARTIFACTS_CODE_PATH_ONLY`

Multiple sub-findings may coexist.

## No-go / no-rescue rules

Do not:
- modify RB Vacancy Opportunity V1;
- use DEN/PIT outcomes;
- design a WR transfer coefficient before the diagnostic is complete;
- broaden vacancy to QUESTIONABLE/DOUBTFUL;
- infer exact vacated share from target-game outcomes;
- tune recipient weights;
- retune M38, WR-R15, TE-R5P, ensemble weights, or residual cap;
- use sportsbook props/odds upstream.

If a rule-order gap is confirmed, any repair must be a separately frozen experiment with historical/prospective evidence. This audit alone cannot promote a fix.
