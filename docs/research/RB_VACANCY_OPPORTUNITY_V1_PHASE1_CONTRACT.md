# RB Vacancy Opportunity V1 — Phase 1 Input + Redistribution Contract

Date: 2026-09-24
Branch: `research-rb-vacancy-opportunity-v1`
Status: **FROZEN BEFORE OUTCOME ATTACHMENT**
Parent: `docs/research/RB_VACANCY_OPPORTUNITY_V1_PLAN.md`
Phase-0 authority: `docs/research/RB_VACANCY_OPPORTUNITY_V1_PHASE0_AUDIT.md`

## Purpose

Freeze the no-outcome construction for the first sanctioned RB teammate-vacancy candidate. This document does not grade target-game outcomes and does not reopen M96.

## Availability event contract

A V1 vacancy event exists only when the canonical pre-opportunity availability ledger marks an RB/FB teammate:

- `definitive_unavailable == 1`; and
- `final_availability_state` is an `UNAVAILABLE_*` state.

`DOUBTFUL` and `QUESTIONABLE` do **not** create V1 vacancy events. The locked current resolver treats them as uncertain, not definitive unavailable.

The unavailable player is already absent from the canonical active-role / PlayerForm / simulation universe. V1 therefore does not add another zeroing rule; it estimates the missing transfer of that player's prior rushing role to surviving RB/FBs.

## Strict-prior evidence contract

### Rush share

Canonical PlayerForm history computes:

`rush_share_game = player rushes / team rushes`

from normalized weekly logs. The current-role runner republishes only prior season plus current-season weeks strictly `< target week`; same-week/future current-season rows are rejected.

For V1, the unavailable player's vacated-share estimate is the **most recent same-team strict-prior `rush_share_game`**. No multi-game blend coefficient is searched. If no same-team prior rush-share observation exists, the vacancy event is not eligible for the first candidate and must be disclosed as `NO_PRIOR_RUSH_SHARE` rather than imputed from target outcomes.

### Snap participation

The repository's canonical snap source exposes `offense_pct` and `offense_snaps`, keyed by season/week/team/player. Existing production entitlement code constructs strict-prior features by filtering source rows to ordinal `< target ordinal`; this is the pattern V1 must reuse.

For V1 successor weighting, use the **most recent same-team strict-prior `offense_pct`** for each surviving RB/FB. This is an entitlement/participation weight only; it is not a target-game snap input.

If a surviving successor has no prior same-team snap row, assign no snap-derived weight in V1. Do not manufacture a coefficient from outcomes. If all eligible successors lack usable prior same-team `offense_pct`, the event is not candidate-eligible and must be disclosed as `NO_SUCCESSOR_PRIOR_SNAP_WEIGHT`.

## Deterministic V1 transfer formula

For team-game `g`:

1. Let `U_g` be definitive-unavailable RB/FB teammates with usable most-recent same-team strict-prior rush share.
2. Let `A_g` be surviving production-eligible RB/FB teammates with usable most-recent same-team strict-prior offense snap percentage.
3. Vacated share:

   `V_g = sum(prior_rush_share_i for i in U_g)`

4. Successor weight for `j in A_g`:

   `w_j = prior_offense_pct_j / sum(prior_offense_pct_k for k in A_g)`

5. Candidate transferred share:

   `transfer_j = V_g * w_j`

6. Candidate RB rushing-share state is the production/strict-prior successor rushing state plus `transfer_j` only at the research candidate seam. V1 changes opportunity/carries only. It does not change YPC or any efficiency variable.

No snap/rush blend parameter exists in V1. No depth multiplier, injury-status weight, hand-tuned successor bonus, threshold search, or sportsbook input is permitted.

## Conservation and audit invariants

Before any outcome is attached, every candidate event must satisfy:

- unavailable identities and active successor identities are disjoint;
- every unavailable identity is `definitive_unavailable == 1`;
- every successor is production-eligible RB/FB;
- every rush-share source row is strictly before target ordinal;
- every snap source row is strictly before target ordinal;
- `V_g >= 0`;
- all `w_j >= 0` and successor weights sum to 1 within numerical tolerance;
- successor transfers sum to `V_g` within numerical tolerance;
- no target-game carries/yards/snaps are present in the no-outcome table;
- no sportsbook line/odds fields are present;
- unresolved/ambiguous identity joins fail closed rather than fuzzy-match silently.

The research artifact must separately disclose excluded vacancy events by reason, including at minimum missing unavailable-player prior rush share, missing successor prior snap weights, ambiguous identity, and no eligible successor.

## Important denominator note

PlayerForm's canonical `rush_share_game` denominator is **all team rushes**, not RB-only carries. V1 preserves that semantic. The transfer therefore represents vacated team rushing-share mass attributable to the unavailable RB/FB. This avoids silently changing the metric to the RB-only `rb_share` used by older STACK2 research.

## Evaluation contract after no-outcome artifact passes

Only after the no-outcome table and invariants are physically produced may target outcomes be attached once for the legitimate evaluation cohort. Primary metrics remain those frozen in the parent plan:

- RB rushing-attempt MAE;
- RB rushing-yard MAE with efficiency frozen;
- bias;
- conservation;
- predeclared vacancy-event/high-volume subsets.

A failed result does not authorize coefficient search or a second formula against the same exposed outcomes.

## Immediate implementation step

Build a research-only vacancy-state constructor that materializes the above fields and exclusions without target-game outcomes. Validate its strict-prior and conservation invariants first. Do not grade in the same step.