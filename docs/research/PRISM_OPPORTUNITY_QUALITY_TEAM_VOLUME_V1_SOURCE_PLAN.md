# PRISM Opportunity Quality — Team High-Value Volume V1 Source / Anti-Retest Plan

Frozen: 2026-10-16

Status: **SOURCE / ANTI-RETEST AUDIT ONLY — NO PREDICTIVE CANDIDATE — NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Architecture position:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`

Scientific parents:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`;
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`;
- `PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`;
- `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`.

## 1. Why this is a distinct question

The closed high-value persistence V1 asked:

> Does a named player's recent scoring-area **share** directly forecast his next-game scoring-area share better than his ordinary opportunity share?

That exact claim is terminal. Do not retest it with:
- different yardline bands;
- different windows;
- lead-back / WR1 / TE1 subsets;
- TD outcomes;
- coefficient tuning.

This source plan asks a different upstream football question:

> Can PRISM reconstruct, strictly before a target game, **how much high-value opportunity the team/game is likely to create as a finite pool**, separately from the already-existing question of which named player receives the pool?

Conceptually:

`TEAM HIGH-VALUE OPPORTUNITY VOLUME -> PLAYER HIGH-VALUE ENTITLEMENT -> MARKET-SPECIFIC OUTCOME`

This separation matters because a player can own a valuable role but have little touchdown/scoring opportunity if his offense does not reach scoring territory, and vice versa.

V1 source work does **not** assert that any particular volume state predicts the next game.

## 2. Historical authority

Source family:
nflverse / nflfastR play-by-play plus canonical schedule/team identity.

Target seasons:
- 2022;
- 2023;
- 2024;
- 2025.

Regular season only.

Target evaluation window for a later mechanism, if source-ready:
Weeks 5-18, so every candidate input can require strictly-prior same-season team history.

No 2026 outcomes.

No sportsbook information.

## 3. High-value finite pools to reconstruct

### A. RB rushing-quality pool

For each team-game:
- total team rushing attempts;
- team rushing attempts inside opponent 20;
- inside opponent 10;
- inside opponent 5;
- goal-to-go rushes where the underlying play state supports it.

Primary V1 source family for later design:
**inside-10 team rushing opportunities**.

The other bands are descriptive/source diagnostics only. Their presence does not authorize after-the-fact selection.

### B. Receiving-quality pool

For each team-game:
- total team targets;
- red-zone targets with `yardline_100 <= 20`;
- optional end-zone targets where target + air-yards semantics support a pass into the end zone.

Primary V1 source family for later design:
**red-zone team target opportunities**.

End-zone targets are optional source authority only and require the frozen air-yards coverage gate.

### C. Team scoring-territory creation

Where source semantics are deterministic and complete, reconstruct team-game counts such as:
- offensive drives reaching opponent 20;
- drives reaching opponent 10;
- offensive plays in opponent 20 / 10;
- scoring-territory possessions/opportunities.

This layer is team opportunity creation, not player share.

Do not use points scored, touchdowns scored, fantasy points or sportsbook team totals as a source substitute.

## 4. Strictly-prior team state inventory

The audit may establish whether each target team-game can attach strictly-prior same-season history for:
- team offensive high-value pool counts/rates;
- opponent high-value opportunity allowed counts/rates;
- team red-zone / inside-10 opportunity frequency per drive or per offensive play where denominators are source-complete;
- total offensive drive/play opportunity needed only as a denominator.

This is source readiness only.

Do not:
- choose a rolling window based on prediction results;
- fit an offense/defense blend;
- fit a coefficient;
- score target games;
- decide whether offense or defense deserves more weight.

If a later predictive mechanism is authorized, its exact prior window/blend must be frozen separately before scoring.

## 5. Player allocation compatibility audit

The source audit must also prove that each finite high-value team pool can be reconciled to named players without inventing opportunity.

For RB rushing:
- stable rusher identity;
- RB/FB/HB position-family bridge;
- player inside-10 rush counts sum to the explicit RB-family portion of team inside-10 rushes;
- non-RB/QB/other rushing remains a complement rather than being silently reassigned.

For receiving:
- stable receiver identity;
- RB/WR/TE family bridge;
- player red-zone targets sum to the explicit modeled-family portion of team red-zone targets;
- unmodeled/other target opportunity remains complement mass.

This is a topology/conservation audit, not a player-share predictor.

## 6. Frozen source gates

Per season, require:

General:
- canonical scheduled team-game coverage >=99%;
- PBP team/opponent identity coverage >=99%;
- `yardline_100` coverage on rush/target opportunity rows >=99%;
- duplicate contradictory player-play identities = 0;
- chronology violations = 0;
- target-game outcome used as candidate predictor = 0;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- production mutation = false.

RB rushing topology:
- rusher stable identity >=99%;
- RB/FB/HB position-family bridge >=98%;
- every regular-season week contains inside-10 rushing opportunities at league level;
- finite/non-negative team and player counts =100%;
- player/team complement conservation =100%.

Receiving topology:
- targeted receiver stable identity >=99%;
- RB/WR/TE position-family bridge >=98%;
- every regular-season week contains red-zone targets at league level;
- finite/non-negative team and player counts =100%;
- player/team complement conservation =100%.

Optional end-zone targets:
- target-row air-yards coverage >=95%;
- deterministic end-zone-target construction documented;
- no use of completion/TD outcome to define the target opportunity.

Strictly-prior team-state attachment for Weeks 5-18:
- offensive team prior-history attach >=99%;
- opponent prior-history attach >=99%;
- at least 4 completed strictly-prior same-season team games for every row admitted to a later scored mechanism;
- zero target/future games in any prior aggregate.

Do not lower a failed gate.

## 7. Source dispositions

Possible terminal source results:
- `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`
- `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY_PARTIAL`
- `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_BLOCKED`
- `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_INTEGRITY_FAILURE`

A source-ready label means only that a later experiment can be designed honestly.

It is not predictive proof.

## 8. If source-ready, next mechanism must be separately frozen

A later mechanism freeze must predeclare, before target scoring:
- one exact team high-value volume baseline;
- one exact strictly-prior candidate construction;
- offense/opponent combination law if any;
- development and independent confirmation seasons;
- primary proper score / count metric;
- mean/count protection gates;
- dependence-aware inference;
- finite pool conservation;
- anti-rescue rules.

It may not change the player's ordinary opportunity allocation to rescue the quality model.

It may not resurrect the closed player high-value-share persistence formula.

## 9. Relationship to ATD

Generic ATD remains uncertified.

This lane does not certify ATD.

It would only address one missing upstream component of the required chain:

`football-only team scoring opportunity -> high-value pool -> player entitlement -> calibrated TD outcome probability`.

TD conversion/calibration remains a separate market-specific problem.

## 10. Integrity at freeze

At this source-plan freeze:
- predictive candidates scored: **0**;
- target outcomes read for candidate selection: **0**;
- fitted coefficients: **0**;
- thresholds searched: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- 2026 outcomes: **0**;
- production changed: **false**.
