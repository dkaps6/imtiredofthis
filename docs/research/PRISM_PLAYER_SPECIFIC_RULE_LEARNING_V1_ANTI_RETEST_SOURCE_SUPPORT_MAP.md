# PRISM Player-Specific Rule Learning V1 — Anti-Retest / Source-Support Map

Date: 2026-10-18

Status: **ARCHITECTURE AUDIT COMPLETE / FIRST SOURCE-ONLY PERSONALIZATION LANE IDENTIFIED / NO PREDICTIVE FIT / NO PRODUCTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Canonical principle:

> **The player should have the rules. The position should not have the player.**

Intended hierarchy:

`league prior -> role/archetype prior -> individual player evidence -> current player state`

This document does **not** authorize a free-standing model per player. It maps which
existing PRISM mechanisms can legally support player-specific rule learning and
which are closed or source-blocked.

## 1. Live reconciliation before this audit

Reconciled before writing:

- `main = c9c389d6375cee1f1a88a49e06733dd68c6a2db3`;
- research branch / PR #672 head before this audit:
  `cfbf688447b4747230d237fb20f2d2e213775d63`;
- PR #672 remains OPEN / DRAFT / mergeable=false;
- exact-head PR-triggered workflow runs at `cfbf688...`: none returned;
- exact-head combined statuses at `cfbf688...`: none returned;
- exact-head PR-triggered workflow runs at `c9c389d...`: none returned;
- exact-head combined statuses at `c9c389d...`: none returned;
- newest Issue #535 material comment after directive `6245756869` is only
  continuity comment `6245793110`; no newer science supersedes the directive.

No paid OddsAPI was used.

## 2. What personalization means in V1

A named player may eventually earn one or more of:

1. **mechanism eligibility** — whether a qualified mechanism is allowed to alter
   this player's projection in this market;
2. **mechanism authority** — how much authority a qualified mechanism receives
   relative to a broader prior;
3. **feature sensitivity** — a player deviation from a role/archetype sensitivity,
   only after stronger support than simple eligibility;
4. **reliability state** — how much confidence is placed in the resulting player
   projection.

The first architecture should prefer **mechanism eligibility / authority** over
unrestricted player-specific coefficients.

Why: PRISM already has mechanisms that are population-qualified but heterogeneous
at named-player grain. That creates a clean question:

> Has this named player accumulated enough strictly-prior OOS evidence to earn
> deviation from the role/archetype prior for this exact mechanism and market?

That question is materially different from retuning the mechanism itself.

## 3. Anti-retest status legend

- **ELIGIBLE PARENT** — qualified mechanism with repeated named-player OOS rows.
  May support a new source-only personalization audit.
- **DEFER / PROTECTED** — scientifically useful, but current prospective or source
  contract makes it a poor first personalization target.
- **SOURCE BLOCKED** — do not score or lower gates.
- **CLOSED** — do not repackage under player-specific language.
- **CONTROL / FROZEN PATH** — preserve; not an open first-pass personalization lane.

## 4. Market/mechanism map

| Market / layer | Existing PRISM authority | Current status | Player-specific mechanism eligibility? | Player-specific sensitivity/weight? | Hierarchical shrinkage feasible? | Strictly-prior rolling support? | Named-player OOS protection? | V1 disposition |
|---|---|---|---|---|---|---|---|---|
| All positions — any offensive participation | Active-Role Participation V1 | QUALIFIED ALL | **Possible but low priority**; V1 already helps 82%–89% of evaluable named players depending on position | **No** for the exact V1 prior strength/window; anti-retest forbids retuning | Yes conceptually | Yes, same-player prior history exists | Yes; equal-player + player/game cluster already passed | **ELIGIBLE PARENT / PRESERVE UNIVERSAL SIGNAL FIRST** |
| RB rushing opportunity | Stage-2 Active-Role -> Conditional Allocation | QUALIFIED | **YES — strongest legal first family** | Later only; do not start with free weights | **YES** | **YES**, if prior OOS row-level mechanism deltas can be reconstructed | **YES**; 2025 had 154 players and 86.78% evaluable-player improvement | **ELIGIBLE PARENT** |
| RB receiving opportunity | Stage-2 Active-Role -> Conditional Allocation | QUALIFIED | **YES — highest heterogeneity priority** | Later only | **YES** | **YES**, subject to row-level source audit | **YES**; 2025 had 154 players but only 47.11% improved | **ELIGIBLE PARENT / PRIORITY 1** |
| TE receiving opportunity | Stage-2 Active-Role -> Conditional Allocation | QUALIFIED | **YES — high heterogeneity priority** | Later only | **YES** | **YES**, subject to row-level source audit | **YES**; 2025 had 141 players and 57.89% improved | **ELIGIBLE PARENT / PRIORITY 1B** |
| QB passing opportunity | Stage-2 Active-Role -> Conditional Allocation | CLOSED | **NO** | **NO** | N/A | N/A | Failed 2025 player/equal-player protections | **CLOSED — DO NOT PERSONALIZE THIS BRIDGE** |
| WR receiving Stage-2 | Stage-2 source audit | 2025 OOS parent missing | **NO today** | **NO today** | Potential later | Not legally established | Not available under required two-season parent | **SOURCE BLOCKED / DEFER** |
| WR/TE target allocation | Player Target-Share Trajectory | HISTORICAL OOS SUPPORTED; protected prospective lineage | **Potential later** | Do not refit trajectory coefficient/window | Yes | Partial: strong 2024 OOS, TE secondary 2025; WR dual-season authority not clean enough for first V1 | Player-cluster bootstrap exists; prospective locks protected | **DEFER / PROTECTED** |
| RB receiving allocation | RB receiving-room share | confirmed retrospective + protected forward lock | Potential later | No duplicate proxy/retune | Potential | Existing forward lineage must remain untouched | Player-level result exists but current prospective lineage is protected | **DEFER / PROTECTED** |
| RB rushing allocation | RB carry/snap player-state work | protected | Potential later | No duplicate allocator | Potential | Existing lock/lineage protected | Preserve | **DEFER / PROTECTED** |
| WR receiving reliability | Opportunity-difficulty trust | QUALIFIED | Already player-specific trust; not a mean eligibility rescue | No mean coefficient authorized | Existing player prior already functions as shrinkage/trust context | Yes | Strong player-cluster confirmation | **PRESERVE AS RELIABILITY LAYER** |
| WR receiving reliability | Efficiency-difficulty trust | QUALIFIED | Same | No mean coefficient authorized | Same | Yes | Strong player-cluster confirmation | **PRESERVE AS RELIABILITY LAYER** |
| WR receiving reliability | 50/50 dual trust | HISTORICALLY SUPPORTED; Weeks 7–10 prospective required | Do not alter prospective protocol | No | Existing fixed integration only | Prospective chronology protected | Weeks 7–10 fixed | **PROTECTED** |
| TE receiving reliability | Allocation-difficulty trust | QUALIFIED | Already player-specific trust | No mean coefficient authorized | Yes as trust context | Yes | 2025 player-cluster confirmation | **PRESERVE AS RELIABILITY LAYER** |
| TE efficiency reliability | TE efficiency trust V1 | CLOSED | **NO** | **NO** | N/A | N/A | Independent CI crossed zero | **CLOSED** |
| Generic reliability | mechanism-instability trust | CLOSED | **NO** | **NO** | N/A | N/A | Failed frozen confirmation | **CLOSED** |
| Generic efficiency volatility | efficiency-tail stability | CLOSED | **NO** | **NO** | N/A | N/A | Failed replication | **CLOSED** |
| RB rushing efficiency | NGS player efficiency identity V1 | SOURCE BLOCKED | **NO today** | **NO today** | Potential later only with new support contract | Frozen support failed in 2024 and 2025 | Not scored | **SOURCE BLOCKED** |
| RB receiving efficiency | YPR/YPT/YAC/xYAC/YACOE transforms | CLOSED | **NO** | **NO** | N/A | N/A | Closed family | **CLOSED** |
| TE receiving efficiency | target-quality efficiency V1 | CLOSED | **NO** | **NO** | N/A | N/A | Closed family | **CLOSED** |
| Player high-value conversion | same-player inside-10 / red-zone conversion | CLOSED ALL | **NO** | **NO** | N/A | N/A | Named-player candidate worsened 2024/2025 | **CLOSED** |
| RB short-yardage carry type | same-player 3rd/4th-and-1/2 history | CLOSED | **NO** | **NO** | N/A | N/A | 10/11 gates passed but 2025 conditional-game player-cluster CI crossed zero | **CLOSED — BINDING PLAYER-STABILITY FAILURE** |
| High-value opportunity quality | reconciled PBP source | SOURCE READY as descriptive/source layer | Not by itself; direct persistence/creation mechanisms closed | No direct player coefficient authorized | Could support future distinct topology mechanism | Source is strictly-prior ready | Would require new frozen mechanism | **SOURCE READY, PREDICTIVE FAMILY NOT OPEN** |
| Team high-value creation | direct team persistence / drive creation | CLOSED | **NO** | **NO** | N/A | N/A | Failed | **CLOSED** |
| Teammate competition | Room Competition State V1 | SOURCE BLOCKED | **NO today** | **NO today** | Potential later only with materially new source | Frozen inactive-room coverage failed | Not scored | **SOURCE BLOCKED** |
| RB/TE current role | discrete depth-role transition V1 | SOURCE BLOCKED | **NO today** | **NO today** | Potential later only with new source parity | 2025 / 2026 explicit week-tag semantics failed | Not scored | **SOURCE BLOCKED** |
| Teammate-availability elasticity | Player Role Elasticity V1 | SOURCE BLOCKED | **NO today** | **NO today** | Potential later only with new legal source | Frozen history coverage failed | Not scored | **SOURCE BLOCKED** |
| Generic opponent matchup | DvP / simple usage × weakness / FMT V1 | CLOSED | **NO** | **NO** | N/A | N/A | Replication failed / collateral harm | **CLOSED** |
| Rich player-conditioned opponent | route/alignment/front/box/scoring-topology match | source parity not established | **NO today** | **NO today** | Potential if genuinely new source arrives | Not yet | Not yet | **SOURCE BLOCKED / FUTURE SOURCE FRONTIER** |
| QB mean projection | M89/M90 promoted path | broad mean hunt frozen | Not first-pass | No generic retune | Existing architecture only | Existing validated lineage | Existing QB gates | **CONTROL / FROZEN PATH** |

## 5. Why Stage-2 mechanism eligibility is the clean first architecture

The strongest first lane is not a new football feature.

It is a **meta-architecture over an already-qualified football mechanism**.

Stage-2 is population-qualified for:
- RB rushing;
- RB receiving;
- TE receiving.

But player-level benefit is heterogeneous:

### 2025 independent confirmation

RB rushing:
- 1,419 player-games / 154 players;
- equal-player CI lower bound +0.277780;
- **86.78%** of evaluable players improved.

RB receiving:
- 1,419 player-games / 154 players;
- equal-player CI lower bound +0.015982;
- only **47.11%** of evaluable players improved.

TE receiving:
- 1,305 player-games / 141 players;
- equal-player CI lower bound +0.008875;
- **57.89%** of evaluable players improved.

This is the exact structural seam the new directive requires.

The pooled / equal-player qualification says the mechanism contains real value.
The named-player improvement shares say the value is not distributed uniformly.

Therefore a new question is legal and materially distinct:

> Using only **prior OOS performance of the already-frozen Stage-2 mechanism**,
> can PRISM determine whether this named player has earned permission for Stage-2
> to receive more authority than his role/archetype prior in this exact market?

This does **not** change Stage-2's formula.

It changes only whether/how strongly a named player is allowed to deviate from
the broader prior based on prior evidence.

## 6. Proposed hierarchy for mechanism eligibility

V1 should not start with arbitrary player coefficients.

Candidate hierarchy:

1. **market prior**
   - historical OOS benefit distribution for the exact qualified market/cell;
2. **role/archetype prior**
   - only from source fields already present before the target game;
   - preferred first candidate: within-room canonical entitlement rank/state,
     if exact historical row-level semantics are present;
   - do not invent a new fitted archetype from target outcomes;
3. **player evidence**
   - same named player's prior OOS baseline-vs-mechanism score deltas;
4. **current player state**
   - current pregame Stage-1 active-role probability + canonical entitlement;
   - no target outcome;
   - no sportsbook.

The amount of player deviation must increase only with support/stability.

## 7. Support tiers — architecture only, not yet a scored rule

The source audit should first report how many target rows can support:

- **Tier 0:** no prior player mechanism observations -> market/role prior only;
- **Tier 1:** 1–2 prior player mechanism observations -> no individual deviation;
- **Tier 2:** 3–5 prior observations -> weak/shrunk player evidence only;
- **Tier 3:** 6–9 prior observations -> moderate player evidence;
- **Tier 4:** 10+ prior observations -> strongest eligible player evidence.

These are **support-reporting bins**, not activation thresholds.

No threshold becomes predictive merely because it is listed here.

A later mechanism freeze must predeclare any actual mapping before scoring.

## 8. Source-only requirements before any personalization score

For each Stage-2 qualified cell, prove that exact historical row-level authority can expose:

- stable player identity;
- season / week / game / team / opponent;
- market/cell;
- canonical pregame entitlement;
- Stage-1 active-role probability;
- baseline Stage-2 score contribution;
- candidate Stage-2 score contribution;
- realized target required only to compute the already-completed OOS score delta;
- target row itself excluded from its own player history;
- chronology violations = 0.

The source audit may summarize prior OOS mechanism-benefit history.

It may **not**:
- decide an activation rule;
- choose a shrinkage strength;
- choose a benefit threshold;
- fit per-player coefficients;
- score 2026;
- use Week-6 outcomes;
- use sportsbook/OddsAPI;
- mutate production.

## 9. Evaluation contract for any later scored V1

No later candidate may qualify on pooled MAE/CRPS alone.

Required protections should include:

- equal-player mean improvement;
- player-cluster CI;
- game/team cluster CI where relevant;
- named-player improvement share;
- median named-player change;
- bottom-decile / worst-tail named-player harm;
- support-conditioned results;
- stability across 2024 development / 2025 independent confirmation where legal;
- conservation / finite-opportunity integrity;
- current-state consistency;
- aggregate market/position score only as a guardrail.

A star player or high-volume room may not hide broad player-level harm.

## 10. Explicit anti-rescue rules

Do **not** use this architecture to reopen:

- QB Stage-2;
- WR Stage-2 without a legitimate 2025 OOS parent;
- player high-value conversion identity;
- RB short-yardage carry-type identity;
- raw recent-share replacement;
- Room Competition State;
- RB/TE depth-role transition;
- teammate-availability role elasticity;
- generic DvP/FMT matchup transmission;
- RB run-location matchup;
- WR target-quality chain;
- RB receiving efficiency transforms;
- TE target-quality efficiency;
- mechanism-instability trust;
- generic efficiency volatility;
- NGS RB V1 by lowering support gates.

A closed mechanism is not revived because a new wrapper says “player-specific.”

## 11. First authorized next action

Freeze and execute a **source-only Stage-2 Player Mechanism Benefit History audit**.

The source audit should answer:

> For each qualified Stage-2 cell, do we have enough strictly-prior same-player
> OOS mechanism-benefit history to support hierarchical player-specific
> mechanism eligibility in a later, separately frozen predictive experiment?

No personalization formula is scored in that audit.

## 12. Integrity

This architecture audit:
- new predictive candidates scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes read: **0**;
- Week-6 outcomes graded: **0**;
- sportsbook predictors: **0**;
- OddsAPI calls: **0**;
- production mutations: **0**.

Disposition:

`PRISM_PLAYER_SPECIFIC_RULE_LEARNING_V1_ARCHITECTURE_MAPPED__STAGE2_MECHANISM_BENEFIT_SOURCE_AUDIT_NEXT`.
