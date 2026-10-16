# Player Identity / Role-State Architecture V1

Date: 2026-10-15

Status: **ARCHITECTURE NOTE / RESEARCH DIRECTION / NO PRODUCTION CHANGE**

## Core principle

The NFL model is an **individual-player projection system**.

Historical data should not primarily answer:

> What does this position usually do?

It should help answer:

> Who is this player, how has his role behaved, what kinds of opportunities does
> he earn, under what football conditions do those opportunities appear, and
> which parts of that historical identity are active this week?

The player's historical identity is a prior / playbook, not a guarantee of what
will happen today.

Current-week football state decides which parts of that playbook are relevant.

## Player / allocation / obstacle model

A useful conceptual frame is:

- **Player identity** = the athlete's own football ability, tendencies, role history and reliability.
- **Opportunity allocation** = how much of the team's available work reaches that player instead of teammates.
- **Obstacle environment** = the defense, matchup, game state, protection/blocking context and other weekly conditions the player must move through.
- **Outcome** = what the named player actually produces in the market being projected.

The opponent is an obstacle, not an identity. A difficult defense can suppress an elite player; a weak defense can make a modest player's path easier. Neither case changes who the player is.

Teammates matter because opportunity is shared. If five players can receive the same class of opportunity, the model must determine which player is most likely to get it and in which situations. If one player owns the role, that concentration itself is part of the projection.

These are conceptual components, not authorization for an unvalidated multiplicative formula.

## Aggregate metric rule

Position-level MAE/RMSE is produced by individual player outcomes.

Therefore:
- lower positional MAE is a desirable consequence of better named-player projections;
- positional MAE is a QA / calibration summary;
- it is not sufficient to optimize a position average while failing to explain which players and situations drive the misses;
- player-specific mechanisms must be evaluated at player-game grain with per-player heterogeneity whenever support allows.

The scientific question is not merely:
> "How do we make RB/WR/TE/QB average error smaller?"

It is:
> "Why did this named player's projection miss or hit in this football situation, and what strictly-prior information could have told us that before the game?"

## Preserve the existing stack

This is additive.

Do not remove, rewrite, or invalidate:
- promoted QB/RB/WR/TE projection science;
- player baselines;
- target/carry entitlement;
- validated opportunity science;
- validated distribution/uncertainty work;
- matchup work that survived its frozen gates;
- player trust/selectivity findings;
- current full-slate production architecture.

The purpose is to add a higher-resolution player-state layer around the real
model, not start over.

## Why broad historical aggregates are insufficient

Position-level history is useful for:
- priors;
- shrinkage;
- support when player history is sparse;
- sanity checks;
- league/environment context.

But a position aggregate cannot fully describe why a named player succeeds or
fails.

Two RBs can have the same expected carries while having radically different:
- goal-line usage;
- red-zone carry share;
- two-minute role;
- third-down role;
- receiving route participation;
- designed-touch profile;
- early-down / short-yardage usage;
- competition for touches;
- role elasticity when teammates are absent;
- efficiency history under those opportunity types.

Likewise, two WRs can have similar target totals but different:
- route participation;
- target depth;
- red-zone target share;
- first-read / designed-target tendency;
- third-down role;
- motion/alignment/route-family usage where supported;
- role response when teammates are inactive;
- target concentration when game conditions change.

The model must preserve these named-player differences.

## Player Identity Card

For every player, maintain a strictly-prior historical identity state.

### 1. Opportunity identity

Examples, position-appropriate:
- snaps / participation;
- routes;
- targets;
- carries;
- target share;
- carry share;
- route participation;
- touches;
- designed opportunities.

Track:
- level;
- trajectory;
- stability;
- recency;
- role changes.

### 2. Situational opportunity identity

Where source-supported, track the player's own share / participation in:
- red zone;
- goal-to-go;
- inside the 10;
- inside the 5;
- third down;
- two-minute;
- short yardage;
- early downs;
- late-game / hurry-up;
- pass-game vs rush-game packages.

These variables are **descriptive football identity** unless a separately
frozen experiment proves direct predictive transmission.

They must not automatically become universal mean multipliers.

### 3. Efficiency identity

Track player-specific efficiency appropriate to the market:
- YPC;
- YPT;
- catch rate;
- yards after catch / contact where supported;
- explosive-play profile;
- success under opportunity types;
- difficulty / reliability of projecting that efficiency.

History informs the player's tendencies and uncertainty. It does not force the
same historical efficiency into the current game when role/context changed.

### 4. Role elasticity

Track how the named player's usage changes when:
- a teammate is absent;
- a teammate returns;
- the depth chart changes;
- the team trails or leads;
- the team changes personnel/formation tendencies;
- coaching or quarterback context changes.

This is one of the most important distinctions between player history and
position history.

### 5. Competition map

A player's opportunity is not independent of teammates.

Maintain:
- who competes for the same carries/targets/routes;
- how opportunity has been split;
- whether the player owns high-value touches even when total volume is shared;
- how the room reallocates opportunities when personnel changes.

## Current-Season / Current-Week State

Historical identity must be reconciled with the present.

Before projecting a player, resolve:
- current starter / availability state;
- teammate availability;
- recent usage trajectory;
- current-season role;
- recent situational usage;
- current team offensive behavior;
- current opponent context using only source-qualified / validated information;
- any material role change that makes older history less representative.

The current state decides how much historical identity should be trusted.

## Opportunity Quality / Topology

Opportunity is not one scalar.

Examples:
- 15 RB carries can be low-value if all are between the 20s;
- 10 carries can have major touchdown value if several occur near the goal line;
- 6 WR targets can differ materially by target depth, red-zone use, first-read
  status or route context;
- 4 TE targets can be highly valuable if concentrated in scoring / conversion
  situations.

Therefore future player research should distinguish:
- **opportunity quantity**
from
- **opportunity quality / location / purpose**.

This distinction can matter differently by market.

## Market-specific projection logic

A single player state should feed different markets differently.

Example RB logic:

### Carries
Primarily driven by:
- current role;
- share;
- game-state exposure;
- teammate competition;
- recent trajectory.

### Rushing yards
Carries plus:
- player efficiency identity;
- run environment;
- blocking/context;
- opponent transmission only where valid;
- uncertainty/trust.

### Anytime TD
Should eventually depend heavily on:
- red-zone / goal-line role;
- team scoring-opportunity likelihood;
- competition for high-value touches;
- receiving TD role where applicable;
- current role/availability.

A player may therefore be:
- a weak rushing-yards play;
- but a strong touchdown candidate.

That is not contradictory. The markets ask different football questions.

## Dobbins example / lesson from SEA-DEN

The manual pregame ATD reasoning illustrated the architecture:

1. identify the named player's current role;
2. inspect recent workload / role trajectory;
3. inspect high-value scoring-area opportunity;
4. understand teammate competition;
5. estimate whether the team can generate scoring situations;
6. choose the market that best matches the role.

The touchdown thesis did **not** require a strong rushing-yardage game.

This is the kind of football distinction the model should eventually be able to
represent explicitly.

This single game is not scientific validation and must not be used to fit a
rule.

## Historical data: correct use

History should help build:
- player identity;
- role ranges;
- situational playbook;
- trajectory priors;
- role-change response;
- efficiency tendencies;
- reliability / uncertainty;
- teammate-allocation behavior.

History should **not** be treated as:
- proof the same usage repeats today;
- a reason to ignore current-season/live role evidence;
- a universal position coefficient;
- a substitute for current availability / team context.

## Relationship to previously closed situational-target research

Prior source work established that third-down, red-zone and two-minute target
earning was real/nonredundant player state.

The exact historical experiment
`NO_ACTIONABLE_PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL`
(run `37637017878`) remains CLOSED.

That experiment asked whether the frozen broad situational-target construction
directly explained future opportunity residuals.

Do not rescue or retune that formula.

This architecture uses situational history differently:
- as part of a named player's descriptive role identity;
- potentially as market-specific state;
- potentially as interaction/context input only after a separately frozen,
  genuinely distinct experiment.

A failed direct scalar transmission does not imply that the underlying football
state is irrelevant to player identity.

## Research hierarchy

Future player-first work should ask, in order:

1. **Identity:** what is this player's stable / recurring football role?
2. **Current state:** which parts of that identity are active now?
3. **Opportunity:** how many chances is he likely to get?
4. **Opportunity quality:** what kinds of chances are they?
5. **Efficiency:** what does this player tend to do with those chances under
   relevant conditions?
6. **Context:** how does this week's team/opponent/availability state modify the
   opportunity path?
7. **Reliability:** how much should we trust the resulting projection?
8. **Market mapping:** which prop market actually corresponds to the football
   thesis?

Position averages remain priors/guardrails, not the primary decision grain.

## Immediate research implication

Continue the already-frozen player efficiency-difficulty trust/selectivity lane.

Do not interrupt or rewrite it from this architecture note.

After that result, the next research frontier should be selected from the
player-identity architecture only if it is genuinely distinct from closed work.

Candidate future source/audit families may include:
- high-value opportunity topology;
- role elasticity under teammate availability;
- teammate competition / room reallocation state;
- current-season role trajectory;
- market-specific situational opportunity state.

Every candidate must pass anti-retest review before scoring.

## Scientific boundary

This document is an architecture / research-direction note.

It:
- fits no model;
- changes no projection;
- changes no production code;
- uses no sportsbook data;
- makes no OddsAPI call;
- promotes no feature.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
