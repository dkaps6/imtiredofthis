# Football Matchup Transmission V1 — Frozen Audit Plan

Status: **FROZEN BEFORE HISTORICAL SCORING — RESEARCH ONLY — NO PRODUCTION CHANGE**

Branch:
`research-football-matchup-transmission-v1`

Date frozen: 2026-10-06

## Why this exists

The Week-4 postmortem showed that calibration and uncertainty problems are real,
but they do not answer a more basic football question:

> Does the production stack actually translate pregame matchup information into
> player-level projections strongly enough, conditioned on the player's current
> role and usage?

The current production code contains extensive team/player context, but a
read-only transmission audit found several important fields are either not
consumed by the generic RB/WR/TE path or only enter through coarse threshold
rules.

This plan tests the football-information layer itself. It does not fit a
sportsbook selector and does not use 2026 outcomes.

## Phase A — deterministic production transmission audit

No outcomes.

Inventory every pregame football feature available in the canonical production
artifacts and classify it as:

- AVAILABLE_AND_CONSUMED
- AVAILABLE_BUT_DROPPED_BEFORE_TEAM_CONTEXT
- TEAM_CONTEXT_PRESENT_BUT_NOT_USED_BY_RULES
- USED_ONLY_BY_SPECIALIST_POSITION_PATH
- USED_BY_GENERIC_SIMULATION
- SOURCE_BLOCKED / NOT_REPRODUCIBLE

At minimum audit:

### Team/game environment
- plays / pace
- PROE / pass tendency
- success rate
- pass rate faced
- rush/pass EPA
- pressure allowed/generated
- explosive-play rate allowed

### Run-defense matchup
- defensive rush EPA
- rush success allowed when reconstructable
- yards before contact allowed per RB rush
- stuff rate
- light/heavy box rate

### Receiving matchup
- WR/TE/RB YPT allowed
- outside/slot YPT allowed
- man/zone rate
- middle-open/closed rate
- pass EPA/YPA/success allowed
- pressure

### Player role
- current/prior rush share
- target share
- route rate
- snap/participation where available
- WR hierarchy / TE entitlement / RB role
- injury / vacancy state

### Player-specific matchup
- WR-CB / route-responsibility evidence only where a reproducible historical
  and live contract exists. Do not restore retired coverage heuristics.

The audit must trace each field through:
source -> artifact -> TeamContext/PlayerContext -> simulation_rules -> simulation
or specialist -> final projection.

## Phase B — historical residual signal audit

Use only leakage-safe historical football data:
- 2024 Weeks 2-18
- 2025 Weeks 2-18

No 2026 outcomes.
No sportsbook lines or odds.
No candidate coefficient fitting.

Question:

> Conditional on the existing production player/role projection, do strict-prior
> defensive matchup variables explain next-game residual error in the expected
> football direction?

Primary markets:

1. RB rush yards
2. RB rush+receiving yards
3. WR receiving yards
4. TE receiving yards
5. RB receiving yards
6. QB pass yards as a control because M89/M90 already contains richer matchup
   context

Predeclared matchup families:

### RB rushing
- opponent def_rush_epa
- opponent rush stuff rate
- opponent yards-before-contact allowed per RB rush
- opponent box rates

### Receiving by role/position
- opponent position-specific YPT allowed (WR/TE/RB)
- outside/slot YPT allowed where player alignment is available
- opponent pass EPA/YPA/success allowed
- man/zone/middle-open context

### Volume/game environment
- offense pace / plays
- offense PROE or pass tendency
- opponent pass/rush tendency faced where leakage-safe
- pressure mismatch

## Phase C — usage x matchup interaction audit

This is the core football question and is frozen before scoring.

Do not ask only whether a defense is weak. Test whether weakness matters more
when the player has the role to exploit it.

Use continuous, predeclared interactions only:

- RB rush opportunity state × run-defense weakness
- WR target/route opportunity × WR/outside/slot matchup weakness
- TE target/route opportunity × TE/zone/middle matchup weakness
- RB target/route opportunity × RB receiving matchup weakness

Opportunity state must be the existing strict-prior production football state.
Do not use sportsbook lines or realized target-game usage.

No threshold search, top-N search, role carveout search or post-hoc player class
selection.

## Replication rule

A matchup family is considered a genuine missing football signal only if:

1. expected-sign residual association appears in both 2024 and 2025;
2. player/game-clustered uncertainty supports the direction;
3. the signal is incremental to existing production role/opportunity state;
4. it is available live with the same semantics;
5. it does not simply restate a previously closed family.

If it fails replication, close it.

If it passes, the next step is a separately frozen full-stack candidate. A
diagnostic PASS does not authorize production.

## Explicit anti-retest rules

Do not:
- revive Rush Pool Evidence Guard;
- rescue Opportunity Authority Priority V1;
- retune Bayesian constants;
- restore the retired WR coverage_penalty heuristic;
- fit against Week-4 outcomes;
- use sportsbook lines upstream;
- create fantasy-points-allowed shortcuts without opponent/opportunity controls;
- use raw yards allowed as a standalone defense rating;
- create post-hoc “Bijan-type” or bellcow-only thresholds.

## Initial deterministic findings that motivated the audit

The Week-4 recovered artifact contains team matchup fields such as:
- `def_rush_epa`
- `explosive_play_rate_allowed`
- `ypt_allowed_wr/te/rb`
- `ypt_allowed_outside/slot`
- yards-before-contact and stuff-rate fields
- coverage and box rates
- current 2026 success-rate context

But the generic TeamContext contract currently carries only a subset, and the
generic rules layer does not consume several fields that are present.

The generic game-script function also currently produces a fixed 57% pass share
once the rules path is active, rather than using the loaded offense PROE/pass
tendency. QB M89/M90 is a separate richer specialist path and must be audited
separately from RB/WR/TE.

This is a transmission audit, not a conclusion that any one omitted field will
improve accuracy.
