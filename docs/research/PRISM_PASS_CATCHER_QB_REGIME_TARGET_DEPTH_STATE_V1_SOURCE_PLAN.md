# PRISM Pass-Catcher × QB-Regime Target-Depth State V1 — Source Plan

Date: 2026-10-17

Status: **FROZEN SOURCE / ANTI-RETEST CONTRACT BEFORE SOURCE RESULT / RESEARCH ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

## Scientific object

The object is the named WR/TE in his current football situation.

This source audit asks only whether historical data can support a later frozen test of:

> Does the identity of the currently selected QB change the **type/depth of the named pass catcher's targets**, even when the player's target entitlement itself is left unchanged?

This is an opportunity-type / target-quality state question. It is not a target-share adjustment and it is not a receiving-yards model.

## Immutable parents / anti-retest

Treat as immutable:

- `PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_CLOSED`, run `39067914161`.
  Direct same-QB target-share adjustment is closed and may not be rescued.
- raw QB-receiver pair YPT is closed.
- C1 target-mass and C3 broad QB/receiver combinations failed player-level protection.
- WR R7 persistent signed air-yard/YAC/explosive level-trait work is closed.
- `PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED` is a distinct uncertainty result:
  prior target-depth dispersion predicts absolute efficiency difficulty; it did not authorize a signed target-depth or YPT shift.
- the universal symmetric target-depth distribution transform is unpromoted/closed.
- `PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED` is a WR efficiency/matchup mechanism and does not answer QB-conditioned opportunity type.
- generic defense-vs-position and generic matchup multipliers remain closed.
- scoring-area recent-share and team-side opportunity-quality persistence families remain closed.

Therefore this V1 may inspect only a genuinely different quantity:
**strictly-prior air-yard target depth for the named player under the currently selected QB**.

## Source authorities

Target seasons:
- 2024 development source population;
- 2025 independent-confirmation source population.

Target families:
- WR;
- TE.

Pregame active-player / QB-regime identity parent:
- frozen Player Active Role source universe from run `38895205962`;
- exact same QB-regime selection semantics already source-qualified in
  `PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1`.

Historical target-type source:
- free regular-season nflverse/nflreadpy PBP;
- seasons 2023-2025;
- official pass-attempt target event;
- receiver stable ID present;
- passer stable ID present;
- sack excluded;
- two-point attempt excluded;
- `air_yards` finite for target-depth state.

No sportsbook or OddsAPI source is permitted.

## Exact pregame state semantics

For each 2024/2025 active WR/TE target player-game:

1. select the current pregame QB with the already-qualified QB-regime rule;
2. collect only the same named receiver's target events for the same team strictly before the target game;
3. retain the latest up to **8 completed receiver-games** with at least one finite-air-yard target, matching the existing target-depth history semantics;
4. baseline-history readiness requires:
   - at least **4** prior receiver-games;
   - at least **10** finite-air-yard targets;
5. same-QB history is the subset of those exact prior-eight target events thrown by the selected current QB;
6. same-QB readiness requires:
   - at least **2 distinct prior receiver-games** with that QB;
   - at least **4 finite-air-yard targets** from that QB.

Persist, but do not score:
- prior-eight mean air yards per target;
- prior-eight target-depth SD;
- prior-eight deep-target rate (air yards >=20);
- same-QB prior mean air yards per target;
- same-QB target-depth SD;
- same-QB deep-target rate;
- prior game/target counts;
- selected QB source and identity;
- maximum source season/week used.

No target-game event may enter any pregame state field.

## Frozen source gates

For every family-season cell (WR 2024, WR 2025, TE 2024, TE 2025):

- stable roster identity rate >= 99%;
- selected QB-regime source-valid team-week rate >= 90%;
- PBP target receiver-ID coverage >= 99%;
- PBP target passer-ID coverage >= 99%;
- PBP target air-yards coverage >= 98%;
- chronology violations = 0;
- exact-mechanism-ready support:
  - WR: >=1000 player-game state rows and >=150 distinct players;
  - TE: >=600 player-game state rows and >=80 distinct players.

Exact-mechanism-ready means the frozen baseline-history and same-QB-history gates above both pass.

All four cells ready:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_READY_ALL`.

Some but not all ready:
`...SOURCE_READY_PARTIAL`.

None ready:
`...SOURCE_BLOCKED`.

There is no gate weakening after result exposure.

## What a source pass would authorize

Only if the relevant cells pass, a **separate scoring freeze** may compare:

- baseline: named player's strictly-prior ordinary target-depth state;
- candidate: named player's strictly-prior target-depth state with the currently selected QB.

Target share, projected targets, room conservation, team pass volume and all existing entitlement logic must remain exactly unchanged.

The first score target must be target depth itself. Receiving yards, receptions, touchdowns and betting markets are explicitly out of scope until separately frozen later.

## Anti-rescue

After source exposure do not:
- lower 4 prior receiver-games;
- lower 10 prior finite-air targets;
- lower 2 same-QB games;
- lower 4 same-QB finite-air targets;
- change eight-game history window;
- pool WR+TE to rescue a family;
- pool 2024+2025 to rescue a season;
- substitute raw pair YPT;
- translate immediately to yards/receptions/TDs;
- fit shrink coefficients;
- add opponent or sportsbook variables;
- use 2026 outcomes.

## Integrity boundary

At source freeze:
- predictive candidates scored: 0;
- fitted coefficients: 0;
- threshold searches: 0;
- target-game outcomes used in predictor: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
