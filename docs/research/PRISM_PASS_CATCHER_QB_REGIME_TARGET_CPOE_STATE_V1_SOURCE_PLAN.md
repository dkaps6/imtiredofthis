# PRISM Pass-Catcher × QB-Regime Target-CPOE State V1 — Source Plan

Date: 2026-10-17

Status: **FROZEN SOURCE / ANTI-RETEST CONTRACT BEFORE SOURCE RESULT / RESEARCH ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

## Scientific object

The object is the named WR/TE in his current football situation.

This source audit asks only whether historical data can support a later frozen test of:

> Does the named pass catcher's conversion-quality state with the currently selected QB contain next-game target-CPOE information beyond the receiver's ordinary strictly-prior CPOE state?

This is an individual efficiency / target-conversion-quality regime question. It is not target entitlement, target depth, receiving yards, or a sportsbook-facing market.

## Immutable anti-retest boundary

Treat as closed and do not rescue:

- direct same-QB target-share allocation V1 — run `39067914161`;
- same-QB target-depth signal V1 — run `39087599967`;
- raw QB-receiver pair YPT;
- WR-R18 receiver target-CPOE -> receiving-yard residual;
- WR-R19 catchability and WR-R20 read-priority families;
- QB M70 completion/CPOE decomposition;
- QB M81 THROW_DECISION_QUALITY / receiver-error families;
- generic historical catch-rate/YPT/YPR corrections;
- direct WR/TE depth-YPT;
- generic DvP / generic opponent multipliers.

This V1 is distinct only because the proposed state is the **interaction**:
same named receiver + currently selected QB + strictly-prior target CPOE,
and the first possible outcome is target CPOE itself.

A later receiving-yard translation is not authorized by this source audit.

## Source authorities

Target seasons:
- 2024 development source population;
- 2025 independent-confirmation source population.

Target families:
- WR;
- TE.

Pregame player/QB identity:
- frozen Player Active Role pregame universe from run `38895205962`;
- existing source-qualified QB-regime selection semantics.

Historical target-conversion source:
- free regular-season nflverse/nflreadpy PBP;
- seasons 2023-2025;
- official pass-attempt target event;
- receiver stable ID present;
- passer stable ID present;
- sack excluded;
- two-point attempt excluded;
- native `cpoe` finite for conversion-quality state.

No sportsbook, OddsAPI, 2026 outcome, or production source is permitted.

## Exact pregame state semantics

For each 2024/2025 active WR/TE target player-game:

1. select the current pregame QB using the existing qualified QB-regime rule;
2. collect only the same receiver's same-team target events strictly before the target game;
3. select the latest up to **8 completed target-bearing receiver-games first**;
4. only after the eight-game window is fixed, retain target events with finite native `cpoe`;
5. ordinary receiver-CPOE readiness requires:
   - at least **4 prior target-bearing receiver-games** inside the selected history;
   - at least **16 finite-CPOE target events** inside those games;
6. same-QB history is the subset of those exact prior-eight finite-CPOE target events thrown by the selected current QB;
7. same-QB readiness requires:
   - at least **2 distinct prior receiver-games** with that QB;
   - at least **8 finite-CPOE target events** from that QB.

Persist, but do not score:
- prior-eight mean CPOE;
- same-QB prior mean CPOE;
- ordinary/same-QB game and target counts;
- selected QB source and identity;
- maximum source season/week.

No target-game event may enter any pregame state field.

## Frozen source gates

Provider quality, every target season:
- PBP target receiver-ID coverage >=99%;
- PBP target passer-ID coverage >=99%;
- PBP native-CPOE coverage >=98%.

Every family-season cell:
- stable roster identity rate >=99%;
- selected QB-regime source-valid team-week rate >=90%;
- chronology violations = 0;
- exact-mechanism-ready support:
  - WR: >=800 player-game state rows and >=120 distinct players;
  - TE: >=400 player-game state rows and >=60 distinct players.

Exact-mechanism-ready means all frozen ordinary-history and same-QB-history gates pass.

All four cells ready:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_STATE_V1_SOURCE_READY_ALL`.

Some but not all:
`...SOURCE_READY_PARTIAL`.

None:
`...SOURCE_BLOCKED`.

No gate weakening is permitted after result exposure.

## What a source pass would authorize

Only a separately frozen direct target-CPOE scoring test:

- baseline = named player's ordinary strictly-prior CPOE state;
- candidate = named player's strictly-prior CPOE state with the currently selected QB;
- target entitlement and target depth unchanged;
- first outcome = next-game named-player target CPOE.

It would not authorize receiving-yard, reception, touchdown or betting-market translation.

## Anti-rescue

After source exposure do not:
- lower 4 prior target-bearing games;
- lower 16 ordinary finite-CPOE targets;
- lower 2 same-QB games;
- lower 8 same-QB finite-CPOE targets;
- change the eight-game window;
- backfill an older ninth game because a recent game lacks CPOE;
- pool WR+TE or 2024+2025 to rescue support;
- substitute completion rate, raw YPT or target depth;
- fit a shrink coefficient;
- use QB-change/WR1/TE1 carveouts;
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
