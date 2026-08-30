# Migration 85 — True Blocker × True Rusher Source / Feasibility Audit

## Status

`PREREGISTERED / SOURCE AUDIT ONLY`

M85 does not fit a QB model. It asks whether exact offensive blocker × defensive pass-rusher assignment information can be obtained historically and in-season with enough fidelity to support a later pregame QB predictive test.

## Why M85 exists

M82 left `TRUE_BLOCKER_X_TRUE_RUSHER_ASSIGNMENT` as a genuinely new source-blocked information frontier. M84 then found the top-weapon matchup idea source-blocked under the current free-source contract.

The football hypothesis is that aggregate pressure strength can hide a specific pass-protection mismatch: a particular tackle/guard/center or protection structure may be repeatedly exposed to a particular edge/interior rusher, materially altering target-game pressure and therefore QB attempts/YPA.

## Prior-work anti-retest boundary

M85 may not relabel as new:

- team pressure rate;
- sacks/hurries/hits;
- generic offensive-line continuity;
- aggregate pass-rush win rate;
- generic number of pass rushers;
- M56 static pressure/pass-defense context;
- M69 pressure/game-script context;
- M80/M81 FTN `n_blitzers`/pressure-response families.

A qualifying source must identify **who blocks whom** or an equivalent exact individual assignment/exposure mechanism.

## Candidate source families

1. NFL Next Gen Stats pressure probability / blocking matchup system.
2. NFL Big Data Bowl / PFF blocker-rusher scouting slices (`blockedPlayerNFLId*`, pressure allowed as blocker).
3. nflverse participation / PBP defensive-player and pass-rusher fields.
4. public PFR / ESPN / other pass-rush and OL advanced tables as potential aggregate-only fallbacks.

## Frozen qualification contract

A source may advance only if it provides:

1. exact blocker-rusher identity or equivalent assignment exposure;
2. materially new information beyond aggregate pressure/OL features;
3. sufficient multi-season historical coverage for leakage-safe development;
4. in-season updates usable before future games;
5. stable machine-readable acquisition;
6. player IDs resolvable to roster/QB-game context;
7. free access for this research phase;
8. no target-game postgame information in a pregame feature.

A scientifically ideal proprietary source is labeled `IDEAL_BUT_PROPRIETARY` and does not advance.

## Allowed outputs

- source inventory;
- endpoint/status checks;
- field/granularity audit;
- history/live/deployability matrix;
- novelty crosswalk;
- final source disposition.

## Prohibited actions

- no QB outcome fitting;
- no sportsbook variables;
- no model-zoo rescue with aggregate pressure features;
- no use of a limited competition slice as if it were a complete 2023-2026 feed.

## Frozen dispositions

- `QUALIFIED_FOR_M86_PREDICTIVE_DEVELOPMENT`
- `HOLD_SOURCE_BLOCKED_NEW_INFORMATION`
- `CLOSED_NO_MATERIALLY_NEW_SOURCE`

If no source qualifies, M86 predictive development is not opened.

`production_actionable = false`
