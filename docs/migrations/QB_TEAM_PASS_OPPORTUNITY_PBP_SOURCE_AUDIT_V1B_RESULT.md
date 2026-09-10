# QB Team Pass Opportunity PBP Source Audit V1B — Result

## Disposition

Both V1B PBP families are source-eligible for a separately preregistered predictive development screen:

- `PENALTY_DRIVE_EXTENSION = SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`
- `FOURTH_DOWN_AGGRESSION = SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`

This is source/provenance evidence only. No target team-pass-opportunity residual, QB attempt residual, passing-yard residual, or sportsbook input was opened in V1B.

## Canonical lineage

- Branch: `research-qb-team-pass-opportunity-pbp-source-audit-v1b`
- Frozen plan: `aaf4331c04df9892868a34a9201db28a778a8968`
- Evaluator commit: `c57f7df42c418b5972b02937caf624f3c5b088c7`
- Tested/workflow head: `aea93deaac6736af8d23c0975747029b7d41ca58`
- Run: `34534419071`
- Job: `103062583287`
- Artifact: `10174823861` (`qb-team-pass-opportunity-pbp-source-audit-v1b`)
- Artifact digest: `sha256:b21a2d588e25a4866882aebb09b239b8c2305d09a8515c37ab5af519f09fbedf`
- Parent A1 result commit: `9b6ef3e087b34d402c6e51a662d203f88898bab7`
- Original V1 source-audit run: `34533408818`

## Why V1B does not overwrite V1

V1 is preserved unchanged. It applied the 95% field-coverage threshold over all PBP rows, where possession-team and down fields are structurally undefined on kickoffs, timeouts, and other non-offensive-play rows.

V1B kept the exact same `95%` threshold but preregistered family-specific football universes before any target residual was opened.

No threshold was lowered after seeing predictive results; V1B contained no predictive results at all.

## Source integrity

Regular-season nflverse/nflfastR PBP rows loaded:

- 2023: `47,399`
- 2024: `47,274`
- 2025: `46,452`

Target M89 identifier rows: `884`

- target outcomes/residuals loaded: `false`
- sportsbook inputs used: `false`
- model fitting used: `false`
- production changed: `false`
- original V1 result preserved: `true`

## PENALTY_DRIVE_EXTENSION

Disposition:

`SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`

Relevant universe:

- regular-season Week 1-18 PBP rows;
- nonblank possession team;
- nonblank defense.

Relevant-universe row counts:

- 2023: `44,877`
- 2024: `44,686`
- 2025: `43,868`

Core source coverage:

- season/week/game/play/team keys: `100%` in all three seasons;
- `first_down_penalty`: `98.0792%` in 2023, `98.0083%` in 2024, `97.9871%` in 2025.

Strict-prior feasibility on exact 884 M89 2024-2025 target rows:

- at least one prior eligible team game: `100%`
- minimum prior team games: `17`
- median prior team games: `33`
- minimum prior offensive PBP rows: `1,311`
- median prior offensive PBP rows: `2,728.5`

Authorized predictive concept remains narrow:

- strict-prior `first_down_penalty` event rate for the offense;
- strict-prior opponent-defense allowed `first_down_penalty` event rate.

Generic accepted/declined penalty rate remains prohibited and is not inferred from the generic penalty flag.

## FOURTH_DOWN_AGGRESSION

Disposition:

`SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`

Relevant universe:

- `down == 4`;
- nonblank possession team and defense;
- populated `play_type`;
- `play_type != no_play`.

Relevant-universe row counts:

- 2023: `4,053`
- 2024: `3,827`
- 2025: `3,797`

Core source coverage is `100%` in every season for:

- season/week/game/play/team keys;
- `down`;
- `play_type`;
- `fourth_down_converted`;
- `fourth_down_failed`.

Strict-prior feasibility on exact 884 M89 target rows:

- at least one prior eligible fourth-down history: `100%`
- minimum prior fourth-down decisions: `89`
- median prior fourth-down decisions: `238`

Authorized later predictive quantities are limited to strict-prior team and opponent-defense summaries derived from:

- go attempts / fourth-down decision opportunities;
- conversions / go attempts;
- pass go attempts / go attempts;
- run go attempts / go attempts.

A go attempt is exactly `fourth_down_converted == 1 OR fourth_down_failed == 1`.

## Scientific meaning

The project now has two source-clean information families that were not previously tested as pregame predictors of the M89-corrected physical team-pass-opportunity residual:

1. penalty-created drive extension;
2. fourth-down aggression/drive extension.

This does **not** establish that either predicts the residual. V1B intentionally did not open that relationship.

## Immediate next authorized step

Freeze D2 before inspecting predictive performance.

D2 may test both families independently under the same fixed temporal/model architecture. A combined model is permitted only if both independent families clear their own preregistered development gates. 2025 target outcomes must remain untouched during D2 candidate selection.
