# RB R26L 2026 Week-1 Regime Transportability V1 — Result

Date: 2026-09-09
Branch: `research-rb-r26l-2026-week1-regime-transportability-v1`
Valid head: `a74550df3166aaf6c80c66b40caac6fba2b649fd`
Run: `34389455694`
Job: `102593883866`
Artifact: `10119058769`
Artifact name: `rb-r26l-2026-week1-regime-transportability-v1`
Artifact digest: `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
Disposition: `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`

## Question

R26K showed that football-coherent states associated with R26's harmful 2020 Week-1 allocation error did not replicate as harmful states in 2021-2025. R26L therefore asked a prospective/source-only question before any 2026 game outcomes:

> Does the actual 2026 Week-1 RB vacancy environment resemble anomalous 2020 or the successful 2021-2025 source regime on frozen, source-aligned pregame football state?

The answer under the frozen R26L rules is **modern-like**.

This result does not itself authorize shadow or production. It authorizes only design of a separately frozen R26M-style prospective qualification/synthesis study.

## Canonical evidence and frozen hashes

Immutable historical parent:
- R26J run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`

Immutable current production parent:
- Full Slate run `34317211395`
- artifact `10090547415`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- production authority `main@f8417f55b04ce0e19baf260e9d532765034c47f1`

Valid-run hashes:
- frozen plan: `1d85486290a3a3d6b43657002d9e7aab3b7688d4675189f053a84f54c9c9f31f`
- implementation lock: `b9d532e209e3c9e542128b4938b32bd0c5c09357ebd0952268e5b7509ded9681`
- mechanical repair note: `8ec6f5c5033ae23813562579d889df6a361698996f794248165c19a21c28a658`
- frozen evaluator: `021bba0d7acb8a96d149766ba9ca707354cb6de077fe55d6390c3fe35ae2187f`
- strict-prior compatibility runner: `d2776f2a0ed84e8ddebd48e6d2446630039df5773ae22e20905aad05a50856b0`

The protected production boundary passed before execution.

## Mechanical execution history

Three attempts failed before a scientific disposition and were repaired mechanically without changing frozen science.

### Attempt 1 — run `34378742255`

`nflreadpy==0.1.5` rejected explicit 2026 weekly-roster loading because its generic current-season validation still returned 2025 on 2026-09-09. The isolated R26L workflow repaired only that validation ceiling. The requested source remained the same nflverse weekly roster source, `weekly_rosters/roster_weekly_2026`.

### Attempt 2 — run `34388840174`

After 2026 roster loading succeeded, the strict-prior R9 identity runtime was asked to load player statistics through 2026, causing a 404 for nonexistent `stats_player_week_2026.parquet`. Because R26L targets 2026 Week 1 and no 2026 game has occurred, strictly-prior identity must end at 2025. A hash-tracked compatibility runner therefore changes only the call-time history endpoint to 2025 while leaving the frozen evaluator unchanged.

### Attempt 3 — run `34389135420`

Primary 2026 source construction succeeded, but the frozen Full Slate artifact contained two copies of `roles_ourlads.csv`, under `data/` and `outputs/`. They were verified byte-identical. The valid workflow verifies the immutable production artifact digest first, asserts those role files are identical, and stages only canonical `data/roles_ourlads.csv` plus unique `data/player_form_consensus.csv` into an isolated root for the already-frozen secondary/non-decisive diagnostics.

No failed attempt produced a scientific R26L disposition.

## Source integrity

All frozen integrity gates passed:
- canonical 2026 Week-1 roster teams: `32`;
- canonical 2025 Week-18 prior roster teams: `32`;
- canonical 2026 Week-1 RB rows: `112`;
- canonical 2025 Week-18 RB rows: `126`;
- 2026 Week-1 vacancy rooms: `31`;
- exited-player rows: `63`;
- departed-player prior-history coverage: `0.8888888888888888`;
- membership violations: `0`;
- prior snapshot: exactly 2025 Week 18;
- normalized 2025 statuses: `ACT/INA` only;
- normalized 2026 statuses observed: `ACT` only;
- target-game outcomes used: `0`;
- target-game participation used: `0`;
- sportsbook football inputs: `0`;
- same-week depth used: `false`;
- predictions regenerated: `false`;
- R9 refit: `false`;
- production parameters changed: `false`;
- R22 changed: `false`;
- receiving-yard means changed: `false`.

## 2026 Week-1 source summary

Across the 31 current vacancy rooms:

| Feature | 2026 | 2020 | 2021-25 mean | Frozen preference |
|---|---:|---:|---:|---|
| `current_room_n` | 3.5161 | 4.4643 | 3.9987 | `MODERN_CLOSER` |
| `continuing_n` | 1.9355 | 2.2500 | 2.0026 | `MODERN_CLOSER` |
| `entrants_n` | 1.5806 | 2.2143 | 1.9960 | `MODERN_CLOSER` |
| `veteran_entry_n` | 0.5806 | 0.8571 | 0.5598 | `MODERN_CLOSER` |
| `veteran_entry_share` | 0.17043 | 0.18699 | 0.14242 | `2020_CLOSER` |
| `exit_history_coverage` | 0.88978 | 0.95238 | 0.91188 | `MODERN_CLOSER` |
| `sum_exit_last8_targets_pg` | 2.2995 | 3.5864 | 3.3128 | `MODERN_CLOSER` |

Frozen directional count:
- `MODERN_CLOSER`: `6/7`;
- `2020_CLOSER`: `1/7`;
- ties: `0`;
- beyond 2020 in the 2020-anomalous direction: `0/7`.

## Frozen normalized-distance result

- mean normalized distance to 2020: `1.2220245842254218`;
- mean normalized distance to 2021-2025 modern mean: `0.7255585711888131`;
- modern / 2020 distance ratio: `0.5937348401617526`.

The frozen modern-like threshold required modern distance to be no greater than `0.75 *` 2020 distance. The actual ratio is about `0.594`, so this gate passes with material room.

Important nuance: 2026 is not simply an average 2021-2025 environment. Several transition/load features are below even the later-period range, particularly room size, entrant count, and vacated receiving load. The scientific conclusion is therefore not "2026 is identical to 2021-2025." It is narrower and exactly what was frozen: **2026 is materially closer to the modern source regime than to anomalous 2020, and it does not extend further in the 2020-anomalous direction.**

That pattern is football-coherent with a comparatively lower-transition/lower-vacated-load Week-1 environment rather than the unusually large, veteran-heavy, high-vacated-load 2020 state identified by R26J.

## Secondary current-production diagnostics

The post-merge Full Slate artifact supplied current Ourlads RB-room size and prior target-share concentration diagnostics across all 32 teams.

These diagnostics were explicitly stamped:
`SECONDARY_ONLY_SOURCE_SEMANTICS_DIFFER_FROM_R26J`

They did not enter the seven-feature classifier and did not determine the R26L disposition.

## Scientific interpretation

R26J established that 2020 was structurally distinct. R26K then established that the football states explaining much of 2020's R26 allocation harm generally reversed direction and were beneficial in 2021-2025, preventing a defensible historical router.

R26L now adds the prospective evidence that the actual 2026 Week-1 source environment is **not 2020-like under the predeclared transportability rules**. Six of seven source features favor the modern regime, aggregate distance favors modern by a substantial margin, and none of the seven features extends beyond 2020 in its anomalous direction.

This materially strengthens the case for prospectively qualifying the preserved R26 Week-1 vacancy/R9 component for the actual 2026 environment rather than building another retrospective 2020 guard.

It does **not** prove that R26 will be accurate in 2026; no 2026 outcomes exist in this study. It instead resolves the narrower transportability question needed before a prospective qualification design can be justified.

## Authority

R26L authorizes:
- design of a separately frozen R26M-style prospective qualification/synthesis study: **true**.

R26L does not authorize:
- exclude 2020: **false**;
- prospective shadow: **false**;
- production promotion: **false**;
- a new router: **false**;
- R9 refit: **false**;
- prediction regeneration as part of R26L: **false**;
- R22 change: **false**;
- receiving-yard mean change: **false**.

## Next frontier

Freeze R26M before generating or evaluating any prospective candidate behavior.

R26M should synthesize the immutable historical R26/R26E/R26J/R26K evidence with the now-qualified 2026 source-regime result and define, in advance, what is required to move the preserved Week-1 vacancy/R9 entitlement component into a prospective 2026 shadow candidate.

Do not invent a 2020 exemption or loosen the historical R26E gate post hoc. The purpose of R26M is to define a prospective qualification contract appropriate to the actual 2026 source regime while preserving the historical failure as explicit evidence and preserving R22/receiving-yard means/production exactly unless separately authorized.