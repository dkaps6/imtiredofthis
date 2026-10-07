# Player Target Depth Dispersion V1 — Frozen Diagnostic Plan

**STATUS: FROZEN BEFORE RESULT. DIAGNOSTIC ONLY. NO PRODUCTION CHANGE.**

## Parent finding

Promoted WR and TE entitlement models leave persistent **player-specific efficiency difficulty** but not stable signed efficiency bias:

WR:
- E1 signed efficiency persistence = FAIL
- E2 absolute efficiency difficulty = PASS
- pooled E2 Spearman = 0.2833

TE:
- E1 signed efficiency persistence = FAIL
- E2 absolute efficiency difficulty = PASS
- pooled E2 Spearman = 0.2240

Therefore this lane asks what actual football characteristic may make one individual receiver intrinsically harder to translate from targets into yards.

## Hypothesis

A receiver whose target depths are more dispersed has a wider and less stable per-target yardage process.

Frozen hypothesis:

> Higher strictly-prior **target-depth dispersion** for the same individual WR/TE should associate with larger absolute efficiency-component error in the next promoted-authority player-game.

This is a player-level uncertainty/difficulty question, **not** a mean YPT correction.

## Novelty / anti-retest

This is materially distinct from closed work:

- WR R7 tested persistent explosive/YAC/air-yard **level traits** as explanations for signed YPR misses; it did not test target-depth dispersion against absolute efficiency difficulty.
- M72 aggregated receiving-weapon traits at team level for QB matchup transmission.
- M75 tested NGS separation/cushion/aDOT/YACOE interactions for mean accuracy.
- WR-R3 width used historical model-error difficulty, not football-only target-depth geometry.
- TE Width V2 was a generic TE distribution-width candidate, not an individual football-state feature.
- QB-receiver pair YPT V1 is closed and is not reused.

No R7/M75/Width-V2 rescue is permitted from this lane.

## Exact promoted authorities

### WR
Run `34238301577`, artifact `10061328722`,
digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`.

Use variant:
`WR_R15_WR1_ANCHORED_PARTICIPATION`.

Target seasons:
- 2023
- 2024

Efficiency component:

`pred_ypt = mc_rec_yards / pred_targets`

`actual_ypt = actual_rec_yards / actual_targets` when actual_targets > 0, else 0

`efficiency_error = actual_targets * (pred_ypt - actual_ypt)`

Target:
`abs_efficiency_error = abs(efficiency_error)`

### TE
Run `34152797603`, artifact `10029942404`,
digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`.

Target seasons:
- 2024
- 2025

Use the identical efficiency-component definition with:
- `candidate_targets_r5p`
- `candidate_rec_yards_r5p`
- `targets`
- `rec_yards`.

## Identity

Bridge exact authority `player_clean_key` to stable GSIS receiver ID through canonical nflreadpy weekly player stats normalized by `player_form_v2._normalize_weekly`.

Require one unambiguous stable ID for the target season/key.

No fuzzy PBP display-name matching.

## Historical football source

Free regular-season nflverse/nflreadpy PBP, 2022-2025.

Target event:
- official pass attempt;
- receiver player ID present;
- sack excluded;
- two-point attempt excluded.

Feature source field:
- `air_yards`.

## Frozen pregame feature

For each target authority player-game:

1. use only the same receiver's target events strictly before target season/week;
2. identify latest up to **8 completed receiver-games** with at least one target;
3. require at least **4 prior receiver-games**;
4. inside those games, require at least **10 finite air-yard target events**;
5. calculate population standard deviation:

`prior8_target_depth_sd = std(air_yards, ddof=0)`.

No target-game play may enter the feature.

No clipping, winsorization, log transform, threshold, or standardization.

Secondary descriptive values:
- prior finite air-target count;
- prior mean air yards per target;
- prior deep-target rate (air_yards >=20).

Those secondary values cannot determine PASS/FAIL and cannot rescue the primary feature.

## Primary diagnostic

For WR and TE separately and by target season:

`Spearman(prior8_target_depth_sd, current_abs_efficiency_error)`

Expected sign:
- **positive**.

Report:
- rows;
- distinct players;
- Spearman;
- depth-SD distribution.

## Support floors

WR:
- 2023 >=500 scoreable rows
- 2024 >=500
- pooled >=100 distinct WRs

TE:
- 2024 >=250
- 2025 >=250
- pooled >=50 distinct TEs

Identity coverage on each exact parent authority:
- >=95%.

## Cluster bootstrap

- 5,000 replicates
- seed `20261007`
- WR/TE: resample stable player clusters with replacement
- combined: cluster by `position_group + player_clean_key`
- preserve all eligible rows from sampled player
- statistic = Spearman(target-depth SD, absolute efficiency error)

## Confirmation gate

`PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED` requires all:

1. support floors pass;
2. WR 2023 rho > 0;
3. WR 2024 rho > 0;
4. TE 2024 rho > 0;
5. TE 2025 rho > 0;
6. pooled WR rho >= 0.05;
7. pooled TE rho >= 0.05;
8. WR bootstrap P(rho > 0) >= 0.95;
9. TE bootstrap P(rho > 0) >= 0.95;
10. combined bootstrap P(rho > 0) >= 0.99;
11. combined pooled rho >= 0.05;
12. zero same/future feature violations;
13. zero sportsbook inputs;
14. zero 2026 outcomes;
15. zero fitted models;
16. production unchanged.

If support passes but gate fails:
`NO_ACTIONABLE_PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY`.

If support fails:
`PLAYER_TARGET_DEPTH_DISPERSION_SOURCE_LIMITED`.

## Interpretation boundary

A confirmed result would mean target-depth geometry helps explain why some individual receivers are systematically harder to project for yards-per-target outcomes.

It would **not** authorize:
- a YPT mean shift;
- a generic WR/TE width multiplier;
- direct production integration.

It would authorize only a separately frozen **mean-neutral player-specific distribution/uncertainty candidate**.

## Anti-rescue

After result exposure, do not search:
- standard deviation windows other than latest 8 games;
- IQR/MAD/range as rescue dispersion measures;
- deep-target-rate-only rescue;
- WR-only or TE-only rescue;
- WR1/WR2+ rescue;
- aDOT mean correction;
- YAC/YPT/YPR mean transforms;
- threshold routing;
- sportsbook conditioning.

Models fit: **0**
Sportsbook inputs: **0**
2026 outcomes: **0**
Production mutations: **0**
