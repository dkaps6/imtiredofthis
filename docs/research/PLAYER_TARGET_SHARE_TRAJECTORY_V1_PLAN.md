# Player Target Share Trajectory V1 — Frozen Historical Diagnostic

**STATUS: FROZEN BEFORE RESULT. NO FITTED CANDIDATE. NO PRODUCTION CHANGE.**

## Motivation

The promoted WR and TE entitlement systems already contain meaningful individual-player information:

- season-to-date / Bayesian target-share state upstream;
- WR M38 hierarchy;
- WR-R15 / TE-R5P participation and snap-continuity features.

Yet exact promoted-authority diagnostics found persistent same-player opportunity error after those layers:

WR:
- O1 signed opportunity persistence pooled rho = +0.1699
- O2 opportunity difficulty pooled rho = +0.3183

TE:
- O1 signed opportunity persistence pooled rho = +0.2500
- O2 opportunity difficulty pooled rho = +0.3936

A separate situational-role family (third down / red zone / two minute) was tested and CLOSED.

The current specialist feature contracts do **not** explicitly encode the direction of recent individual target-share movement.

## Question

> Does a player's strictly-prior **recent target-share trajectory** explain signed target/opportunity error left behind by the exact promoted WR/TE entitlement architecture?

This is a same-player current-role-state diagnostic, not a position-level adjustment.

## Exact promoted authorities

### WR
- run `34238301577`
- artifact `10061328722`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- file `backtests/wr_r15_wr1_anchor_v1/wr_r15_confirmation_predictions.csv`
- variant `WR_R15_WR1_ANCHORED_PARTICIPATION`
- target seasons: 2023, 2024

WR opportunity error:
`(pred_targets - actual_targets) * (mc_rec_yards / pred_targets)`

### TE
- run `34152797603`
- artifact `10029942404`
- digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`
- file `data/backtests/te_r5p_production_contract_refit_v1/te_r5p_oos_player_casebook.csv`
- target seasons: 2024, 2025

TE opportunity error:
`(candidate_targets_r5p - targets) * (candidate_rec_yards_r5p / candidate_targets_r5p)`

Negative opportunity error means the promoted model underpredicted the player's target opportunity.

## Identity bridge

Use canonical nflreadpy weekly player stats and `player_form_v2._normalize_weekly` to map:

`(season, player_clean_key) -> stable player_id`

A target authority row is eligible only when that season/key maps uniquely to one non-empty stable ID.

No fuzzy PBP display-name matching.

Required identity coverage:
- WR >= 95%
- TE >= 95%

## Historical source

Free nflreadpy regular-season PBP, 2023-2025.

A target event is:
- official pass attempt;
- non-empty receiver player ID;
- offense team present;
- sack excluded;
- two-point attempt excluded.

No sportsbook data.

## Frozen trajectory construction

For each exact authority target player-game:

1. use only **same-season, same target-team** completed offensive games strictly before the target week;
2. require at least **4 prior target-team games**;
3. sort completed team games chronologically;
4. define `RECENT2` = latest 2 completed target-team games;
5. define `EARLIER` = all completed target-team games before those recent 2;
6. require EARLIER to contain at least 2 team games;
7. require positive team target denominators in both windows.

Compute:

`recent2_share = player RECENT2 targets / team RECENT2 targets`

`earlier_share = player EARLIER targets / team EARLIER targets`

`trajectory_delta = recent2_share - earlier_share`

The player may have zero targets in one window; zero is valid football information.

No target-game play enters either window.

## Frozen hypothesis and sign

If a player's target role is rising recently relative to his earlier same-season state, a season-to-date average plus participation-based entitlement may lag behind that change.

Therefore:

`trajectory_delta > 0 -> promoted model more likely to underpredict opportunity`

Since opportunity error is predicted minus actual, the expected association is:

`Spearman(trajectory_delta, opportunity_error) < 0`

The opposite sign cannot be rescued after results.

## Diagnostics

For WR:
- 2023 rho
- 2024 rho
- pooled rho
- rows / players
- player-cluster bootstrap P(rho < 0)

For TE:
- 2024 rho
- 2025 rho
- pooled rho
- rows / players
- player-cluster bootstrap P(rho < 0)

Combined:
- pooled WR+TE rho
- player-position cluster bootstrap P(rho < 0)

Secondary descriptive diagnostics:
- trajectory delta SD
- 10th / 50th / 90th percentiles
- fraction abs trajectory >= 0.03
- fraction abs trajectory >= 0.05
- mean opportunity error by trajectory sign (rising / flat-zero / falling)

No threshold is used to select rows or route predictions.

## Support floors

WR:
- >=500 scoreable rows in 2023
- >=500 in 2024
- >=100 distinct WR identities pooled

TE:
- >=250 scoreable rows in 2024
- >=250 in 2025
- >=50 distinct TE identities pooled

Identity coverage >=95% each position.

## Bootstrap

- 5,000 replicates
- seed `20261007`
- WR and TE separately: cluster by stable player identity
- combined: cluster by `position_group + player_clean_key`
- resample clusters with replacement
- preserve all eligible target rows within sampled player cluster
- statistic = Spearman(trajectory_delta, opportunity_error)

## Confirmation gate

`PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED` requires all:

1. support floors pass;
2. WR 2023 rho < 0;
3. WR 2024 rho < 0;
4. TE 2024 rho < 0;
5. TE 2025 rho < 0;
6. pooled WR rho <= -0.05;
7. pooled TE rho <= -0.05;
8. WR cluster-bootstrap P(rho < 0) >= 0.95;
9. TE cluster-bootstrap P(rho < 0) >= 0.95;
10. combined cluster-bootstrap P(rho < 0) >= 0.99;
11. zero target/future leakage;
12. zero sportsbook input;
13. zero fitted model;
14. zero production change.

Otherwise, if support passes:
`NO_ACTIONABLE_PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL`

If support/identity prevents judgment:
`PLAYER_TARGET_SHARE_TRAJECTORY_SOURCE_LIMITED`

## Anti-rescue

After results are exposed, do not search:
- recent1 / recent3 / recent4 windows;
- weighted recency;
- alternate earlier baseline windows;
- target-count floors;
- rising-only / falling-only carveouts;
- WR-only or TE-only rescue;
- WR1 / WR2+ rescue;
- threshold routing;
- residual-history blends;
- sportsbook-conditioned variants.

A confirmed signal would authorize only a separately frozen integration candidate that preserves:
- team target mass;
- WR M38 anchor contract unless separately justified;
- WR-R15 / TE-R5P room conservation;
- QB mean science;
- downstream simulation/distribution invariants.

Models fit: **0**
Sportsbook inputs: **0**
2026 outcomes: **0**
Production mutations: **0**
