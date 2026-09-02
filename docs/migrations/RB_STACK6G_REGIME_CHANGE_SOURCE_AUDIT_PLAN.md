# RB STACK6G — Target-Week Regime-Change Source / Forensic Audit

## Status

Frozen diagnostic/source protocol. No production change. No fitted point model is authorized in STACK6G.

## Why STACK6G exists

STACK6F confirmed that strictly prior team workload history can improve team-RB carry-pool MAE/RMSE and late-season MAE, but it failed the frozen correlation/ranking gate. The remaining failure is therefore not simply global centering: the model still does not identify enough of the specific target games in which the RB room will be materially above or below its recent-history expectation.

Prior research prevents several false-new directions:

- M94/M94B/M94C already modeled total team rushing opportunity, game state, pace and football-only scoring/margin environment.
- M94D already sharpened backfield concentration within the resulting carry pool.
- M95E already modeled the exact decomposition `team rush attempts × RB-room share of all team rushes × player share of RB room`, including team QB-rush share and QB scramble share.
- M95A-D/ENV1 already tested broad matchup/environment/efficiency information.
- STACK6/6B/6C already tested lagged situational role/rotation and contraction families.
- STACK6D/6E already audited target-week availability/inactive competitor state.
- STACK6F already tested a compact strictly-prior team-RB carry-pool history model.

STACK6G therefore tests only a narrower class of information: **pregame-known structural discontinuities that can make the recent team-history window stale for the target game**.

## Hypotheses

### H1 — QB1 regime discontinuity

A team's recent QB rushing/scramble history is an average over the QBs who actually played those prior games. If the timestamp-safe expected QB1 for the target game differs materially from the QB regime that generated the recent history, the recent RB-room share prior may be stale.

This is not a generic "mobile QB steals RB carries" retest. M95E already tested generic QB-rush and scramble shares. STACK6G asks whether **target-week QB identity/change relative to the history window** is enriched in current P3/STACK6F team-RB-pool misses.

### H2 — verified playcaller regime discontinuity

A documented target-week primary playcaller change can make recent team rushing tendency stale. Existing QB research (M68) established a historical contract for season-opening playcaller mappings and documented midseason handoffs. STACK6G asks whether that state can be reconstructed for RB target games and whether changes are enriched in team-RB-pool misses.

## Historical scope

Attempt the broadest exact/source-comparable panel available, in this order:

- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

Do not force older seasons into the panel. Each season must earn inclusion independently through source/timestamp/comparability checks. The primary current failure-atlas grading remains the frozen 2025 P3/STACK6F population because that is the current point-model frontier.

## Source hierarchy

### QB1 target-week identity

Primary candidate: nflverse/nflreadpy depth-chart snapshots, using the latest snapshot strictly before scheduled kickoff. The underlying depth feed is all-position; prior ND2B work filtered it to RB/HB/FB only.

Required checks per season:

1. team-game depth snapshot coverage;
2. presence of snapshot/effective timestamps sufficient to enforce `< kickoff`;
3. QB position coverage;
4. recoverable QB depth rank / QB1 identity;
5. median and p90 age of selected pregame snapshot;
6. no target-game participation or postgame depth state used to select QB1.

### QB rushing profile

Use only strictly prior player/team history. Candidate inputs are prior-game QB rushing/scramble/design metrics already supported by repo/nflverse machinery. Target-game rushing statistics are forbidden.

### Playcaller state

Reuse only the historical contract already documented in M68:

- season-opening mappings frozen from public all-team inventories;
- documented midseason handoffs effective beginning with their public effective week;
- playcaller names are metadata only;
- no target-game or future results may define the caller.

If that mapping cannot be reconstructed at adequate coverage for RB seasons, mark playcaller state source-incomplete rather than fabricating it.

## No-model audit variables

The audit may compute deterministic, predeclared descriptive variables only. No learned coefficients.

QB regime variables:

- `target_qb1_id`
- `prior_game_qb1_id` when recoverable pregame
- `qb1_changed`
- target-QB strictly-prior rushing attempts/game
- target-QB strictly-prior designed-run rate when source permits
- target-QB strictly-prior scramble rate when source permits
- team prior3 QB rush share
- team prior3 QB scramble share when source permits
- target-QB minus recent-team QB-rush propensity delta
- target-QB minus recent-team scramble propensity delta
- new-to-team / insufficient-prior-history flags where deterministically knowable

Playcaller variables:

- `target_playcaller`
- `prior_game_playcaller`
- `playcaller_changed`
- `playcaller_new_to_team`
- prior caller sample-count diagnostics

## Frozen 2025 failure definitions

Use the authoritative corrected STACK6F artifact/run as the team-pool reference:

- run `33578446070`
- branch `research-rb-stack6f-team-pool`
- SHA `a72361a328533101f670e672314e21fa1b8672f4`
- artifact `rb-stack6f-team-pool`

For the main W6-18 atlas, define from frozen P3 team-pool residuals:

- `POOL_OVER_3`: P3 predicted RB carries minus actual RB carries >= 3
- `POOL_OVER_5`: >= 5
- `POOL_UNDER_3`: actual minus predicted >= 3
- `POOL_UNDER_5`: >= 5
- `POOL_ABS_5`: absolute residual >= 5

These are descriptive bins only. They are not candidate-selection thresholds and may not be used to tune a model in STACK6G.

## Required outputs

1. `stack6g_source_coverage_by_season.csv`
2. `stack6g_qb_regime_atlas_2025.csv`
3. `stack6g_qb_regime_summary_2025.csv`
4. `stack6g_playcaller_source_summary.csv`
5. `stack6g_integrity.csv`
6. `stack6g_disposition.csv`

The source table must state, season by season, whether exact QB1 reconstruction is safe enough to use in a future model.

The 2025 atlas must compare regime-change prevalence and continuous QB propensity deltas across the frozen P3 pool-error bins.

## Frozen interpretation rules

STACK6G can end in only one of these dispositions:

- `STACK6G_QB_REGIME_SOURCE_AND_FORENSIC_SIGNAL_SUPPORTED`
- `STACK6G_PLAYCALLER_SOURCE_AND_FORENSIC_SIGNAL_SUPPORTED`
- `STACK6G_MULTIPLE_REGIME_SIGNALS_SUPPORTED`
- `STACK6G_SOURCE_USABLE_BUT_NO_MATERIAL_FORENSIC_SIGNAL`
- `STACK6G_REGIME_SOURCES_NOT_TIMESTAMP_SAFE_OR_INCOMPLETE`

"Supported" is deliberately descriptive, not a production/model promotion. It requires all of:

1. source/timestamp integrity passes for the season(s) used;
2. adequate 2025 coverage to evaluate the current frontier;
3. a directional relationship that is football-coherent and materially concentrated in the relevant pool-error bins rather than appearing only in one hand-picked team/game;
4. no sportsbook input, outcome-based source selection, target participation, or target-game QB rushing used upstream.

No numerical point-model retention gate is defined in STACK6G because no model is fit.

## What happens after STACK6G

If no regime signal is supported, do not fit a STACK6G point correction. Preserve P3 as champion and move to the next genuinely novel RB mechanism.

If one or more regime signals are supported, freeze a separate follow-up model protocol before fitting. That follow-up must use a compact predeclared feature set, preserve P3 allocation semantics, and qualify the **team RB carry pool first** before any player-level recomposition.

Sportsbook/player-prop data remains downstream benchmark only.