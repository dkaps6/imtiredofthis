# Migration 83 — Defensive Adaptive Gameplan Source / Mechanism Audit

## Status

`PREREGISTERED / DIAGNOSTIC ONLY`

M83 does **not** fit a QB passing-yards model. It tests whether a genuinely new pregame observable can be constructed: how a defense is likely to change its tactical behavior against the specific offense/QB archetype it is about to face.

The authoritative football-only full-stack benchmark entering M83 is M82's OOS ensemble:

- 884 canonical-v3 QB games
- combined pass-yard MAE `56.749517`
- RMSE `72.303902`
- correlation `0.149475`
- 100+ yard misses `123`

M83 may not tune that model or use sportsbook variables.

## Research question

The project has repeatedly tested static defensive strength, pressure, coverage, box, game script and offensive tendencies. M83 asks a different question:

> Does a defense systematically deviate from its own normal tactical profile when it faces an offense/QB archetype similar to the upcoming opponent, and can that deviation be predicted strictly before kickoff from prior comparable-opponent games?

This is the `DEFENSIVE_ADAPTIVE_GAMEPLAN` frontier frozen by M82.

## Explicit novelty versus prior migrations

M83 is **not** a retest of:

- M14 standalone man/zone, shell or box frequencies;
- M22-M23 / M45 / M56 generic pressure or static defensive matchup;
- M40-M42 generic team pass tendency;
- M64-M65 state/possession/dropback architecture;
- M67-M69 offense opening/playcaller/tendency signals;
- M80/M81 FTN fields as direct QB residual predictors.

Those quantities may be used only as **descriptors or response variables** inside the new conditional-adaptation construction.

The materially new quantity is:

`predicted target-game defense deviation from its own baseline, conditional on similarity between the upcoming offense and offenses the defense previously faced`.

## Outcome boundary

M83 never reads or scores QB passing yards, attempts, YPA, sportsbook props, lines, spreads, totals, moneylines or betting outcomes.

- 2023 and 2024 defensive tactical responses may be used for mechanism feasibility.
- 2025 source availability/coverage may be audited only.
- 2025 tactical-response performance must not be used to choose the mechanism.
- 2025 QB target outcomes remain reserved for a later untouched confirmation migration if the frontier eventually survives development.

## Source families

### A. FTN charting + nflverse PBP — deployable-history candidate

M80 established an in-season update contract for FTN charting: target-game charting is postgame only and may become strictly-prior information for later games.

FTN/PBP will be used for:

**Offense-archetype descriptors**

- `is_motion`
- `is_screen_pass`
- `is_rpo`
- `is_no_huddle`
- `is_play_action`
- `qb_location` -> deterministic shotgun indicator
- `n_offense_backfield`
- `is_qb_out_of_pocket`
- `is_qb_fault_sack`
- PBP pass-play share

These descriptors are **not standalone M83 predictors**. They define similarity between offenses.

**Defense tactical-response variables**

- `n_blitzers` mean on pass plays
- `n_blitzers > 0` event rate on pass plays
- `n_pass_rushers` mean on pass plays

`n_pass_rushers` was previously exhausted as a direct QB feature; its use here is only as a label of how the defense changed its rush construction.

### B. nflverse participation + PBP — historical auxiliary source

Historical defensive response will also be audited for:

- man rate
- zone rate
- coverage-shell availability/category diversity
- defenders-in-box availability / mean

M80 established that nflverse 2023+ participation is not a trustworthy in-season deployment source. Therefore these variables may demonstrate historical scientific feasibility but **cannot by themselves qualify M84 for deployment** unless a separate live source contract is established.

## Frozen game-level construction

### 1. Actual offense game profiles

For each completed offense-game, aggregate the offense-archetype descriptors listed above.

### 2. Pregame offense archetype

For every offense-game, construct its pregame archetype using only that offense's completed games strictly before the target game.

- trailing window: last `8` completed offense games
- minimum natural history: `3` prior offense games
- no target-game descriptor may enter its own pregame archetype
- prior-season games may be used when they are among the last eight completed games

### 3. Defense baseline

For every defense-game and each response variable, construct the defense's normal baseline using its last `8` completed games strictly before the target game.

Minimum natural defense history for a target evaluation: `6` prior defense games.

### 4. Prior-game adaptation labels

For each completed prior defense-game:

`observed_adaptation = actual defensive response in that game - defense baseline known before that game`

This ensures a prior game contributes a *change from that defense's normal behavior*, rather than merely a static defense strength.

### 5. Comparable-opponent distance

For a target defense-game, compare the upcoming offense's **pregame archetype** to the **pregame archetype that was known before each prior opponent game** faced by that defense.

All rate descriptors are naturally on `[0,1]`. `n_offense_backfield` is deterministically scaled by `/3` and clipped to `[0,1]`.

Distance is frozen as the mean absolute difference across available scaled archetype dimensions. No learned distance metric, feature weighting, clustering, PCA, neural embedding or post-result variable selection is allowed.

Similarity is `1 - distance`, clipped to `[0,1]`.

### 6. Comparable-opponent prediction

For every eligible target defense-game:

- consider only the defense's games strictly before the target;
- require candidate prior games to have a natural pregame opponent archetype;
- select exactly the nearest `4` prior opponent games by frozen distance;
- weight their observed adaptation labels by `1 / (distance + 0.05)`;
- add the weighted predicted adaptation to the target defense's trailing-8 baseline.

No `k` search, threshold search or weight-function search is allowed after results are visible.

## Frozen mechanism-development sample

Mechanism scoring is limited to **2024 regular season Weeks 5-18**, subject to natural history eligibility.

2022-2023 and earlier 2024 games may provide strictly-prior history.

2025 may be used only for source coverage/inventory, not mechanism scoring.

## Source / density gates

All are required for the deployable FTN adaptation family to qualify:

1. FTN and PBP regular-season Weeks 1-18 are available for required history seasons.
2. FTN/PBP play join rate is at least `0.95` in each required scoring/history season.
3. Each primary deployable defense-response field has at least `0.80` usable coverage in each required scoring/history season.
4. At least `80%` of otherwise eligible 2024 defense-games have four prior defense games with natural comparable-opponent archetypes.
5. Median similarity of the selected nearest-four historical opponents is at least `0.70`.

If source failures make these quantities unmeasurable, M83 fails closed.

## Frozen mechanism-predictability gate

M83 compares two forecasts of the target defense's actual tactical response:

- `BASELINE`: defense trailing-8 response only;
- `ADAPTIVE`: trailing-8 baseline + nearest-four comparable-opponent predicted deviation.

For each deployable FTN response metric, report MAE, RMSE and correlation.

`DEFENSIVE_ADAPTATION_MECHANISM_QUALIFIED` requires:

- source/density gates pass;
- ADAPTIVE improves MAE by at least `5%` versus BASELINE on **at least one** deployable response metric;
- the improved metric also has correlation gain at least `+0.05`;
- ADAPTIVE may not worsen MAE by more than `5%` on **all** other deployable response metrics.

This gate is about predicting defensive behavior, not QB yards.

## Historical-only participation interpretation

Participation response metrics receive the same descriptive BASELINE vs ADAPTIVE comparison when source coverage permits, but they cannot independently trigger M84 because their current in-season deployment contract is not qualified.

A strong historical man/zone/shell adaptation result with no live source must be labeled `HISTORICAL_SIGNAL_SOURCE_BLOCKED`, not promoted.

## M84 boundary if M83 qualifies

Only if the deployable FTN adaptive mechanism qualifies may M84 test whether those **frozen pregame adaptive-response features** improve QB prediction.

M84 must:

- use the M82 `56.749517` OOS full-stack benchmark as the authoritative reference;
- develop on 2024 only;
- separately score attempt surprise, YPA surprise, passing-yard MAE/RMSE/correlation and 100+ misses;
- keep 2025 QB outcomes untouched for a later confirmation migration;
- freeze the M83 feature construction unchanged.

If M83 does not qualify, do not rescue the same information with another similarity metric, another `k`, clustering, XGBoost or a model zoo.

## Production action

`production_actionable = false`
