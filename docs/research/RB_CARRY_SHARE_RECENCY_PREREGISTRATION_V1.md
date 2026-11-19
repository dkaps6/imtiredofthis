# RB carry-share recency: PREREGISTRATION V1 (frozen before any arm runs)

**Date:** 2026-11-19. **Research only. No production change.**

## Why this is new: prior-work check

| prior study | what it tested | what it did not test |
|---|---|---|
| `PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1.md` | Lead-back share level. Production's Bayesian layer over-shrinks by about 9-11pp, CI excludes 0. A transform was shipped INERT. | The window PlayerForm measures current share over |
| `PRISM_ROLE_SHRINKAGE_FIX_RESULTS_V1.md` | Shrinkage targets for shares (A4 role-rank anchor best; K4 side-accuracy cost). V2 was preregistered and never run. | Recency |
| `PRISM_SNAP_SIGNAL_AND_NOISE_FLOOR_V1.md` | Target share, where production is near the oracle (headroom 0.0077) | RB carry share |
| `player_form_v2.py` `PLAYERFORM_TEAM_CHANGE_PRIOR_K` | The prior-season pseudo-sample k, for team changers, on target share | Recency within the current season |
| `REVERSE_ENGINEER_2026_W1_10_V1.md` §6 | On raw season-to-date vs last-3 means, the best last-3 weight for RB carries is 0.70 (0.75/0.65/0.50 by season) | **PlayerForm's actual formula as the baseline**; a holdout |

**What production does today** (`player_form_v2._blend`):

    share = (1 - w) * prior_season_share + w * season_to_date_share,  with w = g / (g + 4)

**There is no recent-games component.**

## Question

For RBs, does replacing `season_to_date_share` with a blend of it and the last-3-game share
reduce next-game carry-share error, against PlayerForm's own formula?

## Design

- **Data:** nflverse weekly player stats and snap counts, 2022-2026 REG. 2022 serves only as
  prior-season history.
- **Rows:** RB player-games with an offensive snap. Each must have at least 3 earlier
  current-season games with the same team.
- **Target:** that game's carry share, the player's carries / the team's carries.
- **Arms:**
  - **C0:** PlayerForm's formula above, k = 4. The prior-season share is the player's
    previous-season carries / his team's carries in the games he played. With no prior,
    C0 is season-to-date.
  - **C1:** C0, with `season_to_date_share` replaced by
    `(1-a) * season_to_date + a * last3`.
- **Fitting:** `a` is chosen on a grid from 0 to 1 in steps of 0.05, on 2023-2024 only, then
  frozen.
- **Holdouts:** 2025 (whole season) and 2026 weeks 1-10, scored blind.
- **Primary metric:** the MAE gain, C0 minus C1, with a 95% bootstrap CI clustered by
  team-game.
- **Reported, not gating:**
  - an oracle: the player's full-season mean share on the same rows. It uses future data and
    shows only how much error is left to remove;
  - lead backs (C0 share ≥ 0.40);
  - role-change rows (|last-3 - season-to-date| ≥ 0.15);
  - team-sum check: the mean of the team's summed shares, C0 vs C1.

## Kill and pass

- **K1:** the gain CI includes 0 in either holdout → null. Nothing proceeds.
- **K2:** lead-back MAE gets worse in either holdout, beyond its CI → null.
- **PASS:** a positive gain with the CI excluding 0 in both holdouts, and no K2. A pass makes
  this a candidate only. Promotion would still need the end-to-end projection test. It would
  also need to interact correctly with the Bayesian over-shrink in the carry-concentration
  study, which happens downstream of PlayerForm.

## Prediction (scored after)

C1 wins. Fitted `a` lands at 0.4-0.7. The gain is concentrated in role-change rows and small
overall.

**Named risk:** most RB-games are steady roles, where the last-3 share only adds noise. The
overall gain could then be within noise while role-change rows improve. That would be a
conditional finding, not a pass.
