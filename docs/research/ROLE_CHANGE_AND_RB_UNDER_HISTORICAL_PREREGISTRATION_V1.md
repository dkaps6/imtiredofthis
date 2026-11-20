# Role-change rule + RB-under rule on 2024-2025 historical lines: PREREGISTRATION V1

**Date:** 2026-11-20, frozen before any of this data is scored for these rules.
**Research only.**

## Data, never used by any betting rule in this program

`data/backtests/historical_market_vegas_benchmark_v1/historical_market_vegas_benchmark_detail.csv`:
- 17,786 graded player-prop lines, 2024 and 2025 regular seasons, weeks 1-18;
- mostly DraftKings (17,363), some FanDuel (423);
- `line`, `over_odds`, `under_odds` and `actual`.

It was built for the base-engine Vegas benchmark (that README; naive model sides lost 1-6%).
**No rule tested here has been scored on it.** Players are matched to nflverse gsis ids by team +
week + first-initial/last-name key.

## Test A: the RB rushing-UNDER rule, exactly as frozen in `RB_UNDER_TRACKER.md`

- **Market:** RB `rush_yards`, UNDER at `under_odds`.
- **Flags, all strictly prior:**
  - `dog`: the team is an underdog by more than 2.5, from the nflverse `spread_line`.
  - `cold`: season-to-date YPC is below 0.95 x league RB YPC. Weeks 1-2 use last season plus
    the current season, as in `rb_rush_situation_flags_v1.priors`.
  - `cut`: `carries_pg x (0.79 x lg_ypc + 0.21 x own_ypc) / line > 1.2`.
- **Qualifies:** at least 2 flags, line ≥ 30, and at least 3 prior games.
- **Primary:** the UNDER hit rate and ROI by season, with a game-clustered 95% CI on ROI. Weeks
  above breakeven are reported.

## Test B: the role-change rule (new)

Following `RB_CARRY_SHARE_RECENCY_RESULTS_V1.md`, recent games deserve about 0.85 weight for RB
carry share, while the line puts about 0.04 on them (`TEAM_PLAYER_IDENTITY_2026_V1.md` §3).

- **Market:** RB `rush_yards`.
- **Inputs:** current season, same team, at least 4 prior games.
  - `std` = season-to-date carry share.
  - `l3` = carry share over the last 3 games.
  - `delta = l3 - std`.
- **Rule:**
  - **rising role:** `delta ≥ +0.15` → **OVER**;
  - **falling role:** `delta ≤ -0.15` → **UNDER**.
  - The directions are scored separately.
- **Secondary (reported):** the same rule for receivers, `rec_yards` with target share,
  `delta ≥ +0.08` / `≤ -0.08`.
  - **Predicted not to work:** receiver recency deserves only about 0.2 weight, so receiver
    role swings mostly revert (`REVERSE_ENGINEER_2026_W1_10_V1.md` §6).
  - This doubles as a negative control.

## Test C: the blind sides (reported)

OVER vs UNDER hit rate and ROI by market and season, model ignored. It checks
`WHAT_WE_KNOW_2026_W11.md` §1 ("overs lost 7 of 7 weeks") on two other seasons.

## Pass / fail

- **Test A holds** if the UNDER is above breakeven in **both** 2024 and 2025. Lines of 30+ are
  the rule; sub-30 lines are reported but excluded.
- **Test B holds** for a direction if its ROI is positive in both seasons **and** the pooled
  game-clustered CI excludes 0.
- Anything else is recorded as failing on unseen seasons.

## Prediction

- **A:** positive in both seasons, smaller than the 2026 backtest (66%); about 56-60%.
- **B:** the RB falling-role UNDER is positive; the rising-role OVER is weak. Overs carry the
  market-wide penalty.
- **Receivers:** no edge, possibly reversed.
- **C:** overs lose in both seasons.

**Named risk:** the 2024-25 archive is mostly DraftKings, which may price differently from
FanDuel. Name-key matching can also drop or mismatch players. The match rate is reported.
