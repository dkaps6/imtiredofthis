import numpy as np
import pandas as pd
import pytest

from scripts.research.diagnose_market_implied_game_script_v1 import (
    MIN_PRIOR_WEEKS,
    ROLLING_WINDOW,
    add_rolling_baseline,
    build_cohort,
    fit_and_evaluate,
    load_market_schedule,
)


def _team_weekly(n_weeks=20, seed=3):
    """Four teams, two seasons. Each team's true weekly plays wobble around a
    fixed level the naive rolling-average baseline cannot see coming, but the
    wobble is driven by a synthetic market signal available in the fixture's
    market frame -- so a fitted blend that uses the market column should beat
    the historical-only baseline out of sample.
    """
    rng = np.random.default_rng(seed)
    rows = []
    teams = {"AAA": 64.0, "BBB": 60.0, "CCC": 66.0, "DDD": 58.0}
    for season in (2024, 2025):
        for team, level in teams.items():
            for week in range(1, n_weeks + 1):
                market_signal = rng.uniform(-6, 6)
                plays = level + 1.5 * market_signal + rng.normal(0, 1.0)
                dropback_rate = 0.57 + 0.01 * market_signal + rng.normal(0, 0.01)
                rows.append({
                    "season": season, "week": week, "team": team,
                    "plays_est": plays, "dropback_rate": dropback_rate,
                    "_market_signal": market_signal,
                })
    return pd.DataFrame(rows)


def _market_from_team_weekly(team_weekly: pd.DataFrame) -> pd.DataFrame:
    x = team_weekly[["season", "week", "team", "_market_signal"]].copy()
    x["market_team_implied"] = x["_market_signal"]
    x["market_abs_spread"] = x["_market_signal"].abs()
    x["market_total"] = 44.0
    x["market_team_spread"] = -x["_market_signal"]
    return x.drop(columns=["_market_signal"])


def test_add_rolling_baseline_uses_only_strictly_prior_weeks():
    team_weekly = _team_weekly()
    out = add_rolling_baseline(team_weekly, "plays_est")
    first_week = out.loc[(out.team.eq("AAA")) & (out.season.eq(2024)) & (out.week.eq(1))]
    assert first_week["plays_est_prior_avg"].isna().all()
    later = out.loc[(out.team.eq("AAA")) & (out.season.eq(2024)) & (out.week.eq(10))].iloc[0]
    prior_rows = out.loc[
        (out.team.eq("AAA")) & (out.season.eq(2024)) & (out.week.between(2, 9))
    ]
    expected = prior_rows.tail(ROLLING_WINDOW)["plays_est"].mean()
    assert later["plays_est_prior_avg"] == pytest.approx(expected)


def test_build_cohort_drops_rows_missing_baseline_or_market():
    team_weekly = _team_weekly()
    market = _market_from_team_weekly(team_weekly)
    cohort = build_cohort(team_weekly, market, "plays_est")
    # The very first weeks of the very first season have no prior history at
    # all (history crosses season boundaries by design, so 2025 week 1 is
    # fine -- it has all of 2024 behind it) and must be dropped.
    assert not ((cohort.season.eq(2024)) & (cohort.week.eq(1))).any()
    assert ((cohort.season.eq(2025)) & (cohort.week.eq(1))).any()
    assert cohort["plays_est_prior_avg"].notna().all()
    assert cohort["market_team_implied"].notna().all()


def test_market_blend_beats_fitted_baseline_only_out_of_sample():
    """The correct comparator is the fitted-baseline-only arm, not the raw
    unfit baseline -- otherwise the market arm gets undue credit merely for
    fitting the baseline's own scale/bias (Codex P1 on PR #558).
    """
    team_weekly = _team_weekly(n_weeks=25)
    market = _market_from_team_weekly(team_weekly)
    cohort = build_cohort(team_weekly, market, "plays_est")
    result = fit_and_evaluate(cohort, target_col="plays_est", fit_season=2024, test_season=2025)
    assert result["status"] == "OK"
    for col in ["raw_baseline_mae", "fitted_baseline_only_mae", "baseline_plus_market_mae"]:
        assert col in result
    assert result["baseline_plus_market_mae"] < result["fitted_baseline_only_mae"]
    assert result["incremental_market_mae_improvement"] > 0


def test_load_market_schedule_favors_home_team_when_spread_positive(monkeypatch):
    """nflverse's spread_line is positive when the home team is favored
    (verified against real 2023 results, e.g. DAL home spread_line=+17.5,
    won 49-17) -- the favored side's implied total must be the larger half
    of the game total, not the smaller half (Codex P1 on PR #558).
    """
    fake_schedule = pd.DataFrame([{
        "season": 2023, "week": 10, "game_type": "REG",
        "home_team": "DAL", "away_team": "NYG",
        "spread_line": 17.5, "total_line": 38.5,
    }])

    import sys
    import types

    fake_module = types.ModuleType("nflreadpy")
    fake_module.load_schedules = lambda season: fake_schedule
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_module)

    market = load_market_schedule([2023])
    home = market.loc[market.team.eq("DAL")].iloc[0]
    away = market.loc[market.team.eq("NYG")].iloc[0]
    assert home["market_team_implied"] > away["market_team_implied"]
    assert home["market_team_implied"] == pytest.approx((38.5 + 17.5) / 2.0)
    assert away["market_team_implied"] == pytest.approx((38.5 - 17.5) / 2.0)


def test_fit_and_evaluate_fails_closed_on_insufficient_rows():
    tiny = pd.DataFrame({
        "season": [2024] * 5 + [2025] * 5,
        "plays_est": np.linspace(60, 65, 10),
        "plays_est_prior_avg": np.linspace(59, 64, 10),
        "market_team_implied": np.linspace(-2, 2, 10),
        "market_abs_spread": np.abs(np.linspace(-2, 2, 10)),
    })
    result = fit_and_evaluate(tiny, target_col="plays_est", fit_season=2024, test_season=2025)
    assert result["status"] == "INSUFFICIENT_ROWS"
