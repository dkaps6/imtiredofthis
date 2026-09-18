import pandas as pd

from scripts.backtest.rb_lane_a_m91_parity_diagnostic_v1 import (
    compare_deterministic_trace_arm,
    compare_trace_columns_by_week,
    find_fresh_only_rows,
)


def _row(season, week, team, player_clean_key, market, player="X", position="RB", **extra):
    base = {
        "season": season,
        "week": week,
        "team": team,
        "player_clean_key": player_clean_key,
        "market": market,
        "player": player,
        "position": position,
    }
    base.update(extra)
    return base


def test_find_fresh_only_rows_isolates_extra_identity():
    fresh = pd.DataFrame(
        [
            _row(2024, 1, "KC", "p1", "rush_yards"),
            _row(2024, 1, "KC", "p2", "rush_yards", player="Extra Guy"),
        ]
    )
    canonical = pd.DataFrame([_row(2024, 1, "KC", "p1", "rush_yards")])
    extra = find_fresh_only_rows(fresh, canonical)
    assert len(extra) == 1
    assert extra.iloc[0]["player_clean_key"] == "p2"
    assert extra.iloc[0]["player"] == "Extra Guy"


def test_find_fresh_only_rows_empty_when_identical():
    fresh = pd.DataFrame([_row(2024, 1, "KC", "p1", "rush_yards")])
    canonical = pd.DataFrame([_row(2024, 1, "KC", "p1", "rush_yards")])
    assert find_fresh_only_rows(fresh, canonical).empty


def test_compare_trace_columns_by_week_reports_max_abs_delta():
    fresh = pd.DataFrame(
        [_row(2024, 1, "KC", "p1", "rush_yards", rules_plays_est=65.0, rules_pass_rate=0.6)]
    )
    canonical = pd.DataFrame(
        [_row(2024, 1, "KC", "p1", "rush_yards", rules_plays_est=64.0, rules_pass_rate=0.6)]
    )
    report = compare_trace_columns_by_week(fresh, canonical, columns=["rules_plays_est", "rules_pass_rate"])
    assert report["1"]["matched_rows"] == 1
    assert report["1"]["rules_plays_est"] == 1.0
    assert report["1"]["rules_pass_rate"] == 0.0


def test_compare_trace_columns_by_week_flags_missing_column():
    fresh = pd.DataFrame([_row(2024, 1, "KC", "p1", "rush_yards")])
    canonical = pd.DataFrame([_row(2024, 1, "KC", "p1", "rush_yards")])
    report = compare_trace_columns_by_week(fresh, canonical, columns=["not_a_real_column"])
    assert report["1"]["not_a_real_column"] == "column_missing"


def test_compare_deterministic_trace_arm_excludes_contested_player_from_comparison():
    # arm has the contested player (p2) plus another player (p1); canonical
    # only ever has p1 -- the contested player's own row must never enter
    # the matched-rows comparison either way.
    arm = pd.DataFrame(
        [
            _row(2024, 1, "TB", "p1", "rush_yards", rules_rush_share=0.30, rules_ypc=4.1),
            _row(2024, 1, "TB", "p2", "rush_yards", rules_rush_share=0.10, rules_ypc=3.9),
        ]
    )
    canonical = pd.DataFrame(
        [_row(2024, 1, "TB", "p1", "rush_yards", rules_rush_share=0.30, rules_ypc=4.1)]
    )
    report = compare_deterministic_trace_arm(
        arm, canonical, week=1, team="TB", exclude_player_clean_key="p2"
    )
    assert report["matched_rows"] == 1
    assert report["rules_rush_share"] == 0.0
    assert report["rules_ypc"] == 0.0


def test_compare_deterministic_trace_arm_reports_max_abs_delta_on_other_players():
    arm = pd.DataFrame(
        [_row(2024, 1, "TB", "p1", "rush_yards", rules_rush_share=0.35, rules_ypc=4.1)]
    )
    canonical = pd.DataFrame(
        [_row(2024, 1, "TB", "p1", "rush_yards", rules_rush_share=0.30, rules_ypc=4.1)]
    )
    report = compare_deterministic_trace_arm(
        arm, canonical, week=1, team="TB", exclude_player_clean_key="godwin"
    )
    assert report["matched_rows"] == 1
    assert abs(report["rules_rush_share"] - 0.05) < 1e-9
    assert report["rules_ypc"] == 0.0


def test_compare_deterministic_trace_arm_flags_missing_column():
    arm = pd.DataFrame([_row(2024, 1, "TB", "p1", "rush_yards")])
    canonical = pd.DataFrame([_row(2024, 1, "TB", "p1", "rush_yards")])
    report = compare_deterministic_trace_arm(
        arm,
        canonical,
        week=1,
        team="TB",
        exclude_player_clean_key="godwin",
        columns=["not_a_real_column"],
    )
    assert report["not_a_real_column"] == "column_missing"
