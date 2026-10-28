

def test_opponent_values_and_multipliers_are_preserved():
    """The 5a gap that blocked three questions: names kept, values dropped.

    The first Week-7 preservation reduced cb_coverage_team.csv -- the canonical
    coverage artifact written by build_coverage_v2 and read by team_context_v3
    -- to team/season/week, destroying man_rate, zone_rate, coverage_available
    and coverage_source. These must now survive.
    """
    from scripts.research.preserve_week7_opportunity_trace_v1 import (
        _is_opponent_column)
    must_keep = (
        # canonical coverage artifact
        "man_rate", "zone_rate", "coverage_man_rate", "coverage_zone_rate",
        "coverage_available", "coverage_source",
        # box / scheme
        "light_box_rate", "heavy_box_rate", "middle_open_rate",
        # defensive efficiency and the stranded family
        "def_rush_epa", "def_pass_epa", "explosive_play_rate_allowed",
        "success_rate_def", "rush_stuff_rate_x", "ypt_allowed_rb",
        "yards_before_contact_per_rb_rush_x", "pressure_rate_generated",
        # the REALIZED multipliers, without which gate firing is unrecoverable
        "rules_rush_eff_mult", "rules_pass_eff_mult", "rules_volatility_mult",
        "te_target_mult", "rb_rec_target_mult",
        "rules_ypt", "rules_ypc",
    )
    missing = [c for c in must_keep if not _is_opponent_column(c)]
    assert not missing, f"preservation would still drop {missing}"


def test_widening_does_not_admit_market_columns():
    """The deny-list must still run first."""
    from scripts.research.preserve_week7_opportunity_trace_v1 import (
        _is_opponent_column, _is_route_column)
    for col in ("over_odds", "vegas_line", "implied_prob", "model_proj",
                "edge_pct", "fair_odds", "graded_result", "model_sd",
                "payout", "book_price", "clv_pct", "roi_units"):
        assert not _is_opponent_column(col), f"{col} leaked via opponent class"
        assert not _is_route_column(col), f"{col} leaked via route class"
