#!/usr/bin/env python3
"""Migration 76: QB 40s-MAE information acquisition frontier.

Diagnostic/data-contract only. No predictive model is fit. M76 uses the canonical
v3 football-only stable-QB cohort, quantifies how much attempt/YPA error must be
recovered to enter the 40s, explains the 2024-vs-2025 difficulty split at the
component level, and qualifies genuinely new pregame personnel information before
any M77 predictive experiment is allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 884
EXPECTED_SEASONS = {2024: 444, 2025: 440}

SCENARIOS = [
    ("current", 0.00, 0.00),
    ("att10", 0.10, 0.00), ("att15", 0.15, 0.00), ("att20", 0.20, 0.00),
    ("att25", 0.25, 0.00), ("att30", 0.30, 0.00), ("att35", 0.35, 0.00),
    ("att40", 0.40, 0.00),
    ("ypa10", 0.00, 0.10), ("ypa15", 0.00, 0.15), ("ypa20", 0.00, 0.20),
    ("ypa25", 0.00, 0.25), ("ypa30", 0.00, 0.30), ("ypa40", 0.00, 0.40),
    ("both10_10", 0.10, 0.10), ("both15_15", 0.15, 0.15),
    ("both15_20", 0.15, 0.20), ("both20_15", 0.20, 0.15),
    ("both20_20", 0.20, 0.20), ("both25_25", 0.25, 0.25),
    ("perfect_att", 1.00, 0.00), ("perfect_ypa", 0.00, 1.00),
]

MARKET_TOKENS = ("market", "spread", "moneyline", "sportsbook", "implied_total", "game_total")


def num(x):
    return pd.to_numeric(x, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def to_pd(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def safe_load(fn, *args, **kwargs):
    try:
        return lower(to_pd(fn(*args, **kwargs))), ""
    except Exception as exc:  # source qualification must fail softly
        return pd.DataFrame(), f"{type(exc).__name__}:{exc}"


def metrics(actual, pred) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "tail100": 0}
    err = z.pred - z.actual
    return {
        "n": int(len(z)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(err.mean()),
        "corr": float(z.actual.corr(z.pred)) if len(z) > 2 else np.nan,
        "tail100": int(err.abs().ge(100).sum()),
    }


def require_canonical(base: pd.DataFrame) -> None:
    if len(base) != EXPECTED_ROWS:
        raise RuntimeError(f"M76 expected canonical v3 {EXPECTED_ROWS} rows, got {len(base)}")
    counts = {int(k): int(v) for k, v in num(base.season).value_counts().to_dict().items()}
    if counts != EXPECTED_SEASONS:
        raise RuntimeError(f"M76 canonical v3 season counts drifted: {counts}")
    required = {"season", "week", "team", "opponent", "pred_attempts", "actual_attempts", "pred_pass_yards", "actual_pass_yards"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise RuntimeError(f"M76 canonical v3 missing columns: {missing}")
    bad = [c for c in base.columns if any(tok in c for tok in MARKET_TOKENS)]
    if bad:
        raise RuntimeError(f"M76 football/market boundary violated: {bad}")


def recovered_prediction(base: pd.DataFrame, attempt_recovery: float, ypa_recovery: float) -> pd.Series:
    pa = num(base.pred_attempts)
    aa = num(base.actual_attempts)
    py = num(base.pred_pass_yards)
    actual_yards = num(base.actual_pass_yards)
    pred_ypa = py / pa.replace(0, np.nan)
    actual_ypa = actual_yards / aa.replace(0, np.nan)
    corrected_attempts = pa + attempt_recovery * (aa - pa)
    corrected_ypa = pred_ypa + ypa_recovery * (actual_ypa - pred_ypa)
    return corrected_attempts * corrected_ypa


def recovery_map(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("combined", base)] + [(str(int(s)), g) for s, g in base.groupby("season")]
    for name, ar, yr in SCENARIOS:
        pred = recovered_prediction(base, ar, yr)
        for season_label, g in groups:
            m = metrics(g.actual_pass_yards, pred.loc[g.index])
            rows.append({"scenario": name, "season": season_label, "attempt_recovery": ar, "ypa_recovery": yr, **m})

    # Diagnostic only: minimum equal recovery of both error components needed for <50 MAE.
    for r in np.linspace(0, 1, 101):
        pred = recovered_prediction(base, float(r), float(r))
        m = metrics(base.actual_pass_yards, pred)
        if m["mae"] < 50.0:
            rows.append({"scenario": "min_equal_recovery_below_50", "season": "combined", "attempt_recovery": float(r), "ypa_recovery": float(r), **m})
            break
    return pd.DataFrame(rows)


def difficulty_attribution(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in base.groupby("season"):
        pa, aa = num(g.pred_attempts), num(g.actual_attempts)
        py, ayards = num(g.pred_pass_yards), num(g.actual_pass_yards)
        pypa = py / pa.replace(0, np.nan)
        aypa = ayards / aa.replace(0, np.nan)
        current = metrics(ayards, py)
        perfect_att = metrics(ayards, aa * pypa)
        perfect_ypa = metrics(ayards, pa * aypa)
        rows.append({
            "season": int(season),
            **current,
            "attempt_mae": float((pa - aa).abs().mean()),
            "attempt_bias": float((pa - aa).mean()),
            "attempt_8plus": int((pa - aa).abs().ge(8).sum()),
            "attempt_10plus": int((pa - aa).abs().ge(10).sum()),
            "ypa_mae": float((pypa - aypa).abs().mean()),
            "ypa_bias": float((pypa - aypa).mean()),
            "ypa_1p5plus": int((pypa - aypa).abs().ge(1.5).sum()),
            "ypa_2plus": int((pypa - aypa).abs().ge(2.0).sum()),
            "oracle_attempts_mae": perfect_att["mae"],
            "attempt_oracle_headroom": current["mae"] - perfect_att["mae"],
            "oracle_ypa_mae": perfect_ypa["mae"],
            "ypa_oracle_headroom": current["mae"] - perfect_ypa["mae"],
        })
    out = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    if len(out) == 2:
        a, b = out.iloc[0], out.iloc[1]
        out.loc[:, "mae_gap_vs_other_season"] = [float(a.mae - b.mae), float(b.mae - a.mae)]
    return out


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def season_rows(df: pd.DataFrame, season: int) -> int:
    if df.empty or "season" not in df:
        return 0
    return int(num(df.season).eq(season).sum())


def has_values(df: pd.DataFrame, candidates: list[str]) -> bool:
    c = first_col(df, candidates)
    return bool(c and df[c].notna().any())


def position_group_coverage(df: pd.DataFrame) -> str:
    pos_col = first_col(df, ["position", "pos_abb", "depth_chart_position", "ngs_position"])
    if not pos_col:
        return ""
    vals = set(df[pos_col].dropna().astype(str).str.upper())
    groups = {
        "OL": {"T", "OT", "LT", "RT", "G", "OG", "LG", "RG", "C"},
        "WR_TE_RB": {"WR", "TE", "RB", "FB"},
        "DB": {"CB", "DB", "S", "FS", "SS", "NB"},
        "PASS_RUSH": {"DE", "EDGE", "OLB", "DL", "DT", "NT"},
    }
    return "|".join(k for k, members in groups.items() if vals & members)


def source_contracts() -> pd.DataFrame:
    import nflreadpy as nfl

    depth, e_depth = safe_load(nfl.load_depth_charts, seasons=[2024, 2025, 2026])
    rosters, e_rosters = safe_load(nfl.load_rosters_weekly, seasons=[2024, 2025, 2026])
    snaps, e_snaps = safe_load(nfl.load_snap_counts, seasons=[2023, 2024, 2025, 2026])
    pfr, e_pfr = safe_load(nfl.load_pfr_advstats, seasons=[2023, 2024, 2025, 2026], stat_type="def", summary_level="week")
    injuries, e_inj = safe_load(nfl.load_injuries, seasons=[2024, 2025, 2026])

    depth_time = first_col(depth, ["dt", "week", "date"])
    depth_player = first_col(depth, ["gsis_id", "player_id", "player_name", "full_name"])
    depth_team = first_col(depth, ["team", "club_code"])
    depth_pos = first_col(depth, ["pos_abb", "position", "depth_position", "pos_name"])
    depth_rank = first_col(depth, ["pos_rank", "depth_team", "depth_position"])
    depth_qualified = all([
        season_rows(depth, 2024) > 0,
        season_rows(depth, 2025) > 0,
        season_rows(depth, 2026) > 0,
        depth_time, depth_player, depth_team, depth_pos, depth_rank,
    ])

    roster_week = first_col(rosters, ["week", "dt", "date"])
    roster_player = first_col(rosters, ["gsis_id", "player_id", "full_name"])
    roster_team = first_col(rosters, ["team", "club_code"])
    roster_status = first_col(rosters, ["status", "status_description_abbr"])
    roster_pos = first_col(rosters, ["position", "depth_chart_position", "ngs_position"])
    rosters_qualified = all([
        season_rows(rosters, 2024) > 0,
        season_rows(rosters, 2025) > 0,
        season_rows(rosters, 2026) > 0,
        roster_week, roster_player, roster_team, roster_status, roster_pos,
    ])

    snap_player = first_col(snaps, ["player", "player_id", "pfr_player_id", "gsis_id"])
    snap_team = first_col(snaps, ["team", "team_abbr"])
    snap_week = first_col(snaps, ["week"])
    snap_metric = first_col(snaps, ["offense_snaps", "offense_pct", "defense_snaps", "defense_pct"])
    snaps_qualified = all([
        season_rows(snaps, 2024) > 0,
        season_rows(snaps, 2025) > 0,
        snap_player, snap_team, snap_week, snap_metric,
    ])

    rush_fields = [c for c in [
        "def_pressures", "def_hurries", "def_qb_hits", "def_sacks", "def_blitzes",
        "pressures", "hurries", "qb_hits", "sacks", "blitzes",
    ] if c in pfr.columns]
    pfr_qualified = bool(
        season_rows(pfr, 2024) > 0 and season_rows(pfr, 2025) > 0 and len(set(rush_fields)) >= 2
    )

    injury_historical = season_rows(injuries, 2024) > 0
    injury_post2024 = season_rows(injuries, 2025) > 0 or season_rows(injuries, 2026) > 0

    rows = [
        {
            "source": "nflverse_depth_charts",
            "novel_for_qb": True,
            "historical_2024": season_rows(depth, 2024), "historical_2025": season_rows(depth, 2025), "current_2026": season_rows(depth, 2026),
            "point_in_time_field": depth_time or "", "player_field": depth_player or "", "team_field": depth_team or "", "position_field": depth_pos or "", "rank_status_field": depth_rank or "",
            "position_groups": position_group_coverage(depth),
            "contract_status": "QUALIFIED_EXACT_PERSONNEL" if depth_qualified else "NOT_QUALIFIED",
            "intended_use": "pregame expected starters/depth rank and week-to-week personnel change",
            "source_semantics": "weekly through 2024; append-only ISO timestamp snapshots from 2025 onward",
            "error": e_depth,
        },
        {
            "source": "nflverse_weekly_rosters",
            "novel_for_qb": True,
            "historical_2024": season_rows(rosters, 2024), "historical_2025": season_rows(rosters, 2025), "current_2026": season_rows(rosters, 2026),
            "point_in_time_field": roster_week or "", "player_field": roster_player or "", "team_field": roster_team or "", "position_field": roster_pos or "", "rank_status_field": roster_status or "",
            "position_groups": position_group_coverage(rosters),
            "contract_status": "QUALIFIED_EXACT_PERSONNEL" if rosters_qualified else "NOT_QUALIFIED",
            "intended_use": "active/inactive/reserve/practice-squad player identity around target week",
            "source_semantics": "week-level roster status snapshots",
            "error": e_rosters,
        },
        {
            "source": "pfr_snap_counts",
            "novel_for_qb": True,
            "historical_2024": season_rows(snaps, 2024), "historical_2025": season_rows(snaps, 2025), "current_2026": season_rows(snaps, 2026),
            "point_in_time_field": snap_week or "", "player_field": snap_player or "", "team_field": snap_team or "", "position_field": first_col(snaps, ["position"]) or "", "rank_status_field": snap_metric or "",
            "position_groups": position_group_coverage(snaps),
            "contract_status": "QUALIFIED_STRICTLY_PRIOR_ROLE" if snaps_qualified else "NOT_QUALIFIED",
            "intended_use": "strictly-prior starter/role/continuity baseline; target-game snaps prohibited",
            "source_semantics": "postgame PFR snap counts, safe only when lagged strictly before target game",
            "error": e_snaps,
        },
        {
            "source": "pfr_individual_pass_rush",
            "novel_for_qb": True,
            "historical_2024": season_rows(pfr, 2024), "historical_2025": season_rows(pfr, 2025), "current_2026": season_rows(pfr, 2026),
            "point_in_time_field": first_col(pfr, ["week"]) or "", "player_field": first_col(pfr, ["pfr_player_id", "player_id", "player"]) or "", "team_field": first_col(pfr, ["team"]) or "", "position_field": first_col(pfr, ["position"]) or "", "rank_status_field": "|".join(sorted(set(rush_fields))),
            "position_groups": position_group_coverage(pfr),
            "contract_status": "QUALIFIED_STRICTLY_PRIOR_PASS_RUSH" if pfr_qualified else "SCHEMA_OR_HISTORY_INCOMPLETE",
            "intended_use": "strictly-prior individual pass-rush quality joined to expected defenders",
            "source_semantics": "postgame PFR advanced defense, safe only when lagged strictly before target game",
            "error": e_pfr,
        },
        {
            "source": "nflverse_injuries",
            "novel_for_qb": False,
            "historical_2024": season_rows(injuries, 2024), "historical_2025": season_rows(injuries, 2025), "current_2026": season_rows(injuries, 2026),
            "point_in_time_field": first_col(injuries, ["week", "date", "report_date"]) or "", "player_field": first_col(injuries, ["gsis_id", "player_id", "full_name"]) or "", "team_field": first_col(injuries, ["team"]) or "", "position_field": first_col(injuries, ["position"]) or "", "rank_status_field": first_col(injuries, ["report_status", "practice_status", "status"]) or "",
            "position_groups": position_group_coverage(injuries),
            "contract_status": "BROKEN_AFTER_2024" if injury_historical and not injury_post2024 else "AVAILABLE",
            "intended_use": "supporting exact availability only; generic injury burden already tested in M67",
            "source_semantics": "known nflverse source ended after 2024; cannot authorize M77 by itself",
            "error": e_inj,
        },
    ]
    return pd.DataFrame(rows)


def unresolved_frontiers() -> pd.DataFrame:
    return pd.DataFrame([
        {"frontier": "individual_OL_pass_block_quality_and_assignments", "status": "MISSING_FREE_CONTRACT", "needed_for": "true OL blocker x individual pass-rusher matchup", "revisit_when": "historical+live individual pressure-allowed/pass-block quality and assignment data is acquired"},
        {"frontier": "WR_CB_route_assignment_shadow_share", "status": "MISSING_FREE_CONTRACT", "needed_for": "true receiver x assigned defender matchup beyond M75 aggregate secondary", "revisit_when": "historical+live expected assignment/alignment shares are acquired"},
        {"frontier": "structured_week_specific_gameplan_reporting", "status": "MISSING_STRUCTURED_HISTORY", "needed_for": "why this game deviates from normal DBR/role expectations", "revisit_when": "timestamped pregame coach/beat-report corpus is available"},
        {"frontier": "post_2024_injury_reports", "status": "NFLVERSE_SOURCE_DEAD", "needed_for": "exact target-week practice/game availability", "revisit_when": "replacement historical+live source is established"},
    ])


def no_retest_ledger() -> pd.DataFrame:
    return pd.DataFrame([
        {"family": "pass_rate_DBR_recent_history_state", "migrations": "M40-M42,M64-M65,M73-M74", "rule": "DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family": "aggregate_defense_pressure_EPA_coverage", "migrations": "M45,M56,M69,M72", "rule": "DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family": "opening_script_playcaller", "migrations": "M67-M69,M74", "rule": "DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family": "QB_efficiency_volatility_risk", "migrations": "M70-M71", "rule": "DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family": "receiver_explosive_or_tracking_x_aggregate_secondary", "migrations": "M72,M75", "rule": "DO_NOT_RETEST_WITHOUT_NEW_INFORMATION"},
        {"family": "generic_injury_burden", "migrations": "M67", "rule": "DO_NOT_RETEST_WITHOUT_EXACT_PLAYER_PERSONNEL_IDENTITY"},
        {"family": "new_model_or_subset_on_same_feature_universe", "migrations": "M61,M66+", "rule": "PROHIBITED_AS_STANDALONE_MIGRATION"},
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = lower(pd.read_csv(args.canonical, low_memory=False))
    require_canonical(base)

    rec = recovery_map(base)
    diff = difficulty_attribution(base)
    sources = source_contracts()
    gaps = unresolved_frontiers()
    ledger = no_retest_ledger()

    rec.to_csv(args.out_dir / "m76_recovery_map.csv", index=False)
    diff.to_csv(args.out_dir / "m76_season_difficulty.csv", index=False)
    sources.to_csv(args.out_dir / "m76_source_contracts.csv", index=False)
    gaps.to_csv(args.out_dir / "m76_unresolved_data_frontiers.csv", index=False)
    ledger.to_csv(args.out_dir / "m76_no_retest_ledger.csv", index=False)

    qualified_exact = set(sources.loc[sources.contract_status.eq("QUALIFIED_EXACT_PERSONNEL"), "source"])
    prior_role_ok = bool(sources.contract_status.eq("QUALIFIED_STRICTLY_PRIOR_ROLE").any())
    exact_personnel_gate = {"nflverse_depth_charts", "nflverse_weekly_rosters"}.issubset(qualified_exact) and prior_role_ok

    current = rec[(rec.scenario.eq("current")) & (rec.season.eq("combined"))].iloc[0]
    equal50 = rec[(rec.scenario.eq("min_equal_recovery_below_50")) & (rec.season.eq("combined"))]
    equal_rate = float(equal50.iloc[0].attempt_recovery) if len(equal50) else np.nan
    equal_mae = float(equal50.iloc[0].mae) if len(equal50) else np.nan

    if exact_personnel_gate:
        verdict = "m76_exact_personnel_identity_discontinuity_layer_qualified"
        next_allowed = "M77_exact_personnel_discontinuity"
    else:
        verdict = "m76_exact_personnel_identity_layer_not_yet_qualified"
        next_allowed = "seek_new_personnel_source_before_M77"

    interpretation = pd.DataFrame([{
        "canonical_rows": len(base),
        "baseline_mae": float(current.mae),
        "mae_2024": float(diff.loc[diff.season.eq(2024), "mae"].iloc[0]),
        "mae_2025": float(diff.loc[diff.season.eq(2025), "mae"].iloc[0]),
        "minimum_equal_attempt_ypa_recovery_below_50": equal_rate,
        "mae_at_minimum_equal_recovery": equal_mae,
        "exact_personnel_gate": exact_personnel_gate,
        "qualified_exact_sources": "|".join(sorted(qualified_exact)),
        "m76_interpretation": verdict,
        "next_allowed_migration": next_allowed,
        "predictive_model_fit": False,
        "production_actionable": False,
    }])
    interpretation.to_csv(args.out_dir / "m76_precommitted_interpretation.csv", index=False)

    manifest = {
        "migration": 76,
        "canonical": "qb_frontier_canonical_v3_football_only",
        "expected_rows": EXPECTED_ROWS,
        "expected_seasons": EXPECTED_SEASONS,
        "market_features": False,
        "predictive_models": [],
        "purpose": "40s recovery map + 2024/2025 difficulty attribution + new pregame source qualification + anti-retest ledger",
        "m77_gate": "depth charts + weekly rosters exact personnel contract AND strictly-prior snap role contract",
    }
    (args.out_dir / "m76_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("=== M76 INTERPRETATION ===")
    print(interpretation.to_string(index=False))
    print("\n=== M76 SEASON DIFFICULTY ===")
    print(diff.to_string(index=False))
    print("\n=== M76 SOURCE CONTRACTS ===")
    print(sources.to_string(index=False))
    print("\n=== M76 RECOVERY MAP ===")
    print(rec.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
