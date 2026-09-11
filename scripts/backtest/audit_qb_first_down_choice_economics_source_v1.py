#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

HISTORY_GAMES = 8
SHRINK_GAMES = 4.0
TARGET_SEASONS = [2023, 2024, 2025]
SOURCE_SEASONS = [2022, 2023, 2024, 2025]
PRIMITIVES = [
    "off_fd_pass_epa", "off_fd_run_epa", "def_fd_pass_epa_allowed", "def_fd_run_epa_allowed",
    "off_fd_pass_success", "off_fd_run_success", "def_fd_pass_success_allowed", "def_fd_run_success_allowed",
]
DIFFS = [
    "off_fd_epa_pass_minus_run", "def_fd_epa_pass_minus_run",
    "off_fd_success_pass_minus_run", "def_fd_success_pass_minus_run",
]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def canon(v):
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def regular_only(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    c = "season_type" if "season_type" in x.columns else ("game_type" if "game_type" in x.columns else None)
    if c:
        s = x[c].fillna("").astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            x = x.loc[keep].copy()
    return x


def to_pd(o):
    if isinstance(o, pd.DataFrame):
        return o.copy()
    if hasattr(o, "to_pandas"):
        return o.to_pandas()
    return pd.DataFrame(o)


def load_schedule_targets() -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl
    rows = []
    audit = {}
    for season in TARGET_SEASONS:
        raw = nfl.load_schedules(int(season))
        s = regular_only(to_pd(raw))
        s.columns = [str(c).strip().lower() for c in s.columns]
        required = {"season", "week", "home_team", "away_team"}
        missing = sorted(required - set(s.columns))
        if missing:
            raise RuntimeError(f"schedule {season} missing {missing}")
        s["season"] = num(s["season"])
        s["week"] = num(s["week"])
        s = s.loc[s.season.eq(season) & s.week.between(1, 18)].copy()
        games = len(s)
        for r in s.itertuples(index=False):
            home, away = canon(getattr(r, "home_team")), canon(getattr(r, "away_team"))
            if not home or not away:
                raise RuntimeError(f"blank schedule team {season} W{getattr(r,'week')}")
            rows.append({"season": season, "week": int(getattr(r, "week")), "team": home, "opponent": away})
            rows.append({"season": season, "week": int(getattr(r, "week")), "team": away, "opponent": home})
        audit[str(season)] = {"games": int(games), "expected_team_weeks": int(2 * games)}
    out = pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate schedule target team-week")
    out["ord"] = out.season.astype(int) * 100 + out.week.astype(int)
    return out, audit


def load_first_down_pbp() -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl
    frames = []
    audit = {}
    for season in SOURCE_SEASONS:
        raw = nfl.load_pbp(seasons=[int(season)])
        p = regular_only(to_pd(raw))
        p.columns = [str(c).strip().lower() for c in p.columns]
        required = {"week", "posteam", "defteam", "down", "qb_dropback", "rush_attempt", "epa", "success"}
        missing = sorted(required - set(p.columns))
        if missing:
            raise RuntimeError(f"PBP {season} missing {missing}")
        p["season"] = season
        for c in ["week", "down", "qb_dropback", "rush_attempt", "epa", "success"]:
            p[c] = num(p[c])
        p["posteam"] = p.posteam.fillna("").astype(str).map(canon)
        p["defteam"] = p.defteam.fillna("").astype(str).map(canon)
        two = num(p["two_point_attempt"]).fillna(0).eq(1) if "two_point_attempt" in p else pd.Series(False, index=p.index)
        no_play = num(p["no_play"]).fillna(0).eq(1) if "no_play" in p else pd.Series(False, index=p.index)
        kneel = num(p["qb_kneel"]).fillna(0).eq(1) if "qb_kneel" in p else pd.Series(False, index=p.index)
        pass_origin = p.qb_dropback.fillna(0).eq(1)
        designed_run = p.rush_attempt.fillna(0).eq(1) & ~pass_origin & ~kneel
        eligible = (
            p.week.between(1, 18) & p.down.eq(1) & p.posteam.ne("") & p.defteam.ne("")
            & (pass_origin | designed_run) & ~two & ~no_play
        )
        q = p.loc[eligible, ["season", "week", "posteam", "defteam", "epa", "success"]].copy()
        q["pass_origin"] = pass_origin.loc[eligible].astype(int).to_numpy()
        q["designed_run"] = designed_run.loc[eligible].astype(int).to_numpy()
        q["choice_sum"] = q.pass_origin + q.designed_run
        if not q.choice_sum.eq(1).all():
            raise RuntimeError(f"choice exclusivity failure season={season}")
        audit[str(season)] = {
            "eligible_first_down_plays": int(len(q)),
            "epa_nonnull_rate": float(q.epa.notna().mean()),
            "success_nonnull_rate": float(q.success.notna().mean()),
            "pass_origin_plays": int(q.pass_origin.sum()),
            "designed_run_plays": int(q.designed_run.sum()),
        }
        frames.append(q)
    return pd.concat(frames, ignore_index=True), audit


def game_observations(p: pd.DataFrame) -> pd.DataFrame:
    x = p.copy()
    x["pass_epa_sum"] = np.where(x.pass_origin.eq(1) & x.epa.notna(), x.epa, 0.0)
    x["run_epa_sum"] = np.where(x.designed_run.eq(1) & x.epa.notna(), x.epa, 0.0)
    x["pass_success_sum"] = np.where(x.pass_origin.eq(1) & x.success.notna(), x.success, 0.0)
    x["run_success_sum"] = np.where(x.designed_run.eq(1) & x.success.notna(), x.success, 0.0)
    x["pass_epa_valid"] = (x.pass_origin.eq(1) & x.epa.notna()).astype(int)
    x["run_epa_valid"] = (x.designed_run.eq(1) & x.epa.notna()).astype(int)
    x["pass_success_valid"] = (x.pass_origin.eq(1) & x.success.notna()).astype(int)
    x["run_success_valid"] = (x.designed_run.eq(1) & x.success.notna()).astype(int)
    g = x.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        pass_n=("pass_origin", "sum"), run_n=("designed_run", "sum"),
        pass_epa_sum=("pass_epa_sum", "sum"), run_epa_sum=("run_epa_sum", "sum"),
        pass_epa_valid=("pass_epa_valid", "sum"), run_epa_valid=("run_epa_valid", "sum"),
        pass_success_sum=("pass_success_sum", "sum"), run_success_sum=("run_success_sum", "sum"),
        pass_success_valid=("pass_success_valid", "sum"), run_success_valid=("run_success_valid", "sum"),
    ).rename(columns={"posteam": "team", "defteam": "opponent"})
    g["team"] = g.team.map(canon)
    g["opponent"] = g.opponent.map(canon)
    g["ord"] = g.season.astype(int) * 100 + g.week.astype(int)
    if g.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate first-down game observation")
    return g.sort_values(["ord", "team"]).reset_index(drop=True)


def weighted_value(g: pd.DataFrame, sum_col: str, valid_col: str) -> float:
    den = float(num(g[valid_col]).sum())
    return float(num(g[sum_col]).sum() / den) if den > 0 else np.nan


def prior_league(games: pd.DataFrame, target_ord: int, channel: str, metric: str) -> tuple[float, float]:
    h = games.loc[games.ord < target_ord]
    if h.empty:
        return np.nan, np.nan
    ncol = f"{channel}_n"
    if metric == "epa":
        value = weighted_value(h, f"{channel}_epa_sum", f"{channel}_epa_valid")
    else:
        value = weighted_value(h, f"{channel}_success_sum", f"{channel}_success_valid")
    mean_count = float(num(h[ncol]).sum() / len(h)) if len(h) else np.nan
    return value, mean_count


def hist_value(h: pd.DataFrame, channel: str, metric: str) -> tuple[float, float]:
    n = float(num(h[f"{channel}_n"]).sum())
    if metric == "epa":
        v = weighted_value(h, f"{channel}_epa_sum", f"{channel}_epa_valid")
    else:
        v = weighted_value(h, f"{channel}_success_sum", f"{channel}_success_valid")
    return v, n


def shrink(history_value: float, history_n: float, league_value: float, league_mean_count: float) -> float:
    pseudo = SHRINK_GAMES * league_mean_count
    if not np.isfinite(league_value) or not np.isfinite(pseudo) or pseudo <= 0:
        return np.nan
    if not np.isfinite(history_value) or history_n <= 0:
        return float(league_value)
    return float((history_n * history_value + pseudo * league_value) / (history_n + pseudo))


def build_target_source(targets: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in targets.itertuples(index=False):
        target_ord = int(r.ord)
        off = games.loc[(games.team.eq(r.team)) & (games.ord < target_ord)].sort_values("ord").tail(HISTORY_GAMES)
        deff = games.loc[(games.opponent.eq(r.opponent)) & (games.ord < target_ord)].sort_values("ord").tail(HISTORY_GAMES)
        rec = {
            "season": int(r.season), "week": int(r.week), "team": r.team, "opponent": r.opponent,
            "target_ord": target_ord,
            "off_prior_games": int(len(off)), "def_prior_games": int(len(deff)),
            "off_prior_max_ord": int(off.ord.max()) if len(off) else np.nan,
            "def_prior_max_ord": int(deff.ord.max()) if len(deff) else np.nan,
            "off_prior_pass_n": int(off.pass_n.sum()) if len(off) else 0,
            "off_prior_run_n": int(off.run_n.sum()) if len(off) else 0,
            "def_prior_pass_n": int(deff.pass_n.sum()) if len(deff) else 0,
            "def_prior_run_n": int(deff.run_n.sum()) if len(deff) else 0,
        }
        for channel in ["pass", "run"]:
            for metric in ["epa", "success"]:
                lv, lcount = prior_league(games, target_ord, channel, metric)
                ov, on = hist_value(off, channel, metric)
                dv, dn = hist_value(deff, channel, metric)
                rec[f"off_fd_{channel}_{metric}"] = shrink(ov, on, lv, lcount)
                rec[f"def_fd_{channel}_{metric}_allowed"] = shrink(dv, dn, lv, lcount)
        rec["off_fd_epa_pass_minus_run"] = rec["off_fd_pass_epa"] - rec["off_fd_run_epa"]
        rec["def_fd_epa_pass_minus_run"] = rec["def_fd_pass_epa_allowed"] - rec["def_fd_run_epa_allowed"]
        rec["off_fd_success_pass_minus_run"] = rec["off_fd_pass_success"] - rec["off_fd_run_success"]
        rec["def_fd_success_pass_minus_run"] = rec["def_fd_pass_success_allowed"] - rec["def_fd_run_success_allowed"]
        rows.append(rec)
    return pd.DataFrame(rows)


def qtile(s, q):
    x = num(s).dropna()
    return float(x.quantile(q)) if len(x) else np.nan


def source_summary(z: pd.DataFrame, pbp_audit: dict) -> pd.DataFrame:
    rows = []
    views = [(str(y), z.loc[z.season.eq(y)]) for y in TARGET_SEASONS] + [("POOLED_2023_2025", z)]
    for label, g in views:
        rec = {
            "season": label, "target_team_weeks": int(len(g)),
            "all_primitives_finite_rate": float(g[PRIMITIVES].notna().all(axis=1).mean()),
            "all_differences_finite_rate": float(g[DIFFS].notna().all(axis=1).mean()),
            "min_primitive_coverage": float(min(g[c].notna().mean() for c in PRIMITIVES)),
            "min_difference_coverage": float(min(g[c].notna().mean() for c in DIFFS)),
            "off_prior_games_median": qtile(g.off_prior_games, .50), "off_prior_games_p10": qtile(g.off_prior_games, .10),
            "def_prior_games_median": qtile(g.def_prior_games, .50), "def_prior_games_p10": qtile(g.def_prior_games, .10),
            "off_prior_pass_n_median": qtile(g.off_prior_pass_n, .50), "off_prior_pass_n_p10": qtile(g.off_prior_pass_n, .10),
            "off_prior_run_n_median": qtile(g.off_prior_run_n, .50), "off_prior_run_n_p10": qtile(g.off_prior_run_n, .10),
            "def_prior_pass_n_median": qtile(g.def_prior_pass_n, .50), "def_prior_pass_n_p10": qtile(g.def_prior_pass_n, .10),
            "def_prior_run_n_median": qtile(g.def_prior_run_n, .50), "def_prior_run_n_p10": qtile(g.def_prior_run_n, .10),
            "share_off_prior_games_ge4": float(g.off_prior_games.ge(4).mean()),
            "share_def_prior_games_ge4": float(g.def_prior_games.ge(4).mean()),
            "share_off_pass_n_ge40": float(g.off_prior_pass_n.ge(40).mean()),
            "share_off_run_n_ge30": float(g.off_prior_run_n.ge(30).mean()),
            "share_def_pass_n_ge40": float(g.def_prior_pass_n.ge(40).mean()),
            "share_def_run_n_ge30": float(g.def_prior_run_n.ge(30).mean()),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    targets, schedule_audit = load_schedule_targets()
    pbp, pbp_audit = load_first_down_pbp()
    games = game_observations(pbp)
    z = build_target_source(targets, games)
    summary = source_summary(z, pbp_audit)

    target_expected = {int(y): int(schedule_audit[str(y)]["expected_team_weeks"]) for y in TARGET_SEASONS}
    target_actual = {int(y): int(z.loc[z.season.eq(y)].shape[0]) for y in TARGET_SEASONS}
    primitive_cov = {int(y): float(min(z.loc[z.season.eq(y), c].notna().mean() for c in PRIMITIVES)) for y in TARGET_SEASONS}
    diff_cov = {int(y): float(min(z.loc[z.season.eq(y), c].notna().mean() for c in DIFFS)) for y in TARGET_SEASONS}
    pooled_primitive = float(min(z[c].notna().mean() for c in PRIMITIVES))
    pooled_diff = float(min(z[c].notna().mean() for c in DIFFS))
    all_pbp = pd.concat([
        pd.DataFrame([{"epa": v["epa_nonnull_rate"], "success": v["success_nonnull_rate"], "n": v["eligible_first_down_plays"]}])
        for v in pbp_audit.values()
    ], ignore_index=True)
    total_n = float(all_pbp.n.sum())
    epa_cov = float(np.average(all_pbp.epa, weights=all_pbp.n)) if total_n else np.nan
    success_cov = float(np.average(all_pbp.success, weights=all_pbp.n)) if total_n else np.nan
    pooled = summary.loc[summary.season.eq("POOLED_2023_2025")].iloc[0]
    strict_prior = bool(((num(z.off_prior_max_ord).isna()) | (num(z.off_prior_max_ord) < z.target_ord)).all() and ((num(z.def_prior_max_ord).isna()) | (num(z.def_prior_max_ord) < z.target_ord)).all())
    mutually_exclusive = bool(pbp.choice_sum.eq(1).all())

    gates = {
        "exact_expected_target_team_weeks_each_season": target_actual == target_expected,
        "no_duplicate_target_keys": not z.duplicated(["season", "week", "team"]).any(),
        "primitive_coverage_pooled_ge_0_99": pooled_primitive >= .99,
        "primitive_coverage_each_season_ge_0_98": all(v >= .98 for v in primitive_cov.values()),
        "difference_coverage_pooled_ge_0_99": pooled_diff >= .99,
        "difference_coverage_each_season_ge_0_98": all(v >= .98 for v in diff_cov.values()),
        "share_off_prior_games_ge4_ge_0_95": float(pooled.share_off_prior_games_ge4) >= .95,
        "share_def_prior_games_ge4_ge_0_95": float(pooled.share_def_prior_games_ge4) >= .95,
        "share_off_pass_n_ge40_ge_0_95": float(pooled.share_off_pass_n_ge40) >= .95,
        "share_off_run_n_ge30_ge_0_95": float(pooled.share_off_run_n_ge30) >= .95,
        "share_def_pass_n_ge40_ge_0_95": float(pooled.share_def_pass_n_ge40) >= .95,
        "share_def_run_n_ge30_ge_0_95": float(pooled.share_def_run_n_ge30) >= .95,
        "epa_nonnull_ge_0_99": epa_cov >= .99,
        "success_nonnull_ge_0_99": success_cov >= .99,
        "pass_origin_and_designed_run_mutually_exclusive": mutually_exclusive,
        "all_target_sources_strictly_prior": strict_prior,
        "zero_sportsbook_inputs": True,
        "zero_parent_or_qb_wr_residual_reads": True,
        "zero_model_fitting": True,
        "zero_production_changes": True,
    }
    integrity = all([gates["exact_expected_target_team_weeks_each_season"], gates["no_duplicate_target_keys"], gates["pass_origin_and_designed_run_mutually_exclusive"], gates["all_target_sources_strictly_prior"]])
    qualified = all(gates.values())
    disposition = "FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_QUALIFIED" if qualified else ("FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_BLOCKED" if integrity else "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE")

    deployment = {
        "schedule_source_available_2023_2025": True,
        "pbp_source_contract_supports_completed_game_updates": True,
        "required_fields_share_historical_live_nflverse_pbp_contract": True,
        "sportsbook_required": False,
    }
    result = {
        "migration": "QB_FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_V1",
        "disposition": disposition,
        "production_actionable": False,
        "source_qualified_for_one_later_predictive_screen": bool(qualified),
        "target_expected": target_expected,
        "target_actual": target_actual,
        "primitive_min_coverage_by_season": primitive_cov,
        "difference_min_coverage_by_season": diff_cov,
        "pooled_min_primitive_coverage": pooled_primitive,
        "pooled_min_difference_coverage": pooled_diff,
        "epa_nonnull_eligible_source_plays": epa_cov,
        "success_nonnull_eligible_source_plays": success_cov,
        "gates": gates,
        "schedule_audit": schedule_audit,
        "pbp_audit": pbp_audit,
        "deployment_feasibility": deployment,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "first_down_choice_economics_source_rows.csv", index=False)
    summary.to_csv(a.out_dir / "first_down_choice_economics_source_summary.csv", index=False)
    pd.DataFrame([{"primitive": c, "pooled_coverage": float(z[c].notna().mean()), **{f"coverage_{y}": float(z.loc[z.season.eq(y), c].notna().mean()) for y in TARGET_SEASONS}} for c in PRIMITIVES + DIFFS]).to_csv(a.out_dir / "first_down_choice_economics_field_coverage.csv", index=False)
    (a.out_dir / "first_down_choice_economics_source_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
