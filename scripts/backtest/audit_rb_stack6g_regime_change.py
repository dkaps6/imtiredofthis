#!/usr/bin/env python3
"""RB STACK6G: source/forensic audit for target-week regime discontinuities.

No fitting. No sportsbook input. No target-game participation or target-game QB rushing
is used upstream. The 2025 P3 team-RB-pool residual is grading truth only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_stack2_enriched_allocation as s2
from scripts.backtest import build_qb_playcaller_opening_leverage as m68

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
RB_POS = {"RB", "HB", "FB"}
QB_POS = {"QB"}
START_WEEK = 6


def num(v):
    return pd.to_numeric(v, errors="coerce")


def lower(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def first(df: pd.DataFrame, names: list[str], default=pd.NA) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def schedule_games(schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    x = lower(schedule)
    x["season"] = num(x.get("season"))
    x["week"] = num(x.get("week"))
    x = x.loc[x.season.eq(season) & x.week.between(1, 18)].copy()
    if "game_type" in x.columns:
        reg = x.game_type.fillna("").astype(str).str.upper().eq("REG")
        if reg.any():
            x = x.loc[reg].copy()
    elif "season_type" in x.columns:
        reg = x.season_type.fillna("").astype(str).str.upper().eq("REG")
        if reg.any():
            x = x.loc[reg].copy()
    gt = x.gametime.astype(str) if "gametime" in x.columns else pd.Series("13:00", index=x.index)
    local = pd.to_datetime(x.gameday.astype(str) + " " + gt, errors="coerce")
    east = ZoneInfo("America/New_York")
    x["kickoff_utc"] = [
        pd.Timestamp(v).tz_localize(east).tz_convert("UTC") if not pd.isna(v) else pd.NaT
        for v in local
    ]
    out = []
    for _, r in x.iterrows():
        ht, at = s2.tm(r.home_team), s2.tm(r.away_team)
        out.append({"season": season, "week": int(r.week), "team": ht, "opponent": at, "kickoff_utc": r.kickoff_utc})
        out.append({"season": season, "week": int(r.week), "team": at, "opponent": ht, "kickoff_utc": r.kickoff_utc})
    return pd.DataFrame(out).drop_duplicates(["season", "week", "team"])


def qbmask(d: pd.DataFrame) -> pd.Series:
    a = first(d, ["pos_abb", "position", "depth_position"]).fillna("").astype(str).str.upper()
    n = first(d, ["pos_name"]).fillna("").astype(str).str.lower()
    g = first(d, ["pos_grp"]).fillna("").astype(str).str.lower()
    return a.eq("QB") | n.str.contains("quarterback", regex=False) | g.str.contains("quarterback", regex=False)


def depth_source_audit(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    coverage_rows = []
    all_asof = []
    for season in SEASONS:
        games = schedule_games(schedule, season)
        try:
            d = lower(s2.pdx(nfl.load_depth_charts(seasons=[season])))
        except Exception as exc:
            coverage_rows.append({
                "season": season, "scheduled_team_games": len(games), "depth_rows": 0,
                "timestamp_column_present": 0, "timestamp_parse_rate": 0.0,
                "qb_depth_rows": 0, "pregame_snapshot_coverage": 0.0, "qb1_coverage": 0.0,
                "median_snapshot_age_hours": np.nan, "p90_snapshot_age_hours": np.nan,
                "timestamp_safe_for_qb1": 0, "load_error": str(exc)[:250],
            })
            continue

        ts_col = "dt" if "dt" in d.columns else "snapshot_datetime" if "snapshot_datetime" in d.columns else None
        qbm = qbmask(d)
        qb_rows = int(qbm.sum())
        parse_rate = 0.0
        timestamp_present = int(ts_col is not None)
        if ts_col is None:
            coverage_rows.append({
                "season": season, "scheduled_team_games": len(games), "depth_rows": len(d),
                "timestamp_column_present": 0, "timestamp_parse_rate": 0.0,
                "qb_depth_rows": qb_rows, "pregame_snapshot_coverage": 0.0, "qb1_coverage": 0.0,
                "median_snapshot_age_hours": np.nan, "p90_snapshot_age_hours": np.nan,
                "timestamp_safe_for_qb1": 0, "load_error": "",
            })
            continue

        d["dt_utc"] = pd.to_datetime(d[ts_col], errors="coerce", utc=True)
        parse_rate = float(d.dt_utc.notna().mean()) if len(d) else 0.0
        d["team"] = first(d, ["team", "club_code", "team_abbr"]).map(s2.tm)
        d["player_name"] = first(d, ["player_name", "full_name", "football_name", "player", "name"]).fillna("").astype(str).str.strip()
        d["name_key"] = d.player_name.map(s2.nk)
        d["gsis_id"] = first(d, ["gsis_id", "player_id"]).fillna("").astype(str)
        d["pos_rank_num"] = num(first(d, ["pos_rank", "depth_team", "rank"]))
        d = d.loc[qbmask(d) & d.dt_utc.notna() & d.team.ne("")].copy()

        rows = []
        for _, g in games.iterrows():
            q = d.loc[d.team.eq(g.team) & d.dt_utc.lt(g.kickoff_utc)].copy()
            if q.empty:
                rows.append({**g.to_dict(), "snapshot_utc": pd.NaT, "snapshot_age_hours": np.nan,
                             "target_qb1": "", "target_qb1_key": "", "target_qb1_id": "", "qb1_rank": np.nan})
                continue
            snap = q.dt_utc.max()
            z = q.loc[q.dt_utc.eq(snap)].copy()
            ranked = z.loc[z.pos_rank_num.notna()].sort_values(["pos_rank_num", "player_name"])
            pick = ranked.iloc[0] if len(ranked) else z.sort_values("player_name").iloc[0]
            rows.append({
                **g.to_dict(), "snapshot_utc": snap,
                "snapshot_age_hours": float((g.kickoff_utc - snap).total_seconds() / 3600.0),
                "target_qb1": str(pick.player_name), "target_qb1_key": str(pick.name_key),
                "target_qb1_id": str(pick.gsis_id), "qb1_rank": float(pick.pos_rank_num) if pd.notna(pick.pos_rank_num) else np.nan,
            })
        a = pd.DataFrame(rows)
        all_asof.append(a)
        snap_cov = float(a.snapshot_utc.notna().mean()) if len(a) else 0.0
        q1_cov = float(a.target_qb1_key.ne("").mean()) if len(a) else 0.0
        safe = int(parse_rate >= 0.90 and snap_cov >= 0.90 and q1_cov >= 0.90)
        coverage_rows.append({
            "season": season, "scheduled_team_games": len(games), "depth_rows": len(d),
            "timestamp_column_present": timestamp_present, "timestamp_parse_rate": parse_rate,
            "qb_depth_rows": len(d), "pregame_snapshot_coverage": snap_cov, "qb1_coverage": q1_cov,
            "median_snapshot_age_hours": float(a.snapshot_age_hours.median()) if len(a) else np.nan,
            "p90_snapshot_age_hours": float(a.snapshot_age_hours.quantile(.90)) if len(a) else np.nan,
            "timestamp_safe_for_qb1": safe, "load_error": "",
        })
    coverage = pd.DataFrame(coverage_rows)
    asof = pd.concat(all_asof, ignore_index=True, sort=False) if all_asof else pd.DataFrame()
    return coverage, asof


def playcaller_table(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    for season in SEASONS:
        g = schedule_games(schedule, season).sort_values(["team", "week"])
        mapped = 0
        changes = 0
        for team, tg in g.groupby("team", sort=True):
            prev = ""
            tenure = 0
            for _, r in tg.sort_values("week").iterrows():
                caller = m68.caller_for(season, int(r.week), team)
                if caller:
                    mapped += 1
                changed = int(bool(prev and caller and caller != prev))
                if not caller:
                    tenure = 0
                elif caller == prev:
                    tenure += 1
                else:
                    tenure = 1
                if changed:
                    changes += 1
                rows.append({
                    "season": season, "week": int(r.week), "team": team,
                    "target_playcaller": caller, "prior_game_playcaller": prev,
                    "playcaller_changed": changed, "playcaller_tenure_games": tenure,
                    "playcaller_recent_change": int(changed or (caller and tenure in (2, 3) and prev == caller)),
                })
                prev = caller
        summary.append({
            "season": season, "team_games": len(g), "mapped_team_games": mapped,
            "mapping_coverage": float(mapped / len(g)) if len(g) else 0.0,
            "documented_change_team_games": changes,
            "mapping_available": int(mapped > 0),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def p3_team_pool(casebook: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    z = casebook.copy()
    z["season"] = num(z.get("season", 2025)).fillna(2025).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(s2.tm)
    if "parent_att" not in z.columns:
        if "enriched_att" not in z.columns:
            raise RuntimeError("STACK6 casebook missing parent_att/enriched_att")
        stack_att = num(z.get("stack_att", pd.Series(np.nan, index=z.index)))
        z["parent_att"] = np.where(z.week.eq(1), stack_att, num(z.enriched_att))
    z["parent_att"] = num(z.parent_att)
    p = z.groupby(["season", "week", "team"], as_index=False).agg(p3_team_rb_pool=("parent_att", "sum"))
    rb = logs.loc[logs.position.isin(RB_POS)].groupby(["season", "week", "team"], as_index=False).agg(actual_team_rb_carries=("rushes", "sum"))
    p = p.merge(rb.loc[rb.season.eq(2025)], on=["season", "week", "team"], how="left", validate="one_to_one")
    p["p3_pool_residual"] = p.p3_team_rb_pool - p.actual_team_rb_carries
    p["p3_pool_abs_residual"] = p.p3_pool_residual.abs()
    return p


def add_qb_history(atlas: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    x = atlas.copy()
    qb = logs.loc[logs.position.isin(QB_POS)].copy()
    team_qb = qb.groupby(["season", "week", "team"], as_index=False).agg(team_qb_rushes=("rushes", "sum"))
    total = logs.groupby(["season", "week", "team"], as_index=False).agg(team_total_rushes=("rushes", "sum"))
    team_qb = team_qb.merge(total, on=["season", "week", "team"], how="left")
    team_qb["team_qb_rush_share"] = np.where(team_qb.team_total_rushes.gt(0), team_qb.team_qb_rushes / team_qb.team_total_rushes, np.nan)
    team_qb["order"] = team_qb.season * 100 + team_qb.week
    qb["order"] = qb.season * 100 + qb.week

    qbp = {k: g.sort_values("order") for k, g in qb.groupby("name_key") if k}
    tp = {k: g.sort_values("order") for k, g in team_qb.groupby("team") if k}
    rows = []
    for _, r in x.iterrows():
        order = int(r.season) * 100 + int(r.week)
        ph = qbp.get(str(r.get("target_qb1_key", "")), pd.DataFrame())
        if len(ph):
            ph = ph.loc[num(ph.order).lt(order)].tail(3)
        th = tp.get(str(r.team), pd.DataFrame())
        if len(th):
            th = th.loc[num(th.order).lt(order)].tail(3)
        p_rush = num(ph.rushes).mean() if len(ph) else np.nan
        t_rush = num(th.team_qb_rushes).mean() if len(th) else np.nan
        t_share = num(th.team_qb_rush_share).mean() if len(th) else np.nan
        rows.append({
            "target_qb_prior_games": int(len(ph)),
            "target_qb_prior3_rushes": float(p_rush) if pd.notna(p_rush) else np.nan,
            "team_prior3_qb_rushes": float(t_rush) if pd.notna(t_rush) else np.nan,
            "team_prior3_qb_rush_share": float(t_share) if pd.notna(t_share) else np.nan,
            "qb_rush_propensity_delta": float(p_rush - t_rush) if pd.notna(p_rush) and pd.notna(t_rush) else np.nan,
        })
    return pd.concat([x.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_qb_change(atlas: pd.DataFrame) -> pd.DataFrame:
    x = atlas.sort_values(["team", "week"]).copy()
    x["prior_game_qb1"] = ""
    x["prior_game_qb1_key"] = ""
    x["qb1_changed"] = np.nan
    for team, idx in x.groupby("team", sort=False).groups.items():
        ids = list(idx)
        prior_name, prior_key = "", ""
        for i in ids:
            cur_name = str(x.at[i, "target_qb1"] or "")
            cur_key = str(x.at[i, "target_qb1_key"] or "")
            x.at[i, "prior_game_qb1"] = prior_name
            x.at[i, "prior_game_qb1_key"] = prior_key
            if cur_key and prior_key:
                x.at[i, "qb1_changed"] = int(cur_key != prior_key)
            prior_name, prior_key = cur_name, cur_key
    return x.sort_values(["week", "team"]).reset_index(drop=True)


def add_bins(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    r = z.p3_pool_residual
    z["POOL_OVER_3"] = r.ge(3).astype(int)
    z["POOL_OVER_5"] = r.ge(5).astype(int)
    z["POOL_UNDER_3"] = r.le(-3).astype(int)
    z["POOL_UNDER_5"] = r.le(-5).astype(int)
    z["POOL_ABS_5"] = r.abs().ge(5).astype(int)
    z["NON_EXTREME_ABS_LT3"] = r.abs().lt(3).astype(int)
    return z


def meanv(g: pd.DataFrame, c: str) -> float:
    v = num(g.get(c, pd.Series(dtype=float)))
    return float(v.mean()) if v.notna().any() else np.nan


def subset_summaries(x: pd.DataFrame) -> pd.DataFrame:
    w = x.loc[x.week.ge(START_WEEK)].copy()
    masks = {
        "ALL_W6_18": pd.Series(True, index=w.index),
        "POOL_OVER_3": w.POOL_OVER_3.eq(1),
        "POOL_OVER_5": w.POOL_OVER_5.eq(1),
        "POOL_UNDER_3": w.POOL_UNDER_3.eq(1),
        "POOL_UNDER_5": w.POOL_UNDER_5.eq(1),
        "POOL_ABS_5": w.POOL_ABS_5.eq(1),
        "NON_EXTREME_ABS_LT3": w.NON_EXTREME_ABS_LT3.eq(1),
    }
    rows = []
    for name, mask in masks.items():
        g = w.loc[mask]
        rows.append({
            "subset": name, "n": len(g),
            "mean_p3_pool_residual": meanv(g, "p3_pool_residual"),
            "mean_abs_p3_pool_residual": meanv(g, "p3_pool_abs_residual"),
            "qb1_coverage": float(g.target_qb1_key.ne("").mean()) if len(g) else np.nan,
            "qb1_change_rate": meanv(g, "qb1_changed"),
            "qb_delta_coverage": float(num(g.qb_rush_propensity_delta).notna().mean()) if len(g) else np.nan,
            "mean_target_qb_prior3_rushes": meanv(g, "target_qb_prior3_rushes"),
            "mean_team_prior3_qb_rushes": meanv(g, "team_prior3_qb_rushes"),
            "mean_qb_rush_propensity_delta": meanv(g, "qb_rush_propensity_delta"),
            "mean_abs_qb_rush_propensity_delta": float(num(g.qb_rush_propensity_delta).abs().mean()) if len(g) else np.nan,
            "playcaller_recent_change_rate": meanv(g, "playcaller_recent_change"),
        })
    return pd.DataFrame(rows)


def gate_metrics(atlas: pd.DataFrame, source: pd.DataFrame, play_summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    w = atlas.loc[atlas.week.ge(START_WEEK)].copy()
    qsrc = source.loc[source.season.eq(2025)]
    q1_cov = float(w.target_qb1_key.ne("").mean()) if len(w) else 0.0
    delta_cov = float(num(w.qb_rush_propensity_delta).notna().mean()) if len(w) else 0.0
    q = w[["p3_pool_residual", "qb_rush_propensity_delta"]].apply(num).dropna()
    corr = float(q.p3_pool_residual.corr(q.qb_rush_propensity_delta)) if len(q) >= 3 else np.nan
    qspread = np.nan
    if len(q) >= 8 and q.qb_rush_propensity_delta.nunique() >= 4:
        q = q.copy()
        q["quartile"] = pd.qcut(q.qb_rush_propensity_delta.rank(method="first"), 4, labels=False)
        qspread = float(q.loc[q.quartile.eq(3), "p3_pool_residual"].mean() - q.loc[q.quartile.eq(0), "p3_pool_residual"].mean())
    over5 = meanv(w.loc[w.POOL_OVER_5.eq(1)], "qb_rush_propensity_delta")
    nonext = meanv(w.loc[w.NON_EXTREME_ABS_LT3.eq(1)], "qb_rush_propensity_delta")
    over5_delta_spread = float(over5 - nonext) if pd.notna(over5) and pd.notna(nonext) else np.nan
    qsafe = int(len(qsrc) == 1 and int(qsrc.iloc[0].timestamp_safe_for_qb1) == 1)
    qb_pass = int(qsafe and q1_cov >= .90 and delta_cov >= .75 and pd.notna(corr) and corr >= .10 and pd.notna(qspread) and qspread >= 1.0 and pd.notna(over5_delta_spread) and over5_delta_spread >= .50)

    p2025 = play_summary.loc[play_summary.season.eq(2025)]
    play_cov = float(p2025.iloc[0].mapping_coverage) if len(p2025) else 0.0
    recent = w.loc[w.playcaller_recent_change.eq(1)]
    stable = w.loc[w.playcaller_recent_change.eq(0)]
    recent_n = len(recent)
    abs_spread = meanv(recent, "p3_pool_abs_residual") - meanv(stable, "p3_pool_abs_residual") if len(recent) and len(stable) else np.nan
    abs5_spread = meanv(recent, "POOL_ABS_5") - meanv(stable, "POOL_ABS_5") if len(recent) and len(stable) else np.nan
    play_pass = int(play_cov >= .95 and recent_n >= 8 and pd.notna(abs_spread) and abs_spread >= 1.0 and pd.notna(abs5_spread) and abs5_spread >= .10)

    if qb_pass and play_pass:
        disposition = "STACK6G_MULTIPLE_REGIME_SIGNALS_SUPPORTED"
    elif qb_pass:
        disposition = "STACK6G_QB_REGIME_SOURCE_AND_FORENSIC_SIGNAL_SUPPORTED"
    elif play_pass:
        disposition = "STACK6G_PLAYCALLER_SOURCE_AND_FORENSIC_SIGNAL_SUPPORTED"
    elif not qsafe and play_cov < .95:
        disposition = "STACK6G_REGIME_SOURCES_NOT_TIMESTAMP_SAFE_OR_INCOMPLETE"
    else:
        disposition = "STACK6G_SOURCE_USABLE_BUT_NO_MATERIAL_FORENSIC_SIGNAL"

    gates = pd.DataFrame([{
        "w6_18_n": len(w), "qb_2025_timestamp_safe": qsafe, "qb1_coverage": q1_cov,
        "qb_delta_coverage": delta_cov, "qb_delta_vs_pool_residual_corr": corr,
        "qb_delta_q4_minus_q1_pool_residual": qspread,
        "qb_delta_pool_over5_minus_nonextreme": over5_delta_spread,
        "qb_gate_pass": qb_pass, "playcaller_mapping_coverage": play_cov,
        "playcaller_recent_change_n": recent_n,
        "playcaller_recent_change_abs_residual_spread": abs_spread,
        "playcaller_recent_change_abs5_rate_spread": abs5_spread,
        "playcaller_gate_pass": play_pass, "disposition": disposition,
    }])
    return gates, disposition


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import nflreadpy as nfl

    schedule = lower(s2.pdx(nfl.load_schedules(seasons=SEASONS)))
    source_cov, asof = depth_source_audit(schedule)
    play, play_summary = playcaller_table(schedule)
    logs = s2.load_weekly_logs(SEASONS)

    casebook = one(a.stack6_root, "stack6_2025_casebook.csv")
    pool = p3_team_pool(casebook, logs)
    a25 = asof.loc[asof.season.eq(2025)].copy() if len(asof) else pd.DataFrame(columns=["season", "week", "team"])
    atlas = pool.merge(a25, on=["season", "week", "team"], how="left", validate="one_to_one")
    atlas = atlas.merge(play.loc[play.season.eq(2025)], on=["season", "week", "team"], how="left", validate="one_to_one")
    for c in ["target_qb1", "target_qb1_key", "target_qb1_id", "target_playcaller", "prior_game_playcaller"]:
        if c in atlas.columns:
            atlas[c] = atlas[c].fillna("").astype(str)
    for c in ["playcaller_changed", "playcaller_recent_change", "playcaller_tenure_games"]:
        if c in atlas.columns:
            atlas[c] = num(atlas[c]).fillna(0)
    atlas = add_qb_change(atlas)
    atlas = add_qb_history(atlas, logs)
    atlas = add_bins(atlas)
    summary = subset_summaries(atlas)
    gates, disposition = gate_metrics(atlas, source_cov, play_summary)

    integrity = pd.DataFrame([{
        "fitted_models": 0, "hyperparameter_search": 0, "feature_search": 0,
        "threshold_search": 0, "sportsbook_inputs": 0,
        "target_game_qb_rushing_used_upstream": 0, "target_game_participation_used_upstream": 0,
        "target_game_injury_used_upstream": 0,
        "p3_pool_residual_used_for_grading_only": 1,
        "source_seasons_attempted": ";".join(map(str, SEASONS)),
    }])
    disposition_df = gates.copy()
    disposition_df["production_change"] = 0
    disposition_df["point_model_authorized"] = 0

    source_cov.to_csv(a.out_dir / "stack6g_source_coverage_by_season.csv", index=False)
    atlas.to_csv(a.out_dir / "stack6g_qb_regime_atlas_2025.csv", index=False)
    summary.to_csv(a.out_dir / "stack6g_qb_regime_summary_2025.csv", index=False)
    play_summary.to_csv(a.out_dir / "stack6g_playcaller_source_summary.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6g_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6g_disposition.csv", index=False)

    print("=== STACK6G source coverage ===")
    print(source_cov.to_string(index=False))
    print("=== STACK6G 2025 forensic summary ===")
    print(summary.to_string(index=False))
    print("=== STACK6G playcaller source ===")
    print(play_summary.to_string(index=False))
    print("=== STACK6G gates/disposition ===")
    print(gates.to_string(index=False))
    print("=== STACK6G integrity ===")
    print(integrity.to_string(index=False))
    print(f"STACK6G_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
