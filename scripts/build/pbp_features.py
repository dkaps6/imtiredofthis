#!/usr/bin/env python3
"""Build season-aware NFL PBP feature artifacts in one pass.

This replaces the old family of 2025-hardcoded builders.  Functions are kept
small and deterministic so thin compatibility wrappers can call them directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.pbp import get_pbp

DATA = Path("data")


def _num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _offensive_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    x = pbp.copy()
    x["qb_dropback"] = _num(x, "qb_dropback").astype(int)
    x["rush_attempt"] = _num(x, "rush_attempt").astype(int)
    return x.loc[(x["qb_dropback"].eq(1)) | (x["rush_attempt"].eq(1))].copy()


def build_qb_run_metrics(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = pbp.copy()
    for c in ("qb_scramble", "qb_dropback", "qb_kneel", "rush_attempt"):
        x[c] = _num(x, c).astype(int)
    if "week" not in x.columns:
        raise RuntimeError("PBP missing week")
    passer_col = "passer_player_name" if "passer_player_name" in x.columns else None
    rusher_col = "rusher_player_name" if "rusher_player_name" in x.columns else None
    if passer_col is None:
        raise RuntimeError("PBP missing passer_player_name")
    qb_names = set(x[passer_col].dropna().astype(str))

    def qb_for_row(row):
        if row.get("qb_dropback", 0) == 1 and pd.notna(row.get(passer_col)):
            return row.get(passer_col)
        if rusher_col and row.get("qb_scramble", 0) == 1 and pd.notna(row.get(rusher_col)):
            return row.get(rusher_col)
        if rusher_col and row.get("rush_attempt", 0) == 1 and pd.notna(row.get(rusher_col)):
            name = str(row.get(rusher_col))
            if name in qb_names:
                return name
        return np.nan

    x["qb_name"] = x.apply(qb_for_row, axis=1)
    scrambles = x.loc[x["qb_scramble"].eq(1) & x["qb_name"].notna()].groupby(["qb_name", "week"]).size().rename("scrambles")
    dropbacks = x.loc[x["qb_dropback"].eq(1) & x["qb_name"].notna()].groupby(["qb_name", "week"]).size().rename("dropbacks")
    a = pd.concat([scrambles, dropbacks], axis=1).fillna(0)
    a["scramble_rate"] = np.where(a["dropbacks"] > 0, a["scrambles"] / a["dropbacks"], np.nan)
    a = a.reset_index().rename(columns={"qb_name": "player"})

    qb_rush = x.loc[x["rush_attempt"].eq(1) & x["qb_name"].notna()].copy()
    designed = qb_rush.loc[qb_rush["qb_scramble"].eq(0) & qb_rush["qb_kneel"].eq(0)]
    designed_runs = designed.groupby(["qb_name", "week"]).size().rename("designed_runs")
    qb_rushes = qb_rush.groupby(["qb_name", "week"]).size().rename("qb_rushes")
    snaps = pd.concat([dropbacks, qb_rushes], axis=1).fillna(0)
    snaps["snaps"] = snaps.get("dropbacks", 0) + snaps.get("qb_rushes", 0)
    b = pd.concat([designed_runs, snaps["snaps"]], axis=1).fillna(0)
    b["designed_run_rate"] = np.where(b["snaps"] > 0, b["designed_runs"] / b["snaps"], np.nan)
    b = b.reset_index().rename(columns={"qb_name": "player"})

    combined = a.merge(b, on=["player", "week"], how="outer")
    return a, b, combined


def build_play_volume_splits(pbp: pd.DataFrame) -> pd.DataFrame:
    off = _offensive_plays(pbp)
    required = {"week", "posteam", "defteam"}
    if not required.issubset(off.columns):
        raise RuntimeError(f"PBP missing play-volume columns: {sorted(required - set(off.columns))}")
    off["qtr"] = _num(off, "qtr")
    off["score_differential"] = pd.to_numeric(off.get("score_differential"), errors="coerce")
    rows = []
    for (team, week), g in off.dropna(subset=["posteam", "week"]).groupby(["posteam", "week"]):
        neutral = (g["score_differential"].abs() <= 7) & g["qtr"].isin([1, 2, 3])
        seconds_per_play = np.nan
        if {"game_id", "play_id", "game_seconds_remaining"}.issubset(g.columns):
            d = g.sort_values(["game_id", "play_id"]).copy()
            d["prev"] = d.groupby("game_id")["game_seconds_remaining"].shift(1)
            delta = pd.to_numeric(d["prev"], errors="coerce") - pd.to_numeric(d["game_seconds_remaining"], errors="coerce")
            delta = delta.loc[(delta >= 0) & (delta < 600)]
            seconds_per_play = delta.mean() if not delta.empty else np.nan
        plays_def = int(len(off.loc[(off["defteam"] == team) & (off["week"] == week)]))
        rows.append({
            "team": team, "week": int(week), "plays_offense": len(g), "plays_defense": plays_def,
            "seconds_per_play": seconds_per_play, "neutral_situation_rate": float(neutral.mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def build_run_pass_funnel(pbp: pd.DataFrame) -> pd.DataFrame:
    x = pbp.copy()
    for c in ("pass_attempt", "rush_attempt"):
        x[c] = _num(x, c).astype(int)
    x["epa"] = pd.to_numeric(x.get("epa"), errors="coerce")
    plays = x.loc[x["pass_attempt"].eq(1) | x["rush_attempt"].eq(1)].dropna(subset=["week", "defteam"])
    rows = []
    for week, w in plays.groupby("week"):
        league_run = w.loc[w["rush_attempt"].eq(1), "epa"].mean()
        league_pass = w.loc[w["pass_attempt"].eq(1), "epa"].mean()
        for team, g in w.groupby("defteam"):
            r = g.loc[g["rush_attempt"].eq(1), "epa"].dropna()
            p = g.loc[g["pass_attempt"].eq(1), "epa"].dropna()
            run_epa = r.mean() if not r.empty else np.nan
            pass_epa = p.mean() if not p.empty else np.nan
            rows.append({
                "team": team, "week": int(week),
                "opp_run_lean": run_epa - league_run if pd.notna(run_epa) else np.nan,
                "opp_pass_lean": pass_epa - league_pass if pd.notna(pass_epa) else np.nan,
                "run_success_allowed": float((r > 0).mean()) if not r.empty else np.nan,
                "pass_success_allowed": float((p > 0).mean()) if not p.empty else np.nan,
                "run_epa_allowed": run_epa, "pass_epa_allowed": pass_epa,
            })
    return pd.DataFrame(rows)


def build_script_escalators(pbp: pd.DataFrame) -> pd.DataFrame:
    off = _offensive_plays(pbp).dropna(subset=["posteam", "week"])
    off["qtr"] = _num(off, "qtr")
    off["score_differential"] = pd.to_numeric(off.get("score_differential"), errors="coerce")
    garbage = off["qtr"].eq(4) & off["score_differential"].abs().gt(16)
    off["is_garbage"] = garbage.astype(int)
    off["is_lead"] = (off["score_differential"].gt(7) & ~garbage).astype(int)
    off["is_trail"] = (off["score_differential"].lt(-7) & ~garbage).astype(int)
    off["is_neutral"] = (off["score_differential"].abs().le(7) & off["qtr"].isin([1, 2, 3]) & ~garbage).astype(int)
    out = off.groupby(["posteam", "week"])[["is_lead", "is_trail", "is_neutral", "is_garbage"]].mean().reset_index()
    return out.rename(columns={
        "posteam": "team", "is_lead": "lead_script_pct", "is_trail": "trail_script_pct",
        "is_neutral": "neutral_script_pct", "is_garbage": "garbage_time_pct",
    })


def build_volatility_widening(pbp: pd.DataFrame) -> pd.DataFrame:
    off = _offensive_plays(pbp).dropna(subset=["posteam", "week"])
    off["score_differential"] = pd.to_numeric(off.get("score_differential"), errors="coerce")
    rows = []
    for (team, week), g in off.groupby(["posteam", "week"]):
        pace_std = np.nan
        if {"game_id", "play_id", "game_seconds_remaining"}.issubset(g.columns):
            d = g.sort_values(["game_id", "play_id"]).copy()
            d["prev"] = d.groupby("game_id")["game_seconds_remaining"].shift(1)
            delta = pd.to_numeric(d["prev"], errors="coerce") - pd.to_numeric(d["game_seconds_remaining"], errors="coerce")
            delta = delta.loc[(delta >= 0) & (delta < 600)]
            pace_std = delta.std() if len(delta) > 1 else np.nan
        pass_rate_std = np.nan
        if "drive" in g.columns:
            drive_rates = g.groupby("drive").apply(lambda d: d["qb_dropback"].sum() / max(1, d["qb_dropback"].sum() + d["rush_attempt"].sum()))
            pass_rate_std = drive_rates.std() if len(drive_rates) > 1 else np.nan
        rows.append({
            "team": team, "week": int(week), "pace_std": pace_std,
            "pass_rate_std": pass_rate_std, "score_margin_volatility": g["score_differential"].std(),
        })
    return pd.DataFrame(rows)


def build_coverage_penalties(pbp: pd.DataFrame) -> pd.DataFrame:
    x = pbp.copy()
    x["penalty"] = _num(x, "penalty").astype(int)
    x["penalty_yards"] = _num(x, "penalty_yards")
    x["desc"] = x.get("desc", "").fillna("").astype(str)
    x["penalty_team"] = x.get("penalty_team", "").fillna("").astype(str)
    flags = x.loc[x["penalty"].eq(1)].copy()
    direct = flags.loc[flags["penalty_team"].eq(flags.get("defteam"))].copy()
    text_mask = flags["desc"].str.contains(
        r"defensive\s+holding|defensive\s+pass\s+interference|illegal\s+contact|defensive\s+offside|neutral\s+zone\s+infraction|roughing\s+the\s+passer|illegal\s+hands\s+to\s+the\s+face",
        case=False, regex=True, na=False,
    )
    inferred = flags.loc[~flags.index.isin(direct.index) & text_mask].copy()
    d = pd.concat([direct, inferred], ignore_index=True).dropna(subset=["defteam", "week"])
    d["is_dpi"] = d["desc"].str.contains(r"pass\s+interference", case=False, regex=True, na=False).astype(int)
    d["is_hold"] = d["desc"].str.contains(r"defensive\s+holding", case=False, regex=True, na=False).astype(int)
    return d.groupby(["defteam", "week"]).agg(
        def_penalties=("penalty", "size"), def_penalty_yards=("penalty_yards", "sum"),
        def_dpi_count=("is_dpi", "sum"), def_holding_count=("is_hold", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})


def write_all(season: int) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    pbp = get_pbp(season, min_rows=1)
    if "season_type" in pbp.columns:
        reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            pbp = reg
    print(f"[pbp_features] season={season} rows={len(pbp)}")

    a, b, q = build_qb_run_metrics(pbp)
    a.to_csv(DATA / "qb_scramble_rates.csv", index=False)
    b.to_csv(DATA / "qb_designed_runs.csv", index=False)
    q.to_csv(DATA / "qb_run_metrics.csv", index=False)
    build_play_volume_splits(pbp).to_csv(DATA / "play_volume_splits.csv", index=False)
    build_run_pass_funnel(pbp).to_csv(DATA / "run_pass_funnel.csv", index=False)
    build_script_escalators(pbp).to_csv(DATA / "script_escalators.csv", index=False)
    build_volatility_widening(pbp).to_csv(DATA / "volatility_widening.csv", index=False)
    build_coverage_penalties(pbp).to_csv(DATA / "coverage_penalties.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    write_all(args.season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
