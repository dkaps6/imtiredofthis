#!/usr/bin/env python3
"""R27B V2 pre-plan source/schema audit for genuinely novel RB receiving context.

This script does NOT fit a model, create a candidate projection, tune a feature,
or score receiving-yard prediction error. It only verifies whether the historical
football sources can reconstruct the proposed target-shape/YAC/checkdown/opponent
feature families with strict season/week identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.pbp import get_pbp

RB_POS = {"RB", "FB", "HB", "TB"}


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _load_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    return _normalize_weekly(_to_pandas(raw), int(season))


def _nonblank(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def audit_season(season: int) -> tuple[dict, pd.DataFrame]:
    p = get_pbp(int(season), min_rows=1).copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    if "season_type" in p.columns:
        reg = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            p = reg

    receiver_name_col = _col(p, ("receiver_player_name", "receiver_name", "receiver"))
    receiver_id_col = _col(p, ("receiver_player_id", "receiver_id"))
    required = ["week", "posteam", "defteam", "complete_pass", "yards_gained", "air_yards", "yards_after_catch"]
    missing_required = [c for c in required if c not in p.columns]
    if receiver_name_col is None and receiver_id_col is None:
        missing_required.append("receiver identity")
    if missing_required:
        return {
            "season": int(season),
            "status": "SCHEMA_MISSING",
            "missing_required": ",".join(missing_required),
            "pbp_rows": int(len(p)),
        }, pd.DataFrame()

    p["season"] = int(season)
    p["week"] = pd.to_numeric(p["week"], errors="coerce")
    p["team"] = p["posteam"].map(canon_team)
    p["opponent"] = p["defteam"].map(canon_team)
    p["receiver_player_id_norm"] = (
        p[receiver_id_col].astype("string").fillna("").str.strip()
        if receiver_id_col else ""
    )
    p["receiver_name_norm"] = (
        p[receiver_name_col].astype("string").fillna("").str.strip()
        if receiver_name_col else ""
    )
    p["receiver_clean_key"] = p["receiver_name_norm"].map(_key)

    targeted = (
        _nonblank(p["receiver_player_id_norm"]) | _nonblank(p["receiver_name_norm"])
    ) & p["week"].notna() & p["team"].ne("")
    t = p.loc[targeted].copy()
    if t.empty:
        return {
            "season": int(season),
            "status": "NO_TARGET_ROWS",
            "missing_required": "",
            "pbp_rows": int(len(p)),
            "target_rows": 0,
        }, pd.DataFrame()

    weekly = _load_weekly(int(season)).copy()
    weekly["week"] = pd.to_numeric(weekly["week"], errors="coerce")
    weekly["team"] = weekly["team"].map(canon_team)
    weekly["player_id_norm"] = weekly.get("player_id", "").astype("string").fillna("").str.strip()
    weekly["player_clean_key"] = weekly.get("player_clean_key", weekly.get("player", "")).astype("string").fillna("").map(_key)
    weekly["position"] = weekly.get("position", "").astype("string").fillna("").str.upper().str.strip()

    id_map = weekly.loc[_nonblank(weekly["player_id_norm"]), ["week", "team", "player_id_norm", "position"]].drop_duplicates(
        ["week", "team", "player_id_norm"]
    )
    name_map = weekly.loc[_nonblank(weekly["player_clean_key"]), ["week", "team", "player_clean_key", "position"]].drop_duplicates(
        ["week", "team", "player_clean_key"]
    )

    t = t.merge(
        id_map.rename(columns={"player_id_norm": "receiver_player_id_norm", "position": "position_by_id"}),
        on=["week", "team", "receiver_player_id_norm"], how="left", validate="many_to_one",
    )
    t = t.merge(
        name_map.rename(columns={"player_clean_key": "receiver_clean_key", "position": "position_by_name"}),
        on=["week", "team", "receiver_clean_key"], how="left", validate="many_to_one",
    )
    t["receiver_position"] = t["position_by_id"].replace("", pd.NA).combine_first(t["position_by_name"])
    t["position_resolved"] = t["receiver_position"].notna().astype(int)
    t["receiver_is_rb"] = t["receiver_position"].fillna("").astype(str).str.upper().isin(RB_POS).astype(int)

    t["complete_num"] = pd.to_numeric(t["complete_pass"], errors="coerce")
    t["air_num"] = pd.to_numeric(t["air_yards"], errors="coerce")
    t["yac_num"] = pd.to_numeric(t["yards_after_catch"], errors="coerce")
    t["yards_num"] = pd.to_numeric(t["yards_gained"], errors="coerce")
    rb = t.loc[t["receiver_is_rb"].eq(1)].copy()
    completed_rb = rb.loc[rb["complete_num"].eq(1)].copy()

    # Team official-pass-attempt denominator is source-audited separately. nflverse
    # pass_attempt can include sacks, so use pass_attempt==1 and sack!=1 when both exist.
    pass_attempt = pd.to_numeric(p.get("pass_attempt", 0), errors="coerce").fillna(0).eq(1)
    sack = pd.to_numeric(p.get("sack", 0), errors="coerce").fillna(0).eq(1)
    official_pass_attempt = pass_attempt & ~sack
    team_pa = p.loc[official_pass_attempt & p["week"].notna() & p["team"].ne("")].groupby(["week", "team"]).size()
    rb_targets = rb.groupby(["week", "team"]).size() if not rb.empty else pd.Series(dtype=float)
    env = pd.concat([team_pa.rename("official_pass_attempts"), rb_targets.rename("rb_targets")], axis=1).fillna(0).reset_index()
    env["rb_targets_per_official_pass_attempt"] = np.where(
        env["official_pass_attempts"].gt(0), env["rb_targets"] / env["official_pass_attempts"], np.nan
    )

    season_row = {
        "season": int(season),
        "status": "PASS_SCHEMA",
        "missing_required": "",
        "pbp_rows": int(len(p)),
        "target_rows": int(len(t)),
        "receiver_position_resolved_rate": float(t["position_resolved"].mean()) if len(t) else np.nan,
        "rb_target_rows": int(len(rb)),
        "rb_completed_rows": int(len(completed_rb)),
        "rb_air_yards_nonnull_rate": float(rb["air_num"].notna().mean()) if len(rb) else np.nan,
        "rb_yac_nonnull_completed_rate": float(completed_rb["yac_num"].notna().mean()) if len(completed_rb) else np.nan,
        "rb_yards_nonnull_rate": float(rb["yards_num"].notna().mean()) if len(rb) else np.nan,
        "rb_screen_derivable_rate": float(rb["air_num"].notna().mean()) if len(rb) else np.nan,
        "rb_explosive20_derivable_rate": float(rb["yards_num"].notna().mean()) if len(rb) else np.nan,
        "team_week_checkdown_rows": int(len(env)),
        "team_week_checkdown_finite_rate": float(env["rb_targets_per_official_pass_attempt"].notna().mean()) if len(env) else np.nan,
        "opponent_rb_context_derivable": int(len(rb) > 0 and rb["opponent"].ne("").any()),
        "receiver_id_field": receiver_id_col or "",
        "receiver_name_field": receiver_name_col or "",
    }
    return season_row, env.assign(season=int(season))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2019-2025")
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/r27b_v2_source_audit"))
    a = ap.parse_args()

    token = a.seasons.strip()
    if "-" in token and "," not in token:
        lo, hi = token.split("-", 1)
        seasons = list(range(int(lo), int(hi) + 1))
    else:
        seasons = [int(x.strip()) for x in token.split(",") if x.strip()]

    rows, envs = [], []
    for season in seasons:
        row, env = audit_season(int(season))
        rows.append(row)
        if not env.empty:
            envs.append(env)
        print("[r27b-v2-source]", json.dumps(row, sort_keys=True))

    summary = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out_dir / "r27b_v2_source_coverage_by_season.csv", index=False)
    if envs:
        pd.concat(envs, ignore_index=True).to_csv(a.out_dir / "r27b_v2_team_checkdown_source_rows.csv", index=False)

    schema_pass = bool(len(summary) == len(seasons) and summary["status"].eq("PASS_SCHEMA").all())
    rb_rows_all = bool((pd.to_numeric(summary.get("rb_target_rows"), errors="coerce").fillna(0) > 0).all())
    report = {
        "audit": "RB_R27B_V2_NOVEL_EFFICIENCY_SOURCE_AUDIT",
        "seasons": seasons,
        "schema_pass_all_seasons": schema_pass,
        "rb_target_rows_present_all_seasons": rb_rows_all,
        "model_fit_performed": False,
        "candidate_projection_created": False,
        "prediction_error_scored": False,
        "sportsbook_inputs_used": 0,
        "purpose": "source/schema/identity coverage only before V2 frozen candidate plan",
    }
    (a.out_dir / "r27b_v2_source_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not schema_pass or not rb_rows_all:
        raise RuntimeError("R27B V2 source audit found missing schema or zero RB target rows; inspect evidence before freezing plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
