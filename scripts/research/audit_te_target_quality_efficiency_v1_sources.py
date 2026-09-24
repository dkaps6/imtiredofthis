"""Source/novelty audit for TE receiving target-quality information.

This phase intentionally does not fit or score any predictive model. It audits
whether strict-prior NGS receiving and nflverse PBP target-quality fields are
available densely enough to justify a separately frozen TE efficiency study.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.pbp import get_pbp

TE_POS = {"TE"}
CORE_NGS = [
    "avg_separation",
    "avg_cushion",
    "avg_intended_air_yards",
    "avg_expected_yac",
    "avg_yac_above_expectation",
]
PBP_TARGET_FIELDS = [
    "air_yards",
    "pass_length",
    "pass_location",
    "down",
    "ydstogo",
    "shotgun",
    "no_huddle",
    "score_differential",
]
PBP_CATCH_FIELDS = [
    "yards_after_catch",
    "xyac_mean_yardage",
    "xyac_success",
    "xyac_fd",
]


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def nonblank(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip().ne("")


def key_name(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def load_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    q = _normalize_weekly(_to_pandas(raw), int(season)).copy()
    q["week"] = num(q["week"])
    q["team"] = q["team"].map(canon_team)
    q["player_id_norm"] = q.get("player_id", "").astype("string").fillna("").str.strip()
    q["player_clean_key"] = (
        q.get("player_clean_key", q.get("player", ""))
        .astype("string").fillna("").map(key_name)
    )
    q["position"] = q.get("position", "").astype("string").fillna("").str.upper().str.strip()
    return q


def resolve_pbp_te(season: int) -> tuple[pd.DataFrame, dict]:
    p = lower(get_pbp(int(season), min_rows=1).copy())
    if "season_type" in p.columns:
        reg = p[p["season_type"].fillna("").astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            p = reg
    p["season"] = int(season)
    p["week"] = num(p.get("week"))
    p["team"] = p.get("posteam", "").map(canon_team)

    name_col = first_col(p, ["receiver_player_name", "receiver_name", "receiver"])
    id_col = first_col(p, ["receiver_player_id", "receiver_id"])
    if name_col is None and id_col is None:
        raise RuntimeError(f"{season}: receiver identity unavailable")

    p["receiver_id_norm"] = (
        p[id_col].astype("string").fillna("").str.strip() if id_col else ""
    )
    p["receiver_name_norm"] = (
        p[name_col].astype("string").fillna("").str.strip() if name_col else ""
    )
    p["receiver_name_key"] = p["receiver_name_norm"].map(key_name)

    targeted = (
        (nonblank(p["receiver_id_norm"]) | nonblank(p["receiver_name_norm"]))
        & p["week"].notna()
        & p["team"].ne("")
    )
    t = p[targeted].copy()

    weekly = load_weekly(int(season))
    id_map = (
        weekly.loc[
            nonblank(weekly["player_id_norm"]),
            ["week", "team", "player_id_norm", "player_clean_key", "position"],
        ]
        .drop_duplicates(["week", "team", "player_id_norm"])
        .rename(
            columns={
                "player_id_norm": "receiver_id_norm",
                "player_clean_key": "id_player_key",
                "position": "position_by_id",
            }
        )
    )
    name_map = (
        weekly.loc[
            nonblank(weekly["player_clean_key"]),
            ["week", "team", "player_clean_key", "position"],
        ]
        .drop_duplicates(["week", "team", "player_clean_key"])
        .rename(
            columns={
                "player_clean_key": "receiver_name_key",
                "position": "position_by_name",
            }
        )
    )

    t = t.merge(
        id_map,
        on=["week", "team", "receiver_id_norm"],
        how="left",
        validate="many_to_one",
    )
    t = t.merge(
        name_map,
        on=["week", "team", "receiver_name_key"],
        how="left",
        validate="many_to_one",
    )
    t["player_clean_key"] = (
        t["id_player_key"].replace("", pd.NA)
        .combine_first(t["receiver_name_key"].replace("", pd.NA))
        .fillna("")
    )
    t["receiver_position"] = (
        t["position_by_id"].replace("", pd.NA)
        .combine_first(t["position_by_name"])
        .fillna("")
    )

    te = t[
        t["receiver_position"].astype(str).str.upper().isin(TE_POS)
        & t["player_clean_key"].ne("")
    ].copy()

    complete_col = first_col(te, ["complete_pass"])
    te["complete_num"] = num(te[complete_col]).fillna(0) if complete_col else 0.0
    completed = te["complete_num"].eq(1)

    aliases = {
        "air_yards": ["air_yards"],
        "pass_length": ["pass_length"],
        "pass_location": ["pass_location"],
        "down": ["down"],
        "ydstogo": ["ydstogo", "yards_to_go"],
        "shotgun": ["shotgun"],
        "no_huddle": ["no_huddle"],
        "score_differential": ["score_differential", "posteam_score_differential"],
        "yards_after_catch": ["yards_after_catch"],
        "xyac_mean_yardage": ["xyac_mean_yardage", "xyac_mean_yards"],
        "xyac_success": ["xyac_success"],
        "xyac_fd": ["xyac_fd"],
    }

    audit = {
        "season": int(season),
        "pbp_rows": int(len(p)),
        "target_rows": int(len(t)),
        "te_target_rows": int(len(te)),
        "te_completed_targets": int(completed.sum()),
        "receiver_position_resolved_rate": (
            float(t["receiver_position"].ne("").mean()) if len(t) else 0.0
        ),
    }
    for label in PBP_TARGET_FIELDS:
        col = first_col(te, aliases[label])
        audit[f"{label}_field"] = col or ""
        audit[f"{label}_nonnull_target_rate"] = (
            float(te[col].notna().mean()) if col and len(te) else 0.0
        )
    for label in PBP_CATCH_FIELDS:
        col = first_col(te, aliases[label])
        audit[f"{label}_field"] = col or ""
        audit[f"{label}_nonnull_completed_rate"] = (
            float(te.loc[completed, col].notna().mean())
            if col and completed.any()
            else 0.0
        )
    return te, audit


def resolve_ngs_te(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import nflreadpy as nfl

    try:
        raw = nfl.load_nextgen_stats(seasons=seasons, stat_type="receiving")
        ngs = lower(_to_pandas(raw))
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), {
            "load_error": f"{type(exc).__name__}:{exc}"
        }

    season_col = first_col(ngs, ["season"])
    week_col = first_col(ngs, ["week"])
    team_col = first_col(ngs, ["team_abbr", "team"])
    id_col = first_col(ngs, ["player_gsis_id", "gsis_id", "player_id"])
    name_col = first_col(ngs, ["player_display_name", "player_name", "display_name"])
    pos_col = first_col(ngs, ["player_position", "position", "position_group"])
    if not all([season_col, week_col, team_col]):
        return pd.DataFrame(), pd.DataFrame(), {
            "load_error": "missing_required_identity_columns",
            "columns": "|".join(ngs.columns),
        }

    if "season_type" in ngs.columns:
        reg = ngs[ngs["season_type"].fillna("").astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            ngs = reg

    ngs["season"] = num(ngs[season_col]).astype("Int64")
    ngs["week"] = num(ngs[week_col]).astype("Int64")
    ngs["team"] = ngs[team_col].map(canon_team)
    ngs["player_id_norm"] = (
        ngs[id_col].astype("string").fillna("").str.strip() if id_col else ""
    )
    ngs["player_name_key"] = (
        ngs[name_col].astype("string").fillna("").map(key_name) if name_col else ""
    )
    ngs["position_resolved"] = (
        ngs[pos_col].astype("string").fillna("").str.upper().str.strip()
        if pos_col
        else ""
    )

    try:
        players = lower(_to_pandas(nfl.load_players()))
    except Exception:
        players = pd.DataFrame()
    if not players.empty:
        pid = first_col(players, ["gsis_id", "player_gsis_id", "player_id"])
        ppos = first_col(players, ["position", "position_group"])
        pname = first_col(players, ["display_name", "full_name", "player_name", "football_name"])
        if pid:
            cols = [pid] + ([ppos] if ppos else []) + ([pname] if pname else [])
            bridge = players[cols].dropna(subset=[pid]).drop_duplicates(pid).copy()
            bridge["player_id_norm"] = bridge[pid].astype("string").str.strip()
            bridge["bridge_position"] = (
                bridge[ppos].astype("string").fillna("").str.upper().str.strip()
                if ppos
                else ""
            )
            bridge["bridge_name_key"] = (
                bridge[pname].astype("string").fillna("").map(key_name)
                if pname
                else ""
            )
            bridge = bridge[
                ["player_id_norm", "bridge_position", "bridge_name_key"]
            ]
            ngs = ngs.merge(
                bridge, on="player_id_norm", how="left", validate="many_to_one"
            )
            ngs["position_resolved"] = (
                ngs["position_resolved"].replace("", pd.NA)
                .combine_first(ngs["bridge_position"])
                .fillna("")
            )
            ngs["player_name_key"] = (
                ngs["player_name_key"].replace("", pd.NA)
                .combine_first(ngs["bridge_name_key"])
                .fillna("")
            )

    te = ngs[
        ngs["position_resolved"].isin(TE_POS)
        & ngs["week"].notna()
        & ngs["season"].notna()
        & ngs["team"].ne("")
    ].copy()
    te["player_clean_key"] = te["player_name_key"].astype(str)
    te = te[te["player_clean_key"].ne("")].copy()

    field_aliases = {
        "targets": ["targets"],
        "receptions": ["receptions"],
        "avg_separation": ["avg_separation"],
        "avg_cushion": ["avg_cushion"],
        "avg_intended_air_yards": ["avg_intended_air_yards", "avg_air_distance"],
        "avg_expected_yac": ["avg_expected_yac"],
        "avg_yac_above_expectation": ["avg_yac_above_expectation"],
        "percent_share_of_intended_air_yards": ["percent_share_of_intended_air_yards"],
    }

    rows = []
    for season in seasons:
        s = te[te["season"].eq(int(season))].copy()
        rec = {
            "season": int(season),
            "ngs_receiving_rows": int(ngs["season"].eq(int(season)).sum()),
            "ngs_te_rows": int(len(s)),
            "ngs_te_unique_players": int(s["player_clean_key"].nunique()),
            "ngs_te_unique_player_weeks": int(
                s[["season", "week", "player_clean_key"]].drop_duplicates().shape[0]
            ),
        }
        for label, names in field_aliases.items():
            col = first_col(s, names)
            rec[f"{label}_field"] = col or ""
            rec[f"{label}_nonnull_rate"] = (
                float(s[col].notna().mean()) if col and len(s) else 0.0
            )
        rows.append(rec)
    return te, pd.DataFrame(rows), {"load_error": ""}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2020,2021,2022,2023,2024,2025,2026")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pbp_audits = []
    pbp_frames = []
    pbp_errors = []
    for season in seasons:
        try:
            te, audit = resolve_pbp_te(season)
            pbp_frames.append(te)
            pbp_audits.append(audit)
        except Exception as exc:
            pbp_errors.append({"season": season, "error": f"{type(exc).__name__}:{exc}"})
            pbp_audits.append({"season": season, "te_target_rows": 0, "te_completed_targets": 0})

    pbp_summary = pd.DataFrame(pbp_audits)
    pbp_summary.to_csv(out / "pbp_te_target_quality_coverage.csv", index=False)
    pd.DataFrame(pbp_errors).to_csv(out / "pbp_errors.csv", index=False)

    ngs_te, ngs_summary, ngs_meta = resolve_ngs_te(seasons)
    ngs_summary.to_csv(out / "ngs_te_receiving_coverage.csv", index=False)

    pbp_all = pd.concat(pbp_frames, ignore_index=True, sort=False) if pbp_frames else pd.DataFrame()
    pbp_2026_prior = pbp_all[
        pd.to_numeric(pbp_all.get("season"), errors="coerce").eq(2026)
        & pd.to_numeric(pbp_all.get("week"), errors="coerce").le(2)
    ].copy() if not pbp_all.empty else pd.DataFrame()
    ngs_2026_prior = ngs_te[
        pd.to_numeric(ngs_te.get("season"), errors="coerce").eq(2026)
        & pd.to_numeric(ngs_te.get("week"), errors="coerce").le(2)
    ].copy() if not ngs_te.empty else pd.DataFrame()

    hist_pbp = pbp_summary[pbp_summary["season"].between(2020, 2025)].copy()
    pbp_air_ok = (
        len(hist_pbp)
        and "air_yards_nonnull_target_rate" in hist_pbp
        and hist_pbp["air_yards_nonnull_target_rate"].fillna(0).ge(0.90).all()
    )
    pbp_xyac_ok = (
        len(hist_pbp)
        and "xyac_mean_yardage_nonnull_completed_rate" in hist_pbp
        and hist_pbp["xyac_mean_yardage_nonnull_completed_rate"].fillna(0).ge(0.90).all()
    )
    pbp_current_ok = len(pbp_2026_prior) > 0

    hist_ngs = ngs_summary[ngs_summary["season"].between(2020, 2025)].copy() if not ngs_summary.empty else pd.DataFrame()
    ngs_core_rates = [
        f"{c}_nonnull_rate" for c in CORE_NGS
        if f"{c}_nonnull_rate" in hist_ngs.columns
    ]
    ngs_hist_ok = bool(
        len(hist_ngs)
        and len(ngs_core_rates) >= 4
        and hist_ngs["ngs_te_rows"].fillna(0).gt(0).all()
        and hist_ngs[ngs_core_rates].fillna(0).ge(0.70).all().all()
    )
    ngs_current_ok = len(ngs_2026_prior) > 0

    if ngs_hist_ok and ngs_current_ok:
        disposition = "TE_TARGET_QUALITY_NGS_SOURCE_SUPPORTS_FROZEN_PREDICTIVE_STUDY"
        preferred = "NGS_PLAYER_TRACKING"
    elif pbp_air_ok and pbp_xyac_ok and pbp_current_ok:
        disposition = "TE_TARGET_QUALITY_PBP_SOURCE_SUPPORTS_FROZEN_PREDICTIVE_STUDY"
        preferred = "PBP_TARGET_DEPTH_XYAC"
    else:
        disposition = "TE_TARGET_QUALITY_SOURCE_BLOCKED_FAIL_CLOSED"
        preferred = "NONE"

    payload = {
        "study": "TE_TARGET_QUALITY_EFFICIENCY_V1_SOURCE_AUDIT",
        "disposition": disposition,
        "preferred_next_family": preferred,
        "seasons": seasons,
        "model_fit_performed": False,
        "prediction_error_scored": False,
        "sportsbook_inputs_used": 0,
        "production_changed": False,
        "pbp_historical_air_yards_gate": bool(pbp_air_ok),
        "pbp_historical_xyac_gate": bool(pbp_xyac_ok),
        "pbp_2026_w1_w2_rows": int(len(pbp_2026_prior)),
        "ngs_historical_core_gate": bool(ngs_hist_ok),
        "ngs_2026_w1_w2_rows": int(len(ngs_2026_prior)),
        "ngs_load_error": ngs_meta.get("load_error", ""),
        "pbp_errors": pbp_errors,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# TE Target-Quality Efficiency V1 — Source Audit Result",
        "",
        f"Disposition: `{disposition}`",
        "",
        f"Preferred next family: **{preferred}**",
        f"PBP 2026 W1-W2 strict-prior TE target rows: **{len(pbp_2026_prior)}**",
        f"NGS 2026 W1-W2 strict-prior TE player-week rows: **{len(ngs_2026_prior)}**",
        "",
        "No predictive model was fit; no prediction error was scored; no sportsbook input was used; production was unchanged.",
    ]
    (out / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
