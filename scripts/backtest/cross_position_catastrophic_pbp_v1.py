#!/usr/bin/env python3
"""Cross-position Catastrophic Casebook V1 — Phase B PBP forensics.

This script enriches the frozen Phase-A catastrophic rows with postgame
play-by-play.  Postgame fields are forensic only and are never predictors.
Sportsbook fields are not used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.pbp import get_pbp


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def canon(v) -> str:
    return canon_team(v)


def key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def regular(x: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in x.columns:
        q = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    if "game_type" in x.columns:
        q = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            return q
    return x


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    parts = []
    for season in seasons:
        raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
        x = lower(_normalize_weekly(_to_pandas(raw), int(season)))
        x = x.loc[pd.to_numeric(x["week"], errors="coerce").between(1, 18)].copy()
        x["season"] = int(season)
        x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
        x["team"] = x["team"].map(canon)
        x["player_clean_key"] = x["player_clean_key"].map(key)
        parts.append(x)
    return pd.concat(parts, ignore_index=True, sort=False)


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    parts = []
    for season in seasons:
        x = regular(lower(get_pbp(int(season), min_rows=1)))
        x["season"] = pd.to_numeric(x.get("season"), errors="coerce")
        x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
        x = x.loc[x["season"].eq(int(season)) & x["week"].between(1, 18)].copy()
        x["season"] = int(season)
        x["week"] = x["week"].astype(int)
        x["posteam_canon"] = x.get("posteam", pd.Series("", index=x.index)).map(canon)
        parts.append(x)
        print(f"[phase_b] loaded PBP {season}: {len(x)} rows")
    return pd.concat(parts, ignore_index=True, sort=False)


def weekly_identity_map(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "week", "team", "player_clean_key", "player_id", "position",
            "targets", "receptions", "rec_yards", "rushes", "rush_yards", "pass_att", "pass_yards"]
    w = weekly[[c for c in cols if c in weekly.columns]].copy()
    w["player_id"] = w.get("player_id", "").astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    # One player/team/week row is expected in weekly official stats.  If an upstream
    # duplicate exists, preserve the row with the most football usage.
    w["_usage"] = sum(pd.to_numeric(w.get(c, 0), errors="coerce").fillna(0) for c in ["targets", "rushes", "pass_att"])
    w = w.sort_values("_usage").drop_duplicates(["season", "week", "team", "player_clean_key"], keep="last")
    return w.drop(columns=["_usage"])


def room_context(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()
    w["position"] = w["position"].astype(str).str.upper()
    out = []
    for (season, week, team), g in w.groupby(["season", "week", "team"], dropna=False):
        team_targets = float(pd.to_numeric(g.get("targets", 0), errors="coerce").fillna(0).sum())
        team_rushes = float(pd.to_numeric(g.get("rushes", 0), errors="coerce").fillna(0).sum())
        qb_rushes = float(pd.to_numeric(g.loc[g["position"].eq("QB"), "rushes"], errors="coerce").fillna(0).sum())
        for pos in ["WR", "TE", "RB"]:
            q = g.loc[g["position"].eq(pos) | ((pos == "RB") & g["position"].eq("FB"))].copy()
            room_targets = float(pd.to_numeric(q.get("targets", 0), errors="coerce").fillna(0).sum())
            room_rushes = float(pd.to_numeric(q.get("rushes", 0), errors="coerce").fillna(0).sum())
            max_tgt = float(pd.to_numeric(q.get("targets", 0), errors="coerce").fillna(0).max()) if len(q) else 0.0
            max_rush = float(pd.to_numeric(q.get("rushes", 0), errors="coerce").fillna(0).max()) if len(q) else 0.0
            out.append({"season": season, "week": week, "team": team, "position": pos,
                        "official_team_targets": team_targets, "official_team_rushes": team_rushes,
                        "official_qb_rushes": qb_rushes, "official_room_targets": room_targets,
                        "official_room_rushes": room_rushes, "official_max_room_player_targets": max_tgt,
                        "official_max_room_player_rushes": max_rush})
    return pd.DataFrame(out)


def score_state(g: pd.DataFrame, mask: pd.Series) -> tuple[float, float, float]:
    if "score_differential" not in g.columns or not mask.any():
        return np.nan, np.nan, np.nan
    s = pd.to_numeric(g.loc[mask, "score_differential"], errors="coerce")
    if s.notna().sum() == 0:
        return np.nan, np.nan, np.nan
    return float(s.lt(0).mean()), float(s.gt(0).mean()), float(s.eq(0).mean())


def half_counts(g: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    q = pd.to_numeric(g.get("qtr"), errors="coerce")
    return int((mask & q.le(2)).sum()), int((mask & q.ge(3)).sum())


def last_qtr(g: pd.DataFrame, mask: pd.Series) -> float:
    if not mask.any():
        return np.nan
    return float(pd.to_numeric(g.loc[mask, "qtr"], errors="coerce").max())


def player_forensics(row: pd.Series, game: pd.DataFrame, pid: str) -> dict:
    pos = str(row.position).upper()
    under = float(row.get("signed_error_actual_minus_pred", 0)) > 0
    threshold = float(row.threshold)
    actual = float(row.actual)
    pred = float(row.pred)
    qtr = pd.to_numeric(game.get("qtr"), errors="coerce")
    drives = int(pd.to_numeric(game.get("drive"), errors="coerce").nunique()) if "drive" in game.columns else 0
    offensive_plays = int((num(game, "qb_dropback", 0).fillna(0).eq(1) | num(game, "rush_attempt", 0).fillna(0).eq(1)).sum())

    out: dict[str, object] = {"case_status": "ok", "matched_player_id": pid, "pbp_offensive_plays": offensive_plays,
                             "pbp_offensive_drives": drives}

    if pos == "QB":
        pm = game.get("passer_player_id", pd.Series("", index=game.index)).astype(str).eq(pid)
        attempt = pm & num(game, "pass_attempt", 0).fillna(0).eq(1) & ~num(game, "sack", 0).fillna(0).eq(1)
        pg = game.loc[pm].copy()
        ay = num(game, "passing_yards", 0).fillna(0)
        complete = attempt & num(game, "complete_pass", 0).fillna(0).eq(1)
        yac = num(game, "yards_after_catch")
        longest = float(ay.loc[complete].max()) if complete.any() else 0.0
        max_yac = float(yac.loc[complete].max()) if complete.any() and yac.loc[complete].notna().any() else np.nan
        yac_total = float(yac.loc[complete].fillna(0).sum()) if complete.any() else 0.0
        trailing, leading, tied = score_state(game, attempt)
        h1, h2 = half_counts(game, attempt)
        out.update({
            "pbp_opportunities": int(attempt.sum()), "pbp_yards": float(ay.loc[pm].sum()),
            "first_half_opportunities": h1, "second_half_opportunities": h2,
            "trailing_opportunity_share": trailing, "leading_opportunity_share": leading, "tied_opportunity_share": tied,
            "longest_play_yards": longest, "plays_20plus": int((complete & ay.ge(20)).sum()),
            "plays_40plus": int((complete & ay.ge(40)).sum()), "plays_60plus": int((complete & ay.ge(60)).sum()),
            "yac_total": yac_total, "max_yac": max_yac, "last_opportunity_qtr": last_qtr(game, attempt),
            "sacks": int((pm & num(game, "sack", 0).fillna(0).eq(1)).sum()),
            "interceptions": int((attempt & num(game, "interception", 0).fillna(0).eq(1)).sum()),
        })
    elif pos in {"WR", "TE"}:
        rm = game.get("receiver_player_id", pd.Series("", index=game.index)).astype(str).eq(pid)
        target = rm & num(game, "pass_attempt", 0).fillna(0).eq(1) & ~num(game, "sack", 0).fillna(0).eq(1)
        complete = target & num(game, "complete_pass", 0).fillna(0).eq(1)
        ry = num(game, "receiving_yards", np.nan)
        if ry.notna().sum() == 0:
            ry = num(game, "yards_gained", 0)
        ry = ry.fillna(0)
        yac = num(game, "yards_after_catch")
        longest = float(ry.loc[complete].max()) if complete.any() else 0.0
        max_yac = float(yac.loc[complete].max()) if complete.any() and yac.loc[complete].notna().any() else np.nan
        yac_total = float(yac.loc[complete].fillna(0).sum()) if complete.any() else 0.0
        trailing, leading, tied = score_state(game, target)
        h1, h2 = half_counts(game, target)
        out.update({
            "pbp_opportunities": int(target.sum()), "pbp_receptions": int(complete.sum()), "pbp_yards": float(ry.loc[complete].sum()),
            "first_half_opportunities": h1, "second_half_opportunities": h2,
            "trailing_opportunity_share": trailing, "leading_opportunity_share": leading, "tied_opportunity_share": tied,
            "longest_play_yards": longest, "plays_20plus": int((complete & ry.ge(20)).sum()),
            "plays_40plus": int((complete & ry.ge(40)).sum()), "plays_60plus": int((complete & ry.ge(60)).sum()),
            "yac_total": yac_total, "max_yac": max_yac, "last_opportunity_qtr": last_qtr(game, target),
        })
    elif pos == "RB":
        rm = game.get("rusher_player_id", pd.Series("", index=game.index)).astype(str).eq(pid)
        rush = rm & num(game, "rush_attempt", 0).fillna(0).eq(1)
        ry = num(game, "rushing_yards", np.nan)
        if ry.notna().sum() == 0:
            ry = num(game, "yards_gained", 0)
        ry = ry.fillna(0)
        longest = float(ry.loc[rush].max()) if rush.any() else 0.0
        trailing, leading, tied = score_state(game, rush)
        h1, h2 = half_counts(game, rush)
        out.update({
            "pbp_opportunities": int(rush.sum()), "pbp_yards": float(ry.loc[rush].sum()),
            "first_half_opportunities": h1, "second_half_opportunities": h2,
            "trailing_opportunity_share": trailing, "leading_opportunity_share": leading, "tied_opportunity_share": tied,
            "longest_play_yards": longest, "plays_10plus": int((rush & ry.ge(10)).sum()),
            "plays_20plus": int((rush & ry.ge(20)).sum()), "plays_40plus": int((rush & ry.ge(40)).sum()),
            "plays_60plus": int((rush & ry.ge(60)).sum()), "yac_total": np.nan, "max_yac": np.nan,
            "last_opportunity_qtr": last_qtr(game, rush),
        })
    else:
        return {"case_status": "unsupported_position"}

    longest = float(out.get("longest_play_yards", 0) or 0)
    without = actual - longest if under else actual
    error_without = abs(without - pred)
    resolves = bool(under and error_without < threshold)
    out["actual_without_longest_play"] = without
    out["abs_error_without_longest_play"] = error_without
    out["catastrophe_resolved_without_longest_play"] = resolves
    out["largest_play_share_of_actual"] = longest / actual if actual > 0 else np.nan
    out["early_exit_pattern"] = bool(np.isfinite(out.get("last_opportunity_qtr", np.nan)) and float(out["last_opportunity_qtr"]) <= 2 and float(row.actual_opportunity) < 0.60 * max(float(row.pred_opportunity), 1e-9))
    return out


def classify(row: pd.Series) -> tuple[str, str]:
    pos = str(row.position).upper()
    under = str(row.direction) == "UNDERPROJECTED"
    dom = str(row.dominant_mechanism)
    longest = float(row.get("longest_play_yards", 0) or 0)
    max_yac = pd.to_numeric(pd.Series([row.get("max_yac")]), errors="coerce").iloc[0]
    yac_total = pd.to_numeric(pd.Series([row.get("yac_total")]), errors="coerce").iloc[0]
    actual = max(float(row.actual), 1.0)
    resolves = bool(row.get("catastrophe_resolved_without_longest_play", False))
    early = bool(row.get("early_exit_pattern", False))
    trailing = pd.to_numeric(pd.Series([row.get("trailing_opportunity_share")]), errors="coerce").iloc[0]
    opp_gap = float(row.actual_opportunity) - float(row.pred_opportunity)

    label = "MIXED"
    predictability = "UNRESOLVED"

    if early:
        return "EARLY_EXIT_PATTERN", "LOW_PREDICTABILITY_GAME_EVENT"
    if under and resolves and longest >= (60 if pos == "QB" else 40):
        return ("SINGLE_EXPLOSIVE_PLAY" if pos != "RB" else "SINGLE_EXPLOSIVE_RUN"), "LOW_PREDICTABILITY_GAME_EVENT"
    if under and pos in {"QB", "WR", "TE"} and np.isfinite(max_yac) and max_yac >= 30 and np.isfinite(yac_total) and yac_total >= 0.30 * actual:
        return "YAC_DRIVEN_EXPLOSION", "LOW_PREDICTABILITY_GAME_EVENT"

    if pos == "QB":
        if opp_gap >= 10:
            label = "GAME_SCRIPT_PASS_VOLUME" if np.isfinite(trailing) and trailing >= 0.55 else "PASS_VOLUME_EXPLOSION"
            predictability = "PARTIAL_PREGAME_SIGNAL"
        elif opp_gap <= -8:
            label = "UNEXPECTED_LOW_PASS_VOLUME"
            predictability = "PARTIAL_PREGAME_SIGNAL"
        elif dom == "EFFICIENCY":
            label = "SUSTAINED_EFFICIENCY_EXPLOSION" if under else "SUSTAINED_EFFICIENCY_COLLAPSE"
            predictability = "PARTIAL_PREGAME_SIGNAL"
    elif pos in {"WR", "TE"}:
        if dom == "TEAM_OPPORTUNITY_MISS":
            label = "POSITION_POOL_MISS"
            predictability = "PARTIAL_PREGAME_SIGNAL"
        elif dom == "PLAYER_ENTITLEMENT_MISS":
            label = "PLAYER_ENTITLEMENT_MISS"
            predictability = "PARTIAL_PREGAME_SIGNAL"
        elif dom == "CONVERSION_MISS":
            label = "CONVERSION_MISS"
            predictability = "PARTIAL_PREGAME_SIGNAL"
        elif dom == "EFFICIENCY":
            label = "SUSTAINED_EFFICIENCY_EXPLOSION" if under else "SUSTAINED_EFFICIENCY_COLLAPSE"
            predictability = "PARTIAL_PREGAME_SIGNAL"
    elif pos == "RB":
        if dom in {"OPPORTUNITY_MISS", "TEAM_OPPORTUNITY_MISS"}:
            label = "WORKLOAD_EXPLOSION" if under else "WORKLOAD_COLLAPSE"
            predictability = "PARTIAL_PREGAME_SIGNAL"
        elif dom == "EFFICIENCY":
            label = "SUSTAINED_RUSH_EFFICIENCY_EXPLOSION" if under else "SUSTAINED_RUSH_EFFICIENCY_COLLAPSE"
            predictability = "PARTIAL_PREGAME_SIGNAL"

    return label, predictability


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-a-casebook", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    cases = pd.read_csv(a.phase_a_casebook, low_memory=False)
    cases["position"] = cases["position"].astype(str).str.upper()
    cases["season"] = pd.to_numeric(cases["season"], errors="raise").astype(int)
    cases["week"] = pd.to_numeric(cases["week"], errors="raise").astype(int)
    cases["team"] = cases["team"].map(canon)
    cases["player_clean_key"] = cases["player_clean_key"].map(key)
    seasons = sorted(cases["season"].unique().tolist())

    weekly = load_weekly(seasons)
    wid = weekly_identity_map(weekly)
    rooms = room_context(weekly)
    pbp = load_pbp(seasons)

    idcols = ["season", "week", "team", "player_clean_key", "player_id"]
    x = cases.merge(wid[idcols], on=["season", "week", "team", "player_clean_key"], how="left", validate="many_to_one")
    x = x.merge(rooms, on=["season", "week", "team", "position"], how="left", validate="many_to_one")

    extras = []
    for i, row in x.iterrows():
        pid = str(row.get("player_id", "")).strip()
        if pid in {"", "nan", "None", "<NA>"}:
            extras.append({"case_status": "missing_weekly_identity"})
            continue
        game = pbp.loc[pbp["season"].eq(int(row.season)) & pbp["week"].eq(int(row.week)) & pbp["posteam_canon"].eq(canon(row.team))].copy()
        if game.empty:
            extras.append({"case_status": "missing_offense_pbp"})
            continue
        extras.append(player_forensics(row, game, pid))
        if (i + 1) % 250 == 0:
            print(f"[phase_b] cases {i+1}/{len(x)}")

    out = pd.concat([x.reset_index(drop=True), pd.DataFrame(extras)], axis=1)
    labels = out.apply(classify, axis=1, result_type="expand")
    out["pbp_primary_label"] = labels[0]
    out["pregame_predictability_class"] = labels[1]
    out["pbp_opportunity_gap_vs_official"] = pd.to_numeric(out.get("pbp_opportunities"), errors="coerce") - pd.to_numeric(out.get("actual_opportunity"), errors="coerce")
    out["pbp_yards_gap_vs_official"] = pd.to_numeric(out.get("pbp_yards"), errors="coerce") - pd.to_numeric(out.get("actual"), errors="coerce")

    ok = out["case_status"].astype(str).eq("ok")
    summary = out.groupby(["position", "pbp_primary_label", "pregame_predictability_class"], as_index=False).agg(
        n=("abs_error", "size"), abs_error_mass=("abs_error", "sum"),
        mean_abs_error=("abs_error", "mean"), longest_resolves=("catastrophe_resolved_without_longest_play", "sum")
    )
    summary["position_error_mass_share"] = summary["abs_error_mass"] / summary.groupby("position")["abs_error_mass"].transform("sum")

    pred = out.groupby(["position", "pregame_predictability_class"], as_index=False).agg(
        n=("abs_error", "size"), abs_error_mass=("abs_error", "sum"))
    pred["position_error_mass_share"] = pred["abs_error_mass"] / pred.groupby("position")["abs_error_mass"].transform("sum")

    longsum = out.groupby("position", as_index=False).agg(
        catastrophic_rows=("abs_error", "size"), pbp_ok=("case_status", lambda s: int(s.astype(str).eq("ok").sum())),
        largest_play_resolves=("catastrophe_resolved_without_longest_play", "sum"),
        mean_longest_play=("longest_play_yards", "mean"), p90_longest_play=("longest_play_yards", lambda s: pd.to_numeric(s, errors="coerce").quantile(.9)))
    longsum["largest_play_resolution_rate"] = longsum["largest_play_resolves"] / longsum["catastrophic_rows"]

    q4 = out.loc[out["opportunity_quartile"].astype(str).eq("Q4")].copy()
    q4sum = q4.groupby(["position", "direction", "pbp_primary_label"], as_index=False).agg(
        n=("abs_error", "size"), abs_error_mass=("abs_error", "sum"))
    q4sum["direction_error_mass_share"] = q4sum["abs_error_mass"] / q4sum.groupby(["position", "direction"])["abs_error_mass"].transform("sum")

    action = summary.copy()
    action["actionable_by_mass_or_count"] = (action["position_error_mass_share"] >= .08) | (action["n"] >= 20)
    action["actionable_phase_b"] = action["actionable_by_mass_or_count"] & action["pregame_predictability_class"].isin(["PARTIAL_PREGAME_SIGNAL", "STRUCTURAL_PREGAME_SIGNAL"])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "cross_position_pbp_casebook.csv", index=False)
    summary.to_csv(a.out_dir / "cross_position_pbp_mechanism_summary.csv", index=False)
    pred.to_csv(a.out_dir / "cross_position_predictability_summary.csv", index=False)
    longsum.to_csv(a.out_dir / "cross_position_largest_play_sensitivity.csv", index=False)
    q4sum.to_csv(a.out_dir / "cross_position_q4_pbp_mechanisms.csv", index=False)
    action.to_csv(a.out_dir / "cross_position_phase_b_actionable_clusters.csv", index=False)

    result = {
        "disposition": "PHASE_B_PBP_FORENSICS_COMPLETE",
        "catastrophic_rows": int(len(out)),
        "pbp_ok_rows": int(ok.sum()),
        "pbp_ok_rate": float(ok.mean()),
        "postgame_forensic_fields_used_for_prediction": False,
        "sportsbook_features_used_for_cause_classification": False,
        "actionable_clusters": action.loc[action["actionable_phase_b"]].to_dict("records"),
    }
    (a.out_dir / "cross_position_phase_b_result.json").write_text(json.dumps(result, indent=2, default=str))

    print(json.dumps(result, indent=2, default=str))
    print("\n=== PBP MECHANISMS ===")
    print(summary.sort_values(["position", "position_error_mass_share"], ascending=[True, False]).to_string(index=False))
    print("\n=== PREDICTABILITY ===")
    print(pred.sort_values(["position", "position_error_mass_share"], ascending=[True, False]).to_string(index=False))
    print("\n=== LARGEST PLAY ===")
    print(longsum.to_string(index=False))
    print("\n=== Q4 PBP ===")
    print(q4sum.sort_values(["position", "direction", "direction_error_mass_share"], ascending=[True, True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
