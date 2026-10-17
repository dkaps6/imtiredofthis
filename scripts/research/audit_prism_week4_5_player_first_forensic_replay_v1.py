#!/usr/bin/env python3
"""PRISM Week-4/5 player-first forensic replay V1.

Diagnostic only. The named player in the actual pregame football situation is
primary. This script does not fit coefficients, tune thresholds, alter
production, call OddsAPI, or use sportsbook data upstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly

VERSION = "PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_V1"
SEASON = 2026
DIRECT = {"pass_yards", "rush_yards", "rec_yards", "receptions"}
ALL_W5 = DIRECT | {"rush_rec_yards"}
RB_POS = {"RB", "HB", "FB"}


def num(v: Any) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def family(position: Any) -> str:
    p = str(position or "").upper()
    if p in RB_POS:
        return "RB"
    if p in {"QB", "WR", "TE"}:
        return p
    return p or "UNK"


def opp_from_event(event_id: Any, team: Any) -> str:
    parts = [canon_team(x) for x in str(event_id or "").split("|") if str(x).strip()]
    t = canon_team(team)
    others = [x for x in parts if x and x != t]
    return others[0] if len(others) == 1 else ""


def opportunity_type(market: Any) -> str:
    m = str(market)
    if m == "pass_yards":
        return "pass_attempts"
    if m == "rush_yards":
        return "carries"
    if m in {"rec_yards", "receptions"}:
        return "targets"
    if m == "rush_rec_yards":
        return "mixed_rush_targets"
    return "unknown"


def component_label(opp: Any, eff: Any) -> str:
    try:
        o = abs(float(opp)); e = abs(float(eff))
    except Exception:
        return "INSUFFICIENT_COMPONENT_EVIDENCE"
    if not np.isfinite(o) or not np.isfinite(e):
        return "INSUFFICIENT_COMPONENT_EVIDENCE"
    if max(o, e) <= 1e-12:
        return "MIXED_OPPORTUNITY_EFFICIENCY"
    if o >= 1.5 * e:
        return "OPPORTUNITY_DOMINANT"
    if e >= 1.5 * o:
        return "EFFICIENCY_DOMINANT"
    return "MIXED_OPPORTUNITY_EFFICIENCY"


def market_actual(row: pd.Series) -> float:
    m = str(row.get("market", ""))
    if m == "pass_yards":
        return float(row["pass_yards"]) if np.isfinite(row.get("pass_yards", np.nan)) else np.nan
    if m == "rush_yards":
        return float(row["rush_yards"]) if np.isfinite(row.get("rush_yards", np.nan)) else np.nan
    if m == "rec_yards":
        return float(row["rec_yards"]) if np.isfinite(row.get("rec_yards", np.nan)) else np.nan
    if m == "receptions":
        return float(row["receptions"]) if np.isfinite(row.get("receptions", np.nan)) else np.nan
    if m == "rush_rec_yards":
        ry = row.get("rush_yards", np.nan); cy = row.get("rec_yards", np.nan)
        return float(ry + cy) if np.isfinite(ry) and np.isfinite(cy) else np.nan
    return np.nan


def load_actuals(week: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[SEASON], summary_level="week")
    if hasattr(raw, "to_dicts"):
        raw = pd.DataFrame(raw.to_dicts())
    elif hasattr(raw, "to_pandas"):
        raw = raw.to_pandas()
    x = _normalize_weekly(raw, SEASON)
    x = x.loc[num(x["week"]).eq(week)].copy()
    if x.empty:
        raise RuntimeError(f"nflverse Week-{week} player stats are empty")
    keep = [
        "player_identity_key", "player_id", "player", "player_clean_key", "team", "position",
        "targets", "receptions", "rec_yards", "rushes", "rush_yards", "pass_att", "pass_yards",
        "team_targets", "team_rushes", "tgt_share_game", "rush_share_game", "ypt_game",
        "ypc_game", "catch_rate_game",
    ]
    keep = [cc for cc in keep if cc in x.columns]
    out = x[keep].copy()
    out["team"] = out["team"].map(canon_team)
    return out.drop_duplicates(["player_clean_key", "team"], keep="last")


def prepare_history(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    req = {
        "season", "week", "event_id", "team", "player", "player_clean_key", "position_family",
        "market", "predicted_opportunities", "actual_opportunities", "baseline_projection",
        "actual_output", "model_effective_efficiency", "actual_efficiency", "baseline_error",
        "grading_identity_valid", "component_decomposition_eligible",
    }
    miss = req - set(h.columns)
    if miss:
        raise RuntimeError(f"W1-W4 component history missing {sorted(miss)}")
    x = h.copy()
    x["season"] = num(x["season"]); x["week"] = num(x["week"])
    x = x.loc[x["season"].eq(SEASON) & x["market"].isin(DIRECT)].copy()
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x.apply(lambda r: opp_from_event(r["event_id"], r["team"]), axis=1)
    x["position_family"] = x["position_family"].astype(str).str.upper()
    x["opportunity_type"] = x["market"].map(opportunity_type)
    for c in [
        "predicted_opportunities", "actual_opportunities", "baseline_projection", "actual_output",
        "model_effective_efficiency", "actual_efficiency", "baseline_error",
    ]:
        x[c] = num(x[c])

    rep = x.loc[x["market"].isin({"pass_yards", "rush_yards", "rec_yards"})].copy()
    team_tot = rep.groupby(["week", "team", "opportunity_type"], as_index=False).agg(
        predicted_team_opportunity=("predicted_opportunities", "sum"),
        actual_team_opportunity=("actual_opportunities", "sum"),
    )
    room_tot = rep.groupby(["week", "team", "position_family", "opportunity_type"], as_index=False).agg(
        predicted_room_opportunity=("predicted_opportunities", "sum"),
        actual_room_opportunity=("actual_opportunities", "sum"),
    )
    rep = rep.merge(team_tot, on=["week", "team", "opportunity_type"], how="left")
    rep = rep.merge(room_tot, on=["week", "team", "position_family", "opportunity_type"], how="left")
    rep["predicted_team_share"] = rep["predicted_opportunities"] / rep["predicted_team_opportunity"].replace(0, np.nan)
    rep["actual_team_share"] = rep["actual_opportunities"] / rep["actual_team_opportunity"].replace(0, np.nan)
    rep["predicted_room_share"] = rep["predicted_opportunities"] / rep["predicted_room_opportunity"].replace(0, np.nan)
    rep["actual_room_share"] = rep["actual_opportunities"] / rep["actual_room_opportunity"].replace(0, np.nan)

    comp_rows = []
    for keys, g in rep.groupby(["week", "team", "position_family", "opportunity_type"], sort=False):
        vals = g[["player_clean_key", "actual_room_share"]].dropna()
        for _, r in vals.iterrows():
            others = vals.loc[vals["player_clean_key"] != r["player_clean_key"], "actual_room_share"]
            comp_rows.append({
                "week": keys[0], "team": keys[1], "position_family": keys[2], "opportunity_type": keys[3],
                "player_clean_key": r["player_clean_key"],
                "top_competitor_share": float(others.max()) if len(others) else 0.0,
                "room_hhi": float(np.square(vals["actual_room_share"]).sum()) if len(vals) else np.nan,
            })
    competition = pd.DataFrame(comp_rows)
    rep = rep.merge(
        competition,
        on=["week", "team", "position_family", "opportunity_type", "player_clean_key"],
        how="left",
    )
    return x, rep


def prior_context(history: pd.DataFrame, rep: pd.DataFrame, targets: pd.DataFrame, target_week: int) -> pd.DataFrame:
    out = []
    for _, tr in targets.iterrows():
        pkey = str(tr["player_clean_key"]); market = str(tr["market"]); team = canon_team(tr["team"])
        otype = opportunity_type(market)
        hp = history.loc[(history["week"] < target_week) & history["player_clean_key"].eq(pkey) & history["market"].eq(market)].sort_values("week")
        same_team = hp.loc[hp["team"].eq(team)]
        role_market = "rec_yards" if otype == "targets" else ("rush_yards" if otype == "carries" else "pass_yards")
        rp = rep.loc[(rep["week"] < target_week) & rep["player_clean_key"].eq(pkey) & rep["market"].eq(role_market)].sort_values("week")
        rp_team = rp.loc[rp["team"].eq(team)]

        def mean_col(df: pd.DataFrame, c: str) -> float:
            return float(num(df[c]).mean()) if len(df) and c in df else np.nan
        def last_col(df: pd.DataFrame, c: str) -> float:
            z = num(df[c]).dropna() if len(df) and c in df else pd.Series(dtype=float)
            return float(z.iloc[-1]) if len(z) else np.nan
        def last2_minus_earlier(df: pd.DataFrame, c: str) -> float:
            z = num(df[c]).dropna() if len(df) and c in df else pd.Series(dtype=float)
            if len(z) < 3:
                return np.nan
            return float(z.iloc[-2:].mean() - z.iloc[:-2].mean())

        out.append({
            "target_week": target_week,
            "player_clean_key": pkey,
            "market": market,
            "prior_same_market_games": int(hp["week"].nunique()),
            "prior_same_team_market_games": int(same_team["week"].nunique()),
            "prior_output_mean": mean_col(hp, "actual_output"),
            "prior_output_last": last_col(hp, "actual_output"),
            "prior_player_mae": float(num(hp["baseline_error"]).abs().mean()) if len(hp) else np.nan,
            "prior_opportunity_mean": mean_col(hp, "actual_opportunities"),
            "prior_opportunity_last": last_col(hp, "actual_opportunities"),
            "prior_efficiency_mean": mean_col(hp, "actual_efficiency"),
            "prior_efficiency_last": last_col(hp, "actual_efficiency"),
            "prior_opportunity_trajectory": last2_minus_earlier(hp, "actual_opportunities"),
            "prior_efficiency_trajectory": last2_minus_earlier(hp, "actual_efficiency"),
            "prior_role_games": int(rp_team["week"].nunique()),
            "prior_team_share_mean": mean_col(rp_team, "actual_team_share"),
            "prior_team_share_last": last_col(rp_team, "actual_team_share"),
            "prior_team_share_trajectory": last2_minus_earlier(rp_team, "actual_team_share"),
            "prior_room_share_mean": mean_col(rp_team, "actual_room_share"),
            "prior_room_share_last": last_col(rp_team, "actual_room_share"),
            "prior_top_competitor_share_last": last_col(rp_team, "top_competitor_share"),
            "prior_room_hhi_last": last_col(rp_team, "room_hhi"),
        })
    return pd.DataFrame(out)


def build_week4(board: pd.DataFrame, history: pd.DataFrame, rep: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    req = {"season", "week", "player", "player_clean_key", "team", "opponent", "market", "model_proj", "source_run_id"}
    miss = req - set(board.columns)
    if miss:
        raise RuntimeError(f"Week-4 preserved board missing {sorted(miss)}")
    b = board.copy()
    b["season"] = num(b["season"]); b["week"] = num(b["week"])
    b = b.loc[b["season"].eq(SEASON) & b["week"].eq(4) & b["market"].isin(ALL_W5)].copy()
    b["team"] = b["team"].map(canon_team); b["opponent"] = b["opponent"].map(canon_team)
    invariance = b.groupby(["team", "player_clean_key", "market"])["model_proj"].nunique(dropna=True)
    if (invariance > 1).any():
        raise RuntimeError("Week-4 frozen board model projection varies across sportsbook rows")
    x = b.sort_values(["team", "player_clean_key", "market"]).drop_duplicates(["team", "player_clean_key", "market"], keep="last")
    if not set(num(x["source_run_id"]).dropna().astype(int).unique()).issubset({36935917903}):
        raise RuntimeError("Week-4 board source-run drift")
    x["model_proj"] = num(x["model_proj"])

    a = actuals.copy()
    keep_actual = [cc for cc in [
        "team", "player_clean_key", "position", "targets", "receptions", "rec_yards", "rushes", "rush_yards",
        "pass_att", "pass_yards", "team_targets", "team_rushes", "tgt_share_game", "rush_share_game",
        "ypt_game", "ypc_game", "catch_rate_game",
    ] if cc in a.columns]
    x = x.merge(a[keep_actual], on=["team", "player_clean_key"], how="left", validate="many_to_one")
    x["actual"] = x.apply(market_actual, axis=1)
    x = x.loc[x["actual"].notna()].copy()
    x["position_family"] = x["position"].map(family)
    x["abs_error"] = (x["model_proj"] - x["actual"]).abs()

    h4 = history.loc[history["week"].eq(4) & history["market"].isin(DIRECT)].copy()
    h4 = h4[[
        "team", "player_clean_key", "market", "baseline_projection", "baseline_reconstructed",
        "predicted_opportunities", "actual_opportunities", "model_effective_efficiency",
        "actual_efficiency", "component_decomposition_eligible", "actual_output",
    ]].rename(columns={
        "baseline_projection": "component_reference_projection",
        "baseline_reconstructed": "component_reconstructed_mean",
        "predicted_opportunities": "pred_opportunity",
        "actual_opportunities": "component_actual_opportunity",
        "model_effective_efficiency": "pred_efficiency",
        "actual_efficiency": "component_actual_efficiency",
        "actual_output": "component_actual_output",
    })
    x = x.merge(h4, on=["team", "player_clean_key", "market"], how="left", validate="one_to_one")
    x["actual_opportunity"] = x["component_actual_opportunity"]
    x["realized_efficiency"] = x["component_actual_efficiency"]

    pn = num(x["pred_opportunity"]); an = num(x["actual_opportunity"])
    pe = num(x["pred_efficiency"]); ae = num(x["realized_efficiency"])
    x["opportunity_contribution"] = (an - pn) * (pe + ae) / 2.0
    x["efficiency_contribution"] = (ae - pe) * (pn + an) / 2.0
    x["downstream_contribution"] = num(x["component_reconstructed_mean"]) - x["model_proj"]
    x["final_identity_gap"] = (
        x["opportunity_contribution"] + x["efficiency_contribution"] + x["downstream_contribution"]
        - (x["actual"] - x["model_proj"])
    ).abs()
    x["exact_component_authority"] = (
        x["component_decomposition_eligible"].eq(True)
        & x["component_reconstructed_mean"].notna()
        & x["final_identity_gap"].le(1e-8)
    )
    x["component_scope"] = np.where(
        x["exact_component_authority"],
        "REFERENCE_OPPORTUNITY_EFFICIENCY_PLUS_EXACT_DOWNSTREAM_RESIDUAL",
        "NO_EXACT_COMPONENT_IDENTITY",
    )
    x["component_label"] = [component_label(o, e) for o, e in zip(x["opportunity_contribution"], x["efficiency_contribution"])]

    rr = rep.loc[rep["week"].eq(4), [
        "player_clean_key", "team", "market", "predicted_team_opportunity", "actual_team_opportunity",
        "predicted_team_share", "actual_team_share", "predicted_room_share", "actual_room_share",
    ]].copy()
    rec = rr.loc[rr["market"].eq("rec_yards")].copy()
    rec["market"] = "receptions"
    rr = pd.concat([rr, rec], ignore_index=True)
    x = x.merge(rr, on=["player_clean_key", "team", "market"], how="left", validate="one_to_one")
    x["parent_opportunity_gap"] = x["actual_team_opportunity"] - x["predicted_team_opportunity"]
    x["player_share_gap"] = x["actual_team_share"] - x["predicted_team_share"]
    x["source_week"] = 4
    return x


def build_week5(settled: pd.DataFrame, state: pd.DataFrame, cases: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    req = {"season", "week", "player_clean_key", "player", "team", "opponent", "market", "model_proj", "actual", "position", "football_settlement_status"}
    miss = req - set(settled.columns)
    if miss:
        raise RuntimeError(f"Week-5 settled board missing {sorted(miss)}")
    x = settled.copy()
    x["season"] = num(x["season"]); x["week"] = num(x["week"])
    x = x.loc[x["season"].eq(SEASON) & x["week"].eq(5) & x["football_settlement_status"].eq("SETTLED") & x["market"].isin(ALL_W5)].copy()
    x["team"] = x["team"].map(canon_team); x["opponent"] = x["opponent"].map(canon_team)
    x["position_family"] = x["position"].map(family)
    x["model_proj"] = num(x["model_proj"]); x["actual"] = num(x["actual"])
    x["abs_error"] = (x["model_proj"] - x["actual"]).abs()

    s = state.copy(); s["team"] = s["team"].map(canon_team)
    if s.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("duplicate Week-5 player-state team/player")
    state_cols = [c for c in [
        "team", "player_clean_key", "rules_tgt_share", "rules_rush_share", "rules_ypt", "rules_ypc", "rules_catch_rate",
        "rules_pass_eff_mult", "rules_rush_eff_mult", "expected_targets", "expected_carries", "player_tgt_share", "player_rush_share",
        "player_route_rate", "player_yprr", "player_ypt", "player_ypc", "player_catch_rate", "injury_status", "injury_designation",
        "off_plays_est", "off_proe", "off_success_rate", "opp_success_rate_def", "opp_pressure_generated", "opp_def_pass_epa",
        "opp_def_rush_epa", "opp_explosive_allowed", "workload_score", "mc_expected_pass_attempts",
    ] if c in s.columns]
    x = x.merge(s[state_cols], on=["team", "player_clean_key"], how="left", validate="many_to_one")

    a = actuals.copy()
    x = x.merge(a, on=["team", "player_clean_key"], how="left", suffixes=("", "_stats"), validate="many_to_one")

    x["state_expected_opportunity"] = np.nan
    x["state_rules_efficiency"] = np.nan
    x["actual_opportunity"] = np.nan
    x["realized_efficiency"] = np.nan
    for idx, r in x.iterrows():
        m = str(r["market"]); act = r["actual"]
        if m == "pass_yards":
            po = r.get("mc_expected_pass_attempts", np.nan); ao = r.get("pass_att", np.nan)
            pe = (r["model_proj"] / po) if np.isfinite(po) and po > 0 else np.nan
            ae = (act / ao) if np.isfinite(ao) and ao > 0 else np.nan
        elif m == "rush_yards":
            po = r.get("expected_carries", np.nan); ao = r.get("rushes", np.nan); pe = r.get("rules_ypc", np.nan)
            ae = (act / ao) if np.isfinite(ao) and ao > 0 else np.nan
        elif m == "rec_yards":
            po = r.get("expected_targets", np.nan); ao = r.get("targets", np.nan); pe = r.get("rules_ypt", np.nan)
            ae = (act / ao) if np.isfinite(ao) and ao > 0 else np.nan
        elif m == "receptions":
            po = r.get("expected_targets", np.nan); ao = r.get("targets", np.nan); pe = r.get("rules_catch_rate", np.nan)
            ae = (act / ao) if np.isfinite(ao) and ao > 0 else np.nan
        else:
            po = ao = pe = ae = np.nan
        x.loc[idx, ["state_expected_opportunity", "actual_opportunity", "state_rules_efficiency", "realized_efficiency"]] = [po, ao, pe, ae]

    x["state_reconstructed_mean"] = x["state_expected_opportunity"] * x["state_rules_efficiency"]
    x["final_minus_state_reconstructed"] = x["model_proj"] - x["state_reconstructed_mean"]
    x["state_opportunity_contribution"] = (x["actual_opportunity"] - x["state_expected_opportunity"]) * x["state_rules_efficiency"]
    x["state_efficiency_contribution"] = x["actual_opportunity"] * (x["realized_efficiency"] - x["state_rules_efficiency"])

    c = cases.copy(); c["team"] = c["team"].map(canon_team)
    exact_cols = [c0 for c0 in [
        "team", "player_clean_key", "market", "pred_opportunity", "actual_opportunity", "pred_efficiency", "actual_efficiency",
        "opportunity_contribution", "efficiency_contribution", "rush_opportunity_contribution", "rush_efficiency_contribution",
        "rec_opportunity_contribution", "rec_efficiency_contribution", "component_label", "case_rank", "pred_team_pass_att",
        "pred_team_rush_att", "team_targets", "team_rushes", "final_target_probability", "final_rush_probability",
        "pred_carries", "pred_targets", "rules_ypc", "rules_ypt",
    ] if c0 in c.columns]
    exact = c[exact_cols].copy().rename(columns={
        "actual_opportunity": "exact_actual_opportunity",
        "actual_efficiency": "exact_actual_efficiency",
        "rules_ypc": "exact_rules_ypc",
        "rules_ypt": "exact_rules_ypt",
    })
    x = x.merge(exact, on=["team", "player_clean_key", "market"], how="left", validate="one_to_one")

    x["pred_opportunity"] = x["pred_opportunity"].where(x["component_label"].notna(), x["state_expected_opportunity"])
    x["actual_opportunity"] = x["exact_actual_opportunity"].where(x["component_label"].notna(), x["actual_opportunity"])
    x["pred_efficiency"] = x["pred_efficiency"].where(x["component_label"].notna(), x["state_rules_efficiency"])
    x["realized_efficiency"] = x["exact_actual_efficiency"].where(x["component_label"].notna(), x["realized_efficiency"])

    x["component_reconstructed_mean"] = np.nan
    direct_exact = x["component_label"].notna() & x["market"].isin(DIRECT)
    x.loc[direct_exact, "component_reconstructed_mean"] = (
        num(x.loc[direct_exact, "pred_opportunity"]) * num(x.loc[direct_exact, "pred_efficiency"])
    )
    mixed_exact = x["component_label"].notna() & x["market"].eq("rush_rec_yards")
    x.loc[mixed_exact, "component_reconstructed_mean"] = (
        num(x.loc[mixed_exact, "pred_carries"]) * num(x.loc[mixed_exact, "exact_rules_ypc"])
        + num(x.loc[mixed_exact, "pred_targets"]) * num(x.loc[mixed_exact, "exact_rules_ypt"])
    )
    x["downstream_contribution"] = x["component_reconstructed_mean"] - x["model_proj"]
    x["final_identity_gap"] = (
        num(x["opportunity_contribution"]).fillna(0)
        + num(x["efficiency_contribution"]).fillna(0)
        + num(x["downstream_contribution"])
        - (x["actual"] - x["model_proj"])
    ).abs()
    x["exact_component_authority"] = (
        x["component_label"].notna()
        & x["component_reconstructed_mean"].notna()
        & x["final_identity_gap"].le(1e-8)
    )
    x["component_scope"] = np.where(
        x["exact_component_authority"],
        "FROZEN_PRODUCTION_COMPONENTS_PLUS_EXACT_DOWNSTREAM_RESIDUAL",
        "ALL_ROW_STATE_CONTEXT_ONLY",
    )
    x["source_week"] = 5
    return x


def attach_prior(history: pd.DataFrame, rep: pd.DataFrame, all_rows: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for week in [4, 5]:
        t = all_rows.loc[all_rows["source_week"].eq(week), ["player_clean_key", "market", "team", "position_family"]].drop_duplicates()
        pc = prior_context(history, rep, t, week)
        pieces.append(pc)
    pc = pd.concat(pieces, ignore_index=True)
    return all_rows.merge(pc, left_on=["source_week", "player_clean_key", "market"], right_on=["target_week", "player_clean_key", "market"], how="left", validate="many_to_one")


def role_signal(row: pd.Series) -> str:
    week = int(row["source_week"])
    market = str(row["market"])
    if week == 4 and market in DIRECT:
        p = row.get("predicted_team_share", np.nan); q = row.get("prior_team_share_last", np.nan)
        if np.isfinite(p) and np.isfinite(q):
            return "PRIOR_ROLE_ABOVE_PROJECTED" if q > p else "PRIOR_ROLE_NOT_ABOVE_PROJECTED"
    if week == 5:
        if market == "rush_yards":
            a = row.get("player_rush_share", np.nan); b = row.get("rules_rush_share", np.nan)
        elif market in {"rec_yards", "receptions"}:
            a = row.get("player_tgt_share", np.nan); b = row.get("rules_tgt_share", np.nan)
        else:
            return "NOT_APPLICABLE"
        if np.isfinite(a) and np.isfinite(b):
            return "CURRENT_ROLE_ABOVE_RULES" if a > b else "CURRENT_ROLE_NOT_ABOVE_RULES"
    return "SOURCE_UNAVAILABLE"


def final_component_state(row: pd.Series) -> str:
    if not bool(row.get("exact_component_authority", False)):
        return "INSUFFICIENT_COMPONENT_EVIDENCE"
    vals = {
        "OPPORTUNITY_DOMINANT_FINAL": abs(float(row.get("opportunity_contribution", 0.0))),
        "EFFICIENCY_DOMINANT_FINAL": abs(float(row.get("efficiency_contribution", 0.0))),
        "DOWNSTREAM_TRANSMISSION_DOMINANT_FINAL": abs(float(row.get("downstream_contribution", 0.0))),
    }
    ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    if ordered[0][1] >= 1.5 * max(ordered[1][1], 1e-12):
        return ordered[0][0]
    return "MIXED_FINAL"


def attribution(row: pd.Series) -> tuple[str, str]:
    state = final_component_state(row)
    rs = role_signal(row)
    if state == "OPPORTUNITY_DOMINANT_FINAL":
        return "OPPORTUNITY_QUANTITY_MISREAD", rs
    if state == "EFFICIENCY_DOMINANT_FINAL":
        return "EFFICIENCY_MISREAD", "EFFICIENCY_DOMINATES_FINAL_ERROR_IDENTITY"
    if state == "DOWNSTREAM_TRANSMISSION_DOMINANT_FINAL":
        return "MIXED", f"DOWNSTREAM_TRANSMISSION_DOMINANT__{rs}"
    if state == "MIXED_FINAL":
        return "MIXED", rs
    return "SOURCE_BLOCKED_OR_UNAVAILABLE", "NO_EXACT_FINAL_ERROR_COMPONENT_IDENTITY"


def make_casebook(all_rows: pd.DataFrame) -> pd.DataFrame:
    w4 = all_rows.loc[all_rows["source_week"].eq(4)].copy()
    w4 = w4.sort_values(["abs_error", "player_clean_key", "market"], ascending=[False, True, True]).head(40)
    w4["case_rank_week"] = np.arange(1, len(w4) + 1)
    w5 = all_rows.loc[all_rows["source_week"].eq(5) & all_rows["case_rank"].notna()].copy()
    w5["case_rank_week"] = num(w5["case_rank"]).astype("Int64")
    cases = pd.concat([w4, w5], ignore_index=True, sort=False)
    attrs = cases.apply(attribution, axis=1, result_type="expand")
    attrs.columns = ["primary_attribution", "secondary_evidence"]
    cases = pd.concat([cases.reset_index(drop=True), attrs.reset_index(drop=True)], axis=1)
    cases["role_signal"] = cases.apply(role_signal, axis=1)
    cases["final_component_state"] = cases.apply(final_component_state, axis=1)
    cases["strict_prior_wr_dual_rule_eligible"] = (
        cases["source_week"].eq(5) & cases["position_family"].eq("WR") & num(cases["prior_same_market_games"]).ge(4)
    )
    cases["week4_exact_wr_dual_score_manufactured"] = False
    cases["player_conditioned_matchup_authority"] = "SOURCE_BLOCKED_OR_UNAVAILABLE"
    cases["exact_teammate_availability_preserved"] = cases.get("injury_status", pd.Series(index=cases.index, dtype=object)).notna() | cases.get("injury_designation", pd.Series(index=cases.index, dtype=object)).notna()
    return cases


def make_controls(all_rows: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    case_keys = set(zip(cases["source_week"], cases["player_clean_key"], cases["market"]))
    rows = []
    for _, c in cases.iterrows():
        pool = all_rows.loc[
            all_rows["source_week"].eq(c["source_week"])
            & all_rows["market"].eq(c["market"])
            & all_rows["position_family"].eq(c["position_family"])
        ].copy()
        pool = pool.loc[~pool.apply(lambda r: (r["source_week"], r["player_clean_key"], r["market"]) in case_keys, axis=1)]
        med = pool["abs_error"].median()
        good = pool.loc[pool["abs_error"].le(med)].copy()
        if good.empty:
            good = pool.copy()
        if good.empty:
            continue
        basis = "pred_opportunity" if np.isfinite(c.get("pred_opportunity", np.nan)) and good["pred_opportunity"].notna().any() else "model_proj"
        good["match_distance"] = (num(good[basis]) - float(c.get(basis, np.nan))).abs()
        good = good.sort_values(["match_distance", "abs_error", "player_clean_key"])
        g = good.iloc[0]
        rows.append({
            "case_week": int(c["source_week"]), "case_rank_week": int(c["case_rank_week"]),
            "case_player": c["player"], "case_player_clean_key": c["player_clean_key"], "case_market": c["market"],
            "case_position_family": c["position_family"], "case_model_proj": c["model_proj"], "case_actual": c["actual"],
            "case_abs_error": c["abs_error"], "case_primary_attribution": c["primary_attribution"],
            "control_player": g["player"], "control_player_clean_key": g["player_clean_key"],
            "control_model_proj": g["model_proj"], "control_actual": g["actual"], "control_abs_error": g["abs_error"],
            "control_pred_opportunity": g.get("pred_opportunity", np.nan), "control_prior_team_share_last": g.get("prior_team_share_last", np.nan),
            "match_basis": basis, "match_distance": g["match_distance"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1w4-components", type=Path, required=True)
    ap.add_argument("--w4-board", type=Path, required=True)
    ap.add_argument("--w5-settled", type=Path, required=True)
    ap.add_argument("--w5-state", type=Path, required=True)
    ap.add_argument("--w5-cases", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    history, rep = prepare_history(read_csv(a.w1w4_components, "W1-W4 components"))
    week4_actuals = load_actuals(4)
    week5_actuals = load_actuals(5)
    w4 = build_week4(
        read_csv(a.w4_board, "Week-4 preserved board"),
        history,
        rep,
        week4_actuals,
    )
    w5_source = read_csv(a.w5_settled, "Week-5 settled board")
    w5_source_scope = w5_source.loc[
        num(w5_source["season"]).eq(SEASON)
        & num(w5_source["week"]).eq(5)
        & w5_source["market"].isin(ALL_W5)
    ].copy()
    w5 = build_week5(
        w5_source,
        read_csv(a.w5_state, "Week-5 player state"),
        read_csv(a.w5_cases, "Week-5 frozen casebook"),
        week5_actuals,
    )
    common = sorted(set(w4.columns) | set(w5.columns))
    all_rows = pd.concat([w4.reindex(columns=common), w5.reindex(columns=common)], ignore_index=True, sort=False)
    all_rows = attach_prior(history, rep, all_rows)
    all_rows["role_signal"] = all_rows.apply(role_signal, axis=1)
    all_rows["final_component_state"] = all_rows.apply(final_component_state, axis=1)
    all_rows["signed_error"] = all_rows["model_proj"] - all_rows["actual"]
    all_rows["market_error_median"] = all_rows.groupby(["source_week", "market"])["abs_error"].transform("median")
    all_rows["error_to_market_median"] = all_rows["abs_error"] / all_rows["market_error_median"].replace(0, np.nan)

    cases = make_casebook(all_rows)
    controls = make_controls(all_rows, cases)
    mech = cases.groupby(["primary_attribution", "secondary_evidence"], dropna=False).size().reset_index(name="case_rows").sort_values("case_rows", ascending=False)
    qa = all_rows.groupby(["source_week", "market", "position_family"], dropna=False).agg(
        rows=("player_clean_key", "size"),
        mae=("abs_error", "mean"),
        rmse=("signed_error", lambda z: float(np.sqrt(np.mean(np.square(num(z).dropna())))) if len(num(z).dropna()) else np.nan),
    ).reset_index()

    signals = cases.loc[
        cases["role_signal"].isin({"PRIOR_ROLE_ABOVE_PROJECTED", "CURRENT_ROLE_ABOVE_RULES"}),
        ["source_week", "case_rank_week", "player", "team", "market", "position_family", "role_signal", "primary_attribution",
         "prior_team_share_last", "predicted_team_share", "player_tgt_share", "rules_tgt_share", "player_rush_share", "rules_rush_share",
         "pred_opportunity", "actual_opportunity", "model_proj", "actual"],
    ].copy()
    blocked = cases.loc[
        cases["player_conditioned_matchup_authority"].eq("SOURCE_BLOCKED_OR_UNAVAILABLE")
        | cases["exact_teammate_availability_preserved"].eq(False),
        ["source_week", "case_rank_week", "player", "team", "market", "position_family", "player_conditioned_matchup_authority", "exact_teammate_availability_preserved"],
    ].copy()

    summary = {
        "version": VERSION,
        "status": "PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_EXECUTED__NO_PROMOTION",
        "all_row_total": int(len(all_rows)),
        "week4_all_rows": int(all_rows["source_week"].eq(4).sum()),
        "week5_all_rows": int(all_rows["source_week"].eq(5).sum()),
        "week5_source_scope_rows": int(len(w5_source_scope)),
        "week5_nonsettled_rows_excluded": int(
            (~w5_source_scope["football_settlement_status"].eq("SETTLED")).sum()
        ),
        "week5_nonsettled_status_counts": {
            str(k): int(v)
            for k, v in w5_source_scope.loc[
                ~w5_source_scope["football_settlement_status"].eq("SETTLED"),
                "football_settlement_status",
            ].value_counts(dropna=False).items()
        },
        "week4_exact_component_rows": int((all_rows["source_week"].eq(4) & all_rows["exact_component_authority"].eq(True)).sum()),
        "week5_exact_component_rows": int((all_rows["source_week"].eq(5) & all_rows["exact_component_authority"].eq(True)).sum()),
        "max_final_identity_gap": float(num(all_rows.loc[all_rows["exact_component_authority"].eq(True), "final_identity_gap"]).max()),
        "casebook_downstream_dominant_rows": int(cases["final_component_state"].eq("DOWNSTREAM_TRANSMISSION_DOMINANT_FINAL").sum()),
        "week5_direct_rows_with_actual_opportunity": int((all_rows["source_week"].eq(5) & all_rows["market"].isin(DIRECT) & all_rows["actual_opportunity"].notna()).sum()),
        "casebook_rows": int(len(cases)),
        "matched_control_rows": int(len(controls)),
        "pregame_role_signals_above_projection_or_rules": int(len(signals)),
        "source_blocked_case_rows": int(len(blocked)),
        "week5_four_prior_same_market_rows": int((all_rows["source_week"].eq(5) & num(all_rows["prior_same_market_games"]).ge(4)).sum()),
        "week4_wr_dual_score_manufactured": False,
        "week6_outcomes_used": 0,
        "sportsbook_refetch": False,
        "sportsbook_predictor_used": False,
        "oddsapi_calls": 0,
        "coefficients_fit": 0,
        "thresholds_fit": 0,
        "production_changed": False,
        "boundary": "retrospective player-level forensic diagnosis only; any discovered mechanism requires separate freeze and independent validation",
    }

    all_rows.to_csv(a.out_dir / "prism_week4_5_all_row_replay.csv", index=False, float_format="%.12f")
    cases.to_csv(a.out_dir / "prism_week4_5_named_player_attribution_casebook.csv", index=False, float_format="%.12f")
    controls.to_csv(a.out_dir / "prism_week4_5_matched_correct_controls.csv", index=False, float_format="%.12f")
    mech.to_csv(a.out_dir / "prism_week4_5_repeated_mechanism_summary.csv", index=False)
    signals.to_csv(a.out_dir / "prism_week4_5_pregame_signals_not_transmitted.csv", index=False, float_format="%.12f")
    blocked.to_csv(a.out_dir / "prism_week4_5_source_blocked_signals.csv", index=False)
    qa.to_csv(a.out_dir / "prism_week4_5_aggregate_qa.csv", index=False, float_format="%.12f")
    (a.out_dir / "prism_week4_5_player_first_forensic_replay_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
