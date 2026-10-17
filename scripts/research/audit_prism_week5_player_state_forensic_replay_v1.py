#!/usr/bin/env python3
"""PRISM Week-5 named-player situation forensic replay V1.

Retrospective QA only. Combines preserved strict-prior W1-W4 player history,
Week-5 pregame player-state parity evidence, the settled Week-5 full board, and
the frozen Week-5 top-failure casebook.

No model fitting, no sportsbook predictor, no production promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_2026_WEEK5_PLAYER_STATE_FORENSIC_REPLAY_V1"
SEASON = 2026
TARGET_WEEK = 5
PRIOR_WEEKS = (1, 2, 3, 4)
DIRECT_MARKETS = {"pass_yards", "rush_yards", "rec_yards", "receptions"}
SKILL_DIRECT_MARKETS = {"rush_yards", "rec_yards", "receptions"}
DOMINANCE_RATIO = 1.5


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _w4_component_label(row: pd.Series) -> str:
    o = abs(float(row["w4_opportunity_contribution"]))
    e = abs(float(row["w4_efficiency_contribution"]))
    if not np.isfinite(o) or not np.isfinite(e):
        return "UNRESOLVED"
    hi, lo = max(o, e), min(o, e)
    if hi <= 1e-12:
        return "MIXED_OPPORTUNITY_EFFICIENCY"
    ratio = hi / max(lo, 1e-12)
    if ratio < DOMINANCE_RATIO:
        return "MIXED_OPPORTUNITY_EFFICIENCY"
    return "OPPORTUNITY_DOMINANT" if o > e else "EFFICIENCY_DOMINANT"


def build(
    history: pd.DataFrame,
    target: pd.DataFrame,
    cases: pd.DataFrame,
    player_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    hist_need = {
        "season", "week", "event_id", "team", "player", "player_clean_key",
        "position_family", "market", "predicted_opportunities",
        "actual_opportunities", "model_effective_efficiency",
        "actual_efficiency", "baseline_projection", "baseline_reconstructed",
        "actual_output", "baseline_error", "grading_identity_valid",
        "component_decomposition_eligible",
    }
    target_need = {
        "season", "week", "event_id", "team", "opponent", "player",
        "player_clean_key", "position", "market", "model_proj", "actual",
        "football_settlement_status",
    }
    case_need = {
        "player_clean_key", "player", "team", "opponent", "position",
        "market", "model_proj", "actual", "abs_error", "component_label",
        "case_rank", "pred_opportunity", "actual_opportunity",
        "pred_efficiency", "actual_efficiency",
    }
    state_need = {
        "team", "opponent", "player_clean_key", "player", "position",
        "rules_tgt_share", "rules_rush_share", "player_tgt_share",
        "player_rush_share", "expected_targets", "expected_carries",
        "rules_pass_eff_mult", "rules_rush_eff_mult", "off_proe",
        "opp_def_pass_epa", "opp_def_rush_epa", "opp_success_rate_def",
        "opp_pressure_generated", "opp_explosive_allowed", "injury_status",
        "injury_designation",
    }
    for label, frame, need in (
        ("history", history, hist_need),
        ("target", target, target_need),
        ("cases", cases, case_need),
        ("player_state", player_state, state_need),
    ):
        miss = need - set(frame.columns)
        if miss:
            raise RuntimeError(f"{label} missing {sorted(miss)}")

    h = history.copy()
    h["season"] = num(h["season"])
    h["week"] = num(h["week"])
    h = h.loc[
        h["season"].eq(SEASON)
        & h["week"].isin(PRIOR_WEEKS)
        & h["market"].isin(DIRECT_MARKETS)
        & h["grading_identity_valid"].eq(True)
        & h["component_decomposition_eligible"].eq(True)
    ].copy()
    h["prior_abs_error"] = num(h["baseline_error"]).abs()
    prior = (
        h.groupby(["player_clean_key", "market"], as_index=False)
        .agg(
            prior_games=("week", "nunique"),
            prior_min_week=("week", "min"),
            prior_max_week=("week", "max"),
            prior_player_mae=("prior_abs_error", "mean"),
        )
    )
    prior = prior.loc[
        prior["prior_games"].eq(4)
        & prior["prior_min_week"].eq(1)
        & prior["prior_max_week"].eq(4)
    ].copy()

    t = target.copy()
    t["season"] = num(t["season"])
    t["week"] = num(t["week"])
    t = t.loc[
        t["season"].eq(SEASON)
        & t["week"].eq(TARGET_WEEK)
        & t["football_settlement_status"].eq("SETTLED")
        & t["market"].isin(DIRECT_MARKETS)
    ].copy()
    t["w5_abs_error"] = (num(t["model_proj"]) - num(t["actual"])).abs()
    replay = t.merge(prior, on=["player_clean_key", "market"], how="inner")
    if replay.empty:
        raise RuntimeError("zero strict-prior Week-5 replay rows")
    replay["prior_fragility_pct"] = replay.groupby("market")[
        "prior_player_mae"
    ].rank(method="average", pct=True)
    replay["w5_error_pct"] = replay.groupby("market")["w5_abs_error"].rank(
        method="average", pct=True
    )

    s = player_state.copy()
    for col in (
        "rules_tgt_share", "rules_rush_share", "player_tgt_share",
        "player_rush_share", "expected_targets", "expected_carries",
        "rules_pass_eff_mult", "rules_rush_eff_mult", "off_proe",
        "opp_def_pass_epa", "opp_def_rush_epa", "opp_success_rate_def",
        "opp_pressure_generated", "opp_explosive_allowed",
    ):
        s[col] = num(s[col])
    if s.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("duplicate Week-5 player-state team/player identity")

    s["team_expected_targets_total"] = s.groupby("team")["expected_targets"].transform(
        lambda z: z.fillna(0).sum()
    )
    s["team_expected_carries_total"] = s.groupby("team")["expected_carries"].transform(
        lambda z: z.fillna(0).sum()
    )
    s["target_room_rank"] = s.groupby("team")["expected_targets"].rank(
        method="min", ascending=False
    )
    s["carry_room_rank"] = s.groupby("team")["expected_carries"].rank(
        method="min", ascending=False
    )
    s["target_room_share"] = s["expected_targets"] / s[
        "team_expected_targets_total"
    ].replace(0, np.nan)
    s["carry_room_share"] = s["expected_carries"] / s[
        "team_expected_carries_total"
    ].replace(0, np.nan)

    replay = replay.merge(
        s,
        on=["team", "player_clean_key"],
        how="left",
        suffixes=("", "_state"),
        validate="many_to_one",
    )
    if replay["opponent_state"].isna().any():
        raise RuntimeError("missing Week-5 player state on strict-prior replay rows")

    c = cases.copy()
    if c["case_rank"].duplicated().any():
        raise RuntimeError("duplicate Week-5 case rank")
    if len(c) != 40:
        raise RuntimeError(f"expected frozen top-40 Week-5 cases, got {len(c)}")
    case_state = c.merge(
        s,
        on=["team", "player_clean_key"],
        how="left",
        suffixes=("", "_state"),
        validate="many_to_one",
    )
    if case_state["opponent_state"].isna().any():
        raise RuntimeError("missing Week-5 player state on casebook row")
    case_state = case_state.merge(
        replay[["player_clean_key", "market", "prior_player_mae", "prior_fragility_pct"]],
        on=["player_clean_key", "market"],
        how="left",
        validate="many_to_one",
    )
    case_state["strict_prior_4week_reliability_supported"] = case_state[
        "prior_player_mae"
    ].notna()
    case_state["opportunity_error_signed"] = (
        num(case_state["actual_opportunity"]) - num(case_state["pred_opportunity"])
    )
    case_state["current_minus_rules_role_share"] = np.where(
        case_state["market"].eq("rush_yards"),
        case_state["player_rush_share"] - case_state["rules_rush_share"],
        np.where(
            case_state["market"].isin({"rec_yards", "receptions"}),
            case_state["player_tgt_share"] - case_state["rules_tgt_share"],
            np.nan,
        ),
    )
    case_state["relevant_opponent_metric_available"] = np.where(
        case_state["market"].eq("rush_yards"),
        case_state["opp_def_rush_epa"].notna(),
        np.where(
            case_state["market"].isin({"rec_yards", "receptions", "pass_yards"}),
            case_state["opp_def_pass_epa"].notna(),
            False,
        ),
    )
    case_state["relevant_rules_eff_mult"] = np.where(
        case_state["market"].eq("rush_yards"),
        case_state["rules_rush_eff_mult"],
        np.where(
            case_state["market"].isin({"rec_yards", "receptions", "pass_yards"}),
            case_state["rules_pass_eff_mult"],
            np.nan,
        ),
    )

    w4 = h.loc[h["week"].eq(4)].copy()
    w4["w4_abs_error"] = (
        num(w4["baseline_projection"]) - num(w4["actual_output"])
    ).abs()
    pn = num(w4["predicted_opportunities"])
    an = num(w4["actual_opportunities"])
    pe = num(w4["model_effective_efficiency"])
    ae = num(w4["actual_efficiency"])
    w4["w4_opportunity_contribution"] = (an - pn) * (pe + ae) / 2.0
    w4["w4_efficiency_contribution"] = (ae - pe) * (pn + an) / 2.0
    identity_gap = (
        w4["w4_opportunity_contribution"]
        + w4["w4_efficiency_contribution"]
        - (num(w4["actual_output"]) - num(w4["baseline_reconstructed"]))
    ).abs()
    max_gap = float(identity_gap.max())
    if max_gap > 1e-8:
        raise RuntimeError(f"Week-4 Shapley identity gap {max_gap}")
    w4["component_label_w4"] = w4.apply(_w4_component_label, axis=1)
    w4_top = w4.sort_values(
        ["w4_abs_error", "player_clean_key", "market"],
        ascending=[False, True, True],
    ).head(40).copy()
    repeats = w4_top.merge(
        c[["player_clean_key", "market", "component_label", "abs_error", "case_rank"]],
        on=["player_clean_key", "market"],
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "component_label": "component_label_w5",
            "abs_error": "w5_abs_error",
        }
    )
    repeats["component_state_changed"] = (
        repeats["component_label_w4"] != repeats["component_label_w5"]
    )

    skill = case_state.loc[case_state["market"].isin(SKILL_DIRECT_MARKETS)].copy()
    opp_dom = skill.loc[skill["component_label"].eq("OPPORTUNITY_DOMINANT")].copy()
    if opp_dom.empty:
        raise RuntimeError("zero opportunity-dominant direct skill cases")

    relevant_rec = skill["market"].isin({"rec_yards", "receptions"})
    relevant_rush = skill["market"].eq("rush_yards")
    neutral_mult = (
        (relevant_rec & num(skill["rules_pass_eff_mult"]).sub(1.0).abs().le(1e-12))
        | (relevant_rush & num(skill["rules_rush_eff_mult"]).sub(1.0).abs().le(1e-12))
    )

    result = {
        "version": VERSION,
        "status": "PLAYER_STATE_FORENSIC_REPLAY_COMPLETE__NO_PROMOTION",
        "season": SEASON,
        "target_week": TARGET_WEEK,
        "strict_prior_weeks": list(PRIOR_WEEKS),
        "strict_prior_replay_rows": int(len(replay)),
        "strict_prior_unique_players": int(replay["player_clean_key"].nunique()),
        "week5_frozen_failure_cases": int(len(case_state)),
        "week5_failure_cases_with_4week_reliability": int(
            case_state["strict_prior_4week_reliability_supported"].sum()
        ),
        "week5_direct_skill_failure_cases": int(len(skill)),
        "week5_direct_skill_cases_opportunity_underpredicted": int(
            (skill["opportunity_error_signed"] > 0).sum()
        ),
        "week5_direct_skill_opportunity_dominant_cases": int(len(opp_dom)),
        "week5_opp_dominant_with_opportunity_underpredicted": int(
            (opp_dom["opportunity_error_signed"] > 0).sum()
        ),
        "week5_opp_dominant_current_share_above_rules": int(
            (opp_dom["current_minus_rules_role_share"] > 0).sum()
        ),
        "week5_opp_dominant_median_current_minus_rules_share": float(
            opp_dom["current_minus_rules_role_share"].median()
        ),
        "week5_direct_skill_relevant_opponent_metric_available": int(
            skill["relevant_opponent_metric_available"].sum()
        ),
        "week5_direct_skill_rules_eff_mult_exactly_neutral": int(neutral_mult.sum()),
        "week5_case_injury_status_nonnull": int(case_state["injury_status"].notna().sum()),
        "week5_case_injury_designation_nonnull": int(
            case_state["injury_designation"].notna().sum()
        ),
        "w4_w5_same_player_market_top40_repeats": int(len(repeats)),
        "w4_w5_repeat_component_state_changes": int(
            repeats["component_state_changed"].sum()
        ),
        "w4_w5_repeat_component_state_change_rate": float(
            repeats["component_state_changed"].mean() if len(repeats) else np.nan
        ),
        "w4_shapley_max_identity_gap": max_gap,
        "direct_recent_share_replacement_promoted": False,
        "new_model_fit": 0,
        "thresholds_fit": 0,
        "sportsbook_predictor_used": False,
        "oddsapi_calls": 0,
        "production_changed": False,
        "boundary": (
            "forensic current-season QA only; target outcomes were known before "
            "this analysis. Existing terminal/blocked families remain protected."
        ),
    }

    for name, frame in (
        ("replay", replay), ("case_state", case_state), ("repeats", repeats)
    ):
        payload = frame.sort_values(list(frame.columns[:1])).to_csv(
            index=False, float_format="%.12f", lineterminator="\n"
        ).encode()
        result[f"{name}_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()

    return replay, case_state, repeats, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--failure-cases", type=Path, required=True)
    ap.add_argument("--player-state", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    replay, cases, repeats, result = build(
        read_csv(args.history, "W1-W4 decomposition"),
        read_csv(args.target, "Week-5 full-board settlement"),
        read_csv(args.failure_cases, "Week-5 failure casebook"),
        read_csv(args.player_state, "Week-5 player state"),
    )
    replay.to_csv(args.out_dir / "prism_week5_player_state_replay_rows.csv", index=False)
    cases.to_csv(args.out_dir / "prism_week5_player_state_failure_cases.csv", index=False)
    repeats.to_csv(args.out_dir / "prism_week4_week5_repeat_failure_states.csv", index=False)
    (args.out_dir / "prism_week5_player_state_forensic_replay_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
