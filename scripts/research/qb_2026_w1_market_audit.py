#!/usr/bin/env python3
"""Mechanical harness for the 2026 Week-1 QB no-odds market audit.

The football model is run first against two deliberately different synthetic
lines.  The promoted QB mean must be invariant to those lines before any actual
sportsbook number is read.  Only then may the manually transcribed sportsbook
lines be materialized and compared downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_PLAYERS = Path("data/manual_market/qb_2026_w1_target_players.csv")
ACTUAL_LINES = Path("data/manual_market/qb_2026_w1_fanduel_lines.csv")
PROPS_RAW = Path("outputs/props_raw.csv")
FROZEN = Path("data/qb_2026_w1_no_odds_projections.csv")
COMPARISON = Path("data/qb_2026_w1_market_comparison.csv")


def _key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    return df


def _event_id(team: str, opponent: str) -> str:
    return "|".join(sorted([str(team).strip().upper(), str(opponent).strip().upper()]))


def write_synthetic_props(line: float, odds: int = -110) -> pd.DataFrame:
    """Create target rows with a dummy line that has no football meaning."""
    src = _read(TARGET_PLAYERS, "target player universe")
    required = {"player", "team", "opponent"}
    if not required.issubset(src.columns):
        raise RuntimeError(f"target universe missing {sorted(required - set(src.columns))}")
    if len(src) != 31:
        raise RuntimeError(f"expected 31 posted QB target players, got {len(src)}")
    if src[["player", "team"]].duplicated().any():
        raise RuntimeError("duplicate target QB identity")
    out = src.copy()
    out["event_id"] = [_event_id(t, o) for t, o in zip(out["team"], out["opponent"])]
    out["market"] = "pass_yards"
    out["line"] = float(line)
    out["over_odds"] = int(odds)
    out["under_odds"] = int(odds)
    out["book"] = "SYNTHETIC_QB_AUDIT"
    out["book_title"] = "Synthetic QB Audit"
    PROPS_RAW.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROPS_RAW, index=False)
    print(f"[qb-w1-audit] wrote synthetic target rows={len(out)} line={line} -> {PROPS_RAW}")
    return out


def write_actual_props() -> pd.DataFrame:
    """Materialize the manual sportsbook lines only after football freeze."""
    src = _read(ACTUAL_LINES, "manual sportsbook lines")
    required = {"player", "team", "opponent", "line", "over_odds", "under_odds", "book"}
    if not required.issubset(src.columns):
        raise RuntimeError(f"manual lines missing {sorted(required - set(src.columns))}")
    if len(src) != 31:
        raise RuntimeError(f"expected 31 manual posted QB lines, got {len(src)}")
    out = src.copy()
    out["event_id"] = [_event_id(t, o) for t, o in zip(out["team"], out["opponent"])]
    out["market"] = "pass_yards"
    out["book_title"] = out["book"]
    PROPS_RAW.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROPS_RAW, index=False)
    print(f"[qb-w1-audit] wrote ACTUAL downstream sportsbook rows={len(out)} -> {PROPS_RAW}")
    return out


def _unique_qb(priced_path: Path) -> pd.DataFrame:
    x = _read(priced_path, f"priced output {priced_path}")
    x = x.loc[x["market"].astype(str).str.lower().eq("pass_yards")].copy()
    # One OVER row carries the same football mean and audit metadata as UNDER.
    over = x.loc[x["side"].astype(str).str.upper().eq("OVER")].copy()
    over["player_key_local"] = over["player"].map(_key)
    if over["player_key_local"].duplicated().any():
        raise RuntimeError(f"duplicate QB pricing rows in {priced_path}")
    return over.reset_index(drop=True)


def freeze_projection(a_path: Path, b_path: Path, out_path: Path = FROZEN) -> pd.DataFrame:
    """Prove line invariance and freeze the football-only promoted QB means."""
    a = _unique_qb(a_path)
    b = _unique_qb(b_path)
    if len(a) != 31 or len(b) != 31:
        raise RuntimeError(f"expected 31 QB priced rows in both synthetic passes; got a={len(a)} b={len(b)}")
    joined = a.merge(
        b[["player_key_local", "model_proj", "qb_synthesis_proj"]].rename(
            columns={"model_proj": "model_proj_b", "qb_synthesis_proj": "qb_synthesis_proj_b"}
        ),
        on="player_key_local",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 31:
        raise RuntimeError(f"synthetic line passes did not identity-match all QBs: {len(joined)}")
    model_a = pd.to_numeric(joined["model_proj"], errors="coerce")
    model_b = pd.to_numeric(joined["model_proj_b"], errors="coerce")
    synth_a = pd.to_numeric(joined["qb_synthesis_proj"], errors="coerce")
    synth_b = pd.to_numeric(joined["qb_synthesis_proj_b"], errors="coerce")
    if model_a.isna().any() or model_b.isna().any() or synth_a.isna().any() or synth_b.isna().any():
        raise RuntimeError("non-finite promoted QB projection in synthetic-line gate")
    max_line_diff = float(np.max(np.abs(model_a - model_b)))
    if max_line_diff > 1e-8:
        raise RuntimeError(f"QB model projection changed with synthetic line: max_diff={max_line_diff}")
    if float(np.max(np.abs(model_a - synth_a))) > 1e-8:
        raise RuntimeError("final football mean != promoted QB synthesis mean in synthetic pass A")
    if float(np.max(np.abs(synth_a - synth_b))) > 1e-8:
        raise RuntimeError("promoted QB synthesis changed with synthetic line")
    if not pd.to_numeric(joined["qb_synthesis_applied"], errors="coerce").eq(1).all():
        raise RuntimeError("promoted QB synthesis not applied to every row")
    conv = pd.to_numeric(joined["qb_attempt_conversion"], errors="coerce")
    if conv.isna().any() or not conv.between(0.50, 1.00, inclusive="both").all():
        raise RuntimeError("QB attempt conversion missing/out of range")

    keep = [
        "event_id", "player", "player_clean_key", "team", "opponent", "season", "week",
        "market", "model_proj", "mc_proj", "model_sd", "simulation_iterations",
        "ensemble_proj", "ensemble_status", "ensemble_method", "ensemble_weight_mc",
        "ensemble_weight_ml", "ensemble_weight_state", "ensemble_calibration_rows",
        "ml_proj", "ml_applied", "ml_method", "ml_training_cutoff",
        "state_proj", "state_applied", "state_method", "state_training_cutoff",
        "bayes_applied", "bayes_evidence_state", "rules_applied", "rules_role",
        "qb_synthesis_applied", "qb_synthesis_proj", "qb_synthesis_correction",
        "qb_synthesis_version", "qb_attempt_conversion", "qb_pass_att_share",
        "qb_pred_attempts", "qb_pred_ypa",
    ]
    keep = [c for c in keep if c in joined.columns]
    frozen = joined[keep].copy()
    frozen["football_only_no_odds"] = 1
    frozen["sportsbook_inputs_used"] = 0
    frozen["synthetic_line_invariance_max_abs_diff"] = max_line_diff
    frozen = frozen.sort_values(["team", "player"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(out_path, index=False)
    print(
        f"[qb-w1-audit] FOOTBALL FREEZE PASS rows={len(frozen)} teams={frozen['team'].nunique()} "
        f"line_invariance_max_diff={max_line_diff:.12g} -> {out_path}"
    )
    return frozen


def compare_market(priced_path: Path, frozen_path: Path = FROZEN, out_path: Path = COMPARISON) -> pd.DataFrame:
    """Join real posted lines after freeze and assert football means did not move."""
    frozen = _read(frozen_path, "frozen football projections")
    if not pd.to_numeric(frozen.get("sportsbook_inputs_used", 1), errors="coerce").eq(0).all():
        raise RuntimeError("frozen projection does not prove sportsbook_inputs_used=0")
    priced = _read(priced_path, "actual-line priced output")
    qb = priced.loc[priced["market"].astype(str).str.lower().eq("pass_yards")].copy()
    qb["player_key_local"] = qb["player"].map(_key)
    frozen["player_key_local"] = frozen["player"].map(_key)

    over = qb.loc[qb["side"].astype(str).str.upper().eq("OVER")].copy()
    under = qb.loc[qb["side"].astype(str).str.upper().eq("UNDER")].copy()
    over = over.rename(columns={"fair_prob": "model_p_over", "market_prob": "market_p_over", "edge_pct": "edge_over"})
    under = under[["player_key_local", "fair_prob", "market_prob", "edge_pct"]].rename(
        columns={"fair_prob": "model_p_under", "market_prob": "market_p_under", "edge_pct": "edge_under"}
    )
    out = over.merge(under, on="player_key_local", how="inner", validate="one_to_one")
    out = out.merge(
        frozen[["player_key_local", "model_proj"]].rename(columns={"model_proj": "frozen_model_proj"}),
        on="player_key_local", how="inner", validate="one_to_one"
    )
    if len(out) != 31:
        raise RuntimeError(f"expected 31 downstream QB market rows, got {len(out)}")
    actual_proj = pd.to_numeric(out["model_proj"], errors="coerce")
    frozen_proj = pd.to_numeric(out["frozen_model_proj"], errors="coerce")
    max_diff = float(np.max(np.abs(actual_proj - frozen_proj)))
    if max_diff > 1e-8:
        raise RuntimeError(f"actual sportsbook line changed football projection: max_diff={max_diff}")
    out["projection_minus_line"] = actual_proj - pd.to_numeric(out["vegas_line"], errors="coerce")
    out["football_projection_frozen_before_market"] = 1
    out["sportsbook_inputs_used_in_football_projection"] = 0
    out["actual_line_projection_invariance_max_abs_diff"] = max_diff
    cols = [
        "player", "team", "opponent", "vegas_line", "model_proj", "projection_minus_line",
        "model_sd", "qb_pred_attempts", "qb_pred_ypa", "qb_synthesis_correction",
        "model_p_over", "model_p_under", "market_p_over", "market_p_under",
        "edge_over", "edge_under", "vegas_over_odds", "vegas_under_odds",
        "ensemble_proj", "mc_proj", "ml_proj", "state_proj", "qb_synthesis_version",
        "simulation_iterations", "football_projection_frozen_before_market",
        "sportsbook_inputs_used_in_football_projection", "actual_line_projection_invariance_max_abs_diff",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values("projection_minus_line", ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[qb-w1-audit] downstream market comparison PASS rows={len(out)} max_projection_diff={max_diff:.12g} -> {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser("write-synthetic")
    p_syn.add_argument("--line", type=float, required=True)
    p_syn.add_argument("--odds", type=int, default=-110)

    sub.add_parser("write-actual")

    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--a", type=Path, required=True)
    p_freeze.add_argument("--b", type=Path, required=True)
    p_freeze.add_argument("--out", type=Path, default=FROZEN)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--priced", type=Path, required=True)
    p_cmp.add_argument("--frozen", type=Path, default=FROZEN)
    p_cmp.add_argument("--out", type=Path, default=COMPARISON)

    args = ap.parse_args()
    if args.cmd == "write-synthetic":
        write_synthetic_props(args.line, args.odds)
    elif args.cmd == "write-actual":
        write_actual_props()
    elif args.cmd == "freeze":
        freeze_projection(args.a, args.b, args.out)
    elif args.cmd == "compare":
        compare_market(args.priced, args.frozen, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
