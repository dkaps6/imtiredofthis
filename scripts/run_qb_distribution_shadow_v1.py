#!/usr/bin/env python3
"""Shadow-audit the deployable Phase-J QB distribution selector on a live Full Slate.

This script never changes production pricing output.  It reconstructs the exact
canonical QB Monte Carlo from the rule-layer inputs, verifies parity to the
priced ``mc_proj``, builds the frozen C2 QB distribution anchored to the exact
M89/M90 production mean, applies the deployable Phase-J selector, and writes an
auditable sidecar.

Sportsbook fields may be read only after both football distributions exist, for
reporting probability deltas.  They are never selector/model inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.qb_distribution_state_v1 import load_artifact, select_c2
from scripts.simulation_c2_qb_candidate import apply_c2, lookup, simulate_with_states

DATA = Path("data")
OUTPUTS = Path("outputs")
RULE_INPUTS = DATA / "model_rule_simulation_inputs.csv"
PRICED = OUTPUTS / "props_priced_clean.csv"
STATE_CONTEXT = DATA / "qb_distribution_state_context.csv"
OUT = OUTPUTS / "qb_distribution_shadow_v1.csv"
RESULT = OUTPUTS / "qb_distribution_shadow_v1_result.json"


def _f(v, default=np.nan) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(np.asarray(a, float), p))


def _game_key(row: pd.Series) -> str:
    event = row.get("event_id")
    if pd.notna(event) and str(event).strip():
        return str(event)
    return "|".join(sorted([str(row.get("team", "")), str(row.get("opponent", ""))]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-inputs", default=str(RULE_INPUTS))
    ap.add_argument("--priced", default=str(PRICED))
    ap.add_argument("--state-context", default=str(STATE_CONTEXT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--result", default=str(RESULT))
    args = ap.parse_args()

    for p in [Path(args.rule_inputs), Path(args.priced), Path(args.state_context)]:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"missing shadow input: {p}")

    metrics = pd.read_csv(args.rule_inputs, low_memory=False)
    metrics.columns = [str(c).lower() for c in metrics.columns]
    priced = pd.read_csv(args.priced, low_memory=False)
    priced.columns = [str(c).lower() for c in priced.columns]
    ctx = pd.read_csv(args.state_context, low_memory=False)
    ctx.columns = [str(c).lower() for c in ctx.columns]

    qb = priced.loc[priced["market"].astype(str).str.lower().eq("pass_yards")].copy()
    if qb.empty:
        raise RuntimeError("shadow audit found no priced pass_yards rows")
    # Pricing writes OVER/UNDER rows from the same underlying player distribution.
    qb = qb.sort_values(["event_id", "team", "player_clean_key", "side"]).drop_duplicates(
        ["event_id", "team", "player_clean_key"], keep="first"
    )
    if not pd.to_numeric(qb.get("qb_synthesis_applied", 0), errors="coerce").eq(1).all():
        raise RuntimeError("shadow audit requires promoted M89/M90 QB means on every QB row")

    art = load_artifact()
    if int(art.get("sportsbook_inputs_used", 1)) != 0:
        raise RuntimeError("selector artifact sportsbook leakage flag")
    required_context = [
        "team", "opponent", "pass_opportunity_spot", "pass_efficiency_spot",
        "rush_opportunity_spot", "rush_efficiency_spot", "sportsbook_inputs_used",
    ]
    missing = [c for c in required_context if c not in ctx.columns]
    if missing:
        raise RuntimeError(f"state context missing {missing}")
    if not pd.to_numeric(ctx["sportsbook_inputs_used"], errors="coerce").eq(0).all():
        raise RuntimeError("state context sportsbook leakage flag")
    if ctx.duplicated("team").any():
        raise RuntimeError("state context duplicate teams")

    base = simulate_with_states(metrics)

    # Final promoted means are the C2 anchors.  Team/event uniqueness is a hard
    # contract because C2 exposes only the primary QB distribution per team.
    anchors: dict[tuple[str, str], float] = {}
    for _, r in qb.iterrows():
        key = (_game_key(r), str(r.get("team", "")))
        mean = _f(r.get("model_proj"))
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid QB promoted mean for {key}: {mean}")
        if key in anchors and abs(anchors[key] - mean) > 1e-8:
            raise RuntimeError(f"multiple promoted QB means for team/event {key}")
        anchors[key] = mean

    c2 = apply_c2(base, metrics, anchor_map=anchors, seed=5601)
    ctx_by_team = ctx.set_index("team", drop=False)

    rows = []
    for _, r in qb.iterrows():
        team = str(r.get("team", ""))
        if team not in ctx_by_team.index:
            raise RuntimeError(f"missing distribution-state context for team={team}")
        cr = ctx_by_team.loc[team]
        if isinstance(cr, pd.DataFrame):
            raise RuntimeError(f"duplicate context rows for team={team}")

        raw = lookup(base, r, "pass_yards")
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"canonical shadow QB array missing player={r.get('player')} team={team}")
        raw = np.asarray(raw, float)
        conv = _f(r.get("qb_attempt_conversion"))
        share = _f(r.get("qb_pass_att_share"), 1.0)
        if not np.isfinite(conv) or not 0.50 <= conv <= 1.0:
            raise RuntimeError(f"invalid QB attempt conversion player={r.get('player')}: {conv}")
        if not np.isfinite(share):
            share = 1.0
        share = float(np.clip(share, 0.0, 1.0))
        canonical_component = raw * conv * share
        rebuilt_mc = float(np.mean(canonical_component))
        priced_mc = _f(r.get("mc_proj"))
        if not np.isfinite(priced_mc):
            raise RuntimeError(f"priced mc_proj missing player={r.get('player')}")

        final_mean = _f(r.get("model_proj"))
        if rebuilt_mc <= 0 or not np.isfinite(final_mean):
            raise RuntimeError(f"invalid shadow means player={r.get('player')}")
        canonical = canonical_component * (final_mean / rebuilt_mc)

        feature_values = {
            "pass_opportunity_spot": _f(cr.get("pass_opportunity_spot")),
            "pass_efficiency_spot": _f(cr.get("pass_efficiency_spot")),
            "rush_opportunity_spot": _f(cr.get("rush_opportunity_spot")),
            "rush_efficiency_spot": _f(cr.get("rush_efficiency_spot")),
            "pred_qb_attempts": _f(r.get("qb_pred_attempts")),
            "week": _f(r.get("week")),
        }
        selected, delta, selector_version = select_c2(feature_values, art)
        c2_arr = lookup(c2, r, "pass_yards")
        if selected and (c2_arr is None or len(c2_arr) == 0):
            raise RuntimeError(f"selected C2 QB array missing player={r.get('player')} team={team}")
        chosen = np.asarray(c2_arr, float) if selected else canonical

        line = _f(r.get("vegas_line"))
        canonical_over = float(np.mean(canonical > line)) if np.isfinite(line) else np.nan
        chosen_over = float(np.mean(chosen > line)) if np.isfinite(line) else np.nan
        rows.append({
            "season": int(_f(r.get("season"), 0)),
            "week": int(_f(r.get("week"), 0)),
            "event_id": r.get("event_id"),
            "team": team,
            "opponent": r.get("opponent"),
            "player": r.get("player"),
            "player_clean_key": r.get("player_clean_key"),
            "qb_synthesis_version": r.get("qb_synthesis_version"),
            "selector_version": selector_version,
            "selector_delta_pass_attempts": float(delta),
            "selector_c2_selected": int(selected),
            "pred_qb_attempts": feature_values["pred_qb_attempts"],
            "pass_opportunity_spot": feature_values["pass_opportunity_spot"],
            "pass_efficiency_spot": feature_values["pass_efficiency_spot"],
            "rush_opportunity_spot": feature_values["rush_opportunity_spot"],
            "rush_efficiency_spot": feature_values["rush_efficiency_spot"],
            "priced_mc_proj": priced_mc,
            "rebuilt_mc_proj": rebuilt_mc,
            "mc_rebuild_gap": rebuilt_mc - priced_mc,
            "promoted_mean": final_mean,
            "canonical_shadow_mean": float(np.mean(canonical)),
            "selected_shadow_mean": float(np.mean(chosen)),
            "canonical_mean_gap": float(np.mean(canonical) - final_mean),
            "selected_mean_gap": float(np.mean(chosen) - final_mean),
            "canonical_sd": float(np.std(canonical, ddof=1)),
            "selected_sd": float(np.std(chosen, ddof=1)),
            "canonical_p10": _q(canonical, 0.10),
            "canonical_p50": _q(canonical, 0.50),
            "canonical_p90": _q(canonical, 0.90),
            "selected_p10": _q(chosen, 0.10),
            "selected_p50": _q(chosen, 0.50),
            "selected_p90": _q(chosen, 0.90),
            "vegas_line_audit_only": line,
            "canonical_over_prob_audit_only": canonical_over,
            "selected_over_prob_audit_only": chosen_over,
            "over_prob_delta_audit_only": chosen_over - canonical_over if np.isfinite(chosen_over) and np.isfinite(canonical_over) else np.nan,
            "sportsbook_inputs_to_selector": 0,
        })

    out = pd.DataFrame(rows).sort_values(["event_id", "team", "player_clean_key"]).reset_index(drop=True)
    max_mc_gap = float(out["mc_rebuild_gap"].abs().max())
    max_canonical_mean_gap = float(out["canonical_mean_gap"].abs().max())
    max_selected_mean_gap = float(out["selected_mean_gap"].abs().max())
    selected_n = int(out["selector_c2_selected"].sum())

    gates = {
        "real_slate_mc_parity_le_1e8": bool(max_mc_gap <= 1e-8),
        "canonical_mean_anchor_le_1e8": bool(max_canonical_mean_gap <= 1e-8),
        "selected_mean_anchor_le_1e8": bool(max_selected_mean_gap <= 1e-8),
        "selector_selected_at_least_one_qb": bool(selected_n > 0),
        "selector_sportsbook_inputs_zero": bool(out["sportsbook_inputs_to_selector"].eq(0).all()),
        "all_qb_rows_covered": bool(len(out) == len(qb)),
    }
    disposition = "QB_DISTRIBUTION_SHADOW_WEEK1_PASS" if all(gates.values()) else "QB_DISTRIBUTION_SHADOW_WEEK1_FAIL"
    result = {
        "disposition": disposition,
        "qb_rows": int(len(out)),
        "selected_qb_rows": selected_n,
        "selector_version": str(art["version"]),
        "max_mc_rebuild_gap": max_mc_gap,
        "max_canonical_mean_gap": max_canonical_mean_gap,
        "max_selected_mean_gap": max_selected_mean_gap,
        "mean_canonical_sd": float(out["canonical_sd"].mean()),
        "mean_selected_sd": float(out["selected_sd"].mean()),
        "mean_abs_over_prob_delta_selected": float(out.loc[out.selector_c2_selected.eq(1), "over_prob_delta_audit_only"].abs().mean()) if selected_n else np.nan,
        "gates": gates,
        "production_pricing_modified": 0,
        "sportsbook_inputs_to_selector": 0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[qb_distribution_shadow] wrote {len(out)} QB rows -> {out_path}")
    if disposition.endswith("FAIL"):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
