#!/usr/bin/env python3
"""Diagnostic-only post-specialist cross-market consistency audit.

No target-game outcomes, sportsbook inputs, fitted parameters, new candidate
variants, or production mutations are permitted.

Authority is the immutable no-live-odds 2026 Week-3 Full Slate artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.modeling.rb_rush_rec_conservation_v2 import build_candidate_map as build_rb_rr_v2
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.run_pricing_with_full_roster_universe_v1 import _canonical_game
from scripts.simulation_v2 import lookup, simulate

DATA = Path("data")
FORBIDDEN = {
    "actual", "actual_rushes", "actual_rush_yards", "actual_rec_yards",
    "target_game_snaps", "line", "source_line", "over_odds", "under_odds",
    "odds", "book", "book_title", "sportsbook", "bookmaker", "market_prob",
    "edge_pct", "edge_abs", "vegas_line", "vegas_odds", "team_wp",
}
EXPECTED = {
    "rush_att": (0.3164919683016017, 0.6528957474344519, 0.0306122842639465, 1984),
    "rush_yards": (0.5569542426070742, 0.4430457573929258, 0.0, 3207),
    "rec_yards": (0.659889, 0.292427, 0.047684, 4413),
    "receptions": (0.554082, 0.444566, 0.001352, 4413),
}


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"{label} empty: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(v) -> float:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) else np.nan


def key_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def pos_family(v) -> str:
    p = "" if v is None or pd.isna(v) else str(v).upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    return p


def forbidden_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in FORBIDDEN or lc.startswith("sportsbook_"):
            out.append(str(c))
    return sorted(out)


def diag_value(df: pd.DataFrame, team: str, player: str, col: str) -> float:
    q = df.loc[
        df["team"].astype(str).str.upper().eq(team)
        & key_series(df["player_clean_key"]).eq(player)
    ]
    if len(q) != 1:
        raise RuntimeError(f"expected one diagnostic row {team}/{player} for {col}, got {len(q)}")
    return num(q.iloc[0].get(col))


def ensemble_one(market: str, mc: float, ml: float, state: float, weights: pd.DataFrame) -> pd.Series:
    x = apply_ensemble(pd.DataFrame([{
        "market": market, "mc_proj": mc, "ml_proj": ml, "state_proj": state,
    }]), weights=weights)
    if len(x) != 1:
        raise RuntimeError(f"ensemble failed for {market}")
    return x.iloc[0]


def quantiles_abs(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().abs()
    if x.empty:
        return {"n": 0, "median_abs": np.nan, "p90_abs": np.nan, "max_abs": np.nan}
    return {
        "n": int(len(x)),
        "median_abs": float(x.median()),
        "p90_abs": float(x.quantile(0.90)),
        "max_abs": float(x.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-run", type=int, required=True)
    ap.add_argument("--source-artifact", type=int, required=True)
    ap.add_argument("--source-digest", required=True)
    ap.add_argument("--source-sha", required=True)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus = read(DATA / "player_form_consensus.csv", "player consensus")
    ml = read(DATA / "model_ml_diagnostics.csv", "ML diagnostics")
    state = read(DATA / "model_state_diagnostics.csv", "State diagnostics")
    weights = read(DATA / "model_ensemble_weights.csv", "ensemble weights")

    for label, df in (("consensus", consensus), ("ml", ml), ("state", state), ("weights", weights)):
        bad = forbidden_columns(df)
        if bad:
            raise RuntimeError(f"{label} contains forbidden sportsbook/outcome fields: {bad}")

    if not pd.to_numeric(consensus["season"], errors="coerce").eq(2026).all():
        raise RuntimeError("source season is not exactly 2026")
    if not pd.to_numeric(consensus["week"], errors="coerce").eq(3).all():
        raise RuntimeError("source week is not exactly Week 3")

    # Freeze/check exact market-specific production weights from the authority.
    weight_audit = []
    for market, exp in EXPECTED.items():
        q = weights.loc[weights["market"].astype(str).str.lower().eq(market)]
        if len(q) != 1:
            raise RuntimeError(f"expected exactly one weight row for {market}, got {len(q)}")
        r = q.iloc[0]
        got = (
            num(r.get("mc_weight")), num(r.get("ml_weight")),
            num(r.get("state_weight")), int(num(r.get("calibration_rows"))),
        )
        max_gap = max(abs(got[i] - exp[i]) for i in range(3))
        if max_gap > 1e-12 or got[3] != exp[3]:
            raise RuntimeError(f"frozen production weight drift {market}: got={got} expected={exp}")
        weight_audit.append({
            "market": market, "mc_weight": got[0], "ml_weight": got[1],
            "state_weight": got[2], "calibration_rows": got[3],
            "max_expected_weight_gap": max_gap,
        })

    consensus["team"] = consensus["team"].astype(str).str.upper().str.strip()
    consensus["player_clean_key"] = key_series(consensus["player_clean_key"])
    consensus["position_family"] = consensus["position"].map(pos_family)
    if consensus.duplicated(["team", "player_clean_key"]).any():
        bad = consensus.loc[
            consensus.duplicated(["team", "player_clean_key"], keep=False),
            ["team", "player_clean_key", "player"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"duplicate football identities: {bad}")

    prepared = consensus.copy()
    prepared["event_id"] = [
        _canonical_game(t, o, s, w)
        for t, o, s, w in zip(
            prepared["team"], prepared["opponent"], prepared["season"], prepared["week"]
        )
    ]
    prepared["market"] = "football_universe"
    prepared = apply_bayesian_to_metrics(prepared)
    prepared = apply_rules_to_metrics(prepared)
    if not pd.to_numeric(prepared["rules_applied"], errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("canonical rules failed to apply to full football universe")
    if forbidden_columns(prepared):
        raise RuntimeError(f"prepared universe contains forbidden fields: {forbidden_columns(prepared)}")

    trace_rows: list[dict] = []
    sims = simulate(prepared, allocation_trace=trace_rows)
    if int(sims.iterations) != 25000:
        raise RuntimeError(f"unexpected simulation iterations={sims.iterations}")

    ml["team"] = ml["team"].astype(str).str.upper().str.strip()
    ml["player_clean_key"] = key_series(ml["player_clean_key"])
    state["team"] = state["team"].astype(str).str.upper().str.strip()
    state["player_clean_key"] = key_series(state["player_clean_key"])

    rb = prepared.loc[prepared["position_family"].isin(["RB", "FB"])].copy()
    rows = []
    expanded_v2 = []

    for _, p in rb.iterrows():
        team = str(p["team"]).upper().strip()
        pkey = str(p["player_clean_key"]).strip()
        event = str(p["event_id"])

        comps = {}
        for market in ("rush_att", "rush_yards", "rec_yards", "receptions"):
            draws = lookup(sims, p, market)
            if draws is None or len(draws) == 0:
                raise RuntimeError(f"missing MC distribution {team}/{pkey}/{market}")
            mc = float(np.mean(np.asarray(draws, dtype=float)))
            mlv = diag_value(ml, team, pkey, f"ml_{market}")
            stv = diag_value(state, team, pkey, f"state_{market}")
            ens = ensemble_one(market, mc, mlv, stv, weights)
            comps[market] = {
                "mc": mc, "ml": mlv, "state": stv,
                "ensemble": float(ens["ensemble_proj"]),
                "w_mc": float(ens["ensemble_weight_mc"]),
                "w_ml": float(ens["ensemble_weight_ml"]),
                "w_state": float(ens["ensemble_weight_state"]),
                "status": str(ens["ensemble_status"]),
            }
            if market in {"rush_yards", "rec_yards"}:
                expanded_v2.append({
                    "event_id": event, "player": p.get("player"),
                    "player_clean_key": pkey, "team": team,
                    "opponent": p.get("opponent"), "position_group": p.get("position_family"),
                    "position": p.get("position"), "season": p.get("season"),
                    "week": p.get("week"), "market": market,
                    "ml_proj": mlv, "state_proj": stv,
                })

        # Add the identity row required by the protected V2 adapter.
        expanded_v2.append({
            "event_id": event, "player": p.get("player"),
            "player_clean_key": pkey, "team": team,
            "opponent": p.get("opponent"), "position_group": p.get("position_family"),
            "position": p.get("position"), "season": p.get("season"),
            "week": p.get("week"), "market": "rush_rec_yards",
            "ml_proj": diag_value(ml, team, pkey, "ml_rush_rec_yards"),
            "state_proj": diag_value(state, team, pkey, "state_rush_rec_yards"),
        })

        a = comps["rush_att"]
        y = comps["rush_yards"]
        mc_att, mc_yards = a["mc"], y["mc"]
        ens_att, ens_yards = a["ensemble"], y["ensemble"]
        mc_ypc = mc_yards / mc_att if mc_att > 0 else np.nan
        ensemble_ypc = ens_yards / ens_att if ens_att > 0 else np.nan

        # Exact local derivative of final implied YPC if only the joint MC
        # opportunity path is scaled proportionally while MC efficiency and all
        # frozen ML/State components remain fixed.
        c_att = ens_att - a["w_mc"] * mc_att
        c_yards = ens_yards - y["w_mc"] * mc_yards
        derivative = np.nan
        if ens_att > 0:
            derivative = (
                y["w_mc"] * mc_yards * ens_att
                - ens_yards * a["w_mc"] * mc_att
            ) / (ens_att ** 2)

        rows.append({
            "season": 2026, "week": 3, "event_id": event, "team": team,
            "opponent": p.get("opponent"), "player": p.get("player"),
            "player_clean_key": pkey, "position_family": p.get("position_family"),
            "rules_rush_share": num(p.get("rules_rush_share")),
            "rules_ypc": num(p.get("rules_ypc")),
            "mc_rush_att": mc_att, "mc_rush_yards": mc_yards,
            "ensemble_rush_att": ens_att, "ensemble_rush_yards": ens_yards,
            "mc_implied_ypc": mc_ypc,
            "ensemble_implied_ypc": ensemble_ypc,
            "ensemble_minus_mc_implied_ypc": ensemble_ypc - mc_ypc if np.isfinite(ensemble_ypc) and np.isfinite(mc_ypc) else np.nan,
            "ensemble_minus_rules_ypc": ensemble_ypc - num(p.get("rules_ypc")) if np.isfinite(ensemble_ypc) else np.nan,
            "mc_minus_rules_ypc": mc_ypc - num(p.get("rules_ypc")) if np.isfinite(mc_ypc) else np.nan,
            "rush_att_w_mc": a["w_mc"], "rush_att_w_ml": a["w_ml"], "rush_att_w_state": a["w_state"],
            "rush_yards_w_mc": y["w_mc"], "rush_yards_w_ml": y["w_ml"], "rush_yards_w_state": y["w_state"],
            "rush_att_non_mc_contribution": c_att,
            "rush_yards_non_mc_contribution": c_yards,
            "d_ensemble_implied_ypc_d_mc_opportunity_scale": derivative,
            "mc_opportunity_scale_implied_ypc_derivative": 0.0,
            "ensemble_rush_att_status": a["status"],
            "ensemble_rush_yards_status": y["status"],
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("RB/FB cross-market audit produced zero rows")
    if detail.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("RB/FB audit identity is not unique")
    detail.to_csv(out_dir / "rb_rushing_cross_market_detail.csv", index=False)

    # Team finite-volume accounting from the exact simulator.
    trace = pd.DataFrame(trace_rows)
    if trace.empty:
        raise RuntimeError("allocation trace empty")
    trace = trace.drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
    team_rows = []
    for (event, team), tg in trace.groupby(["event_id", "team"], sort=True):
        team_prepared = prepared.loc[
            prepared["event_id"].astype(str).eq(str(event))
            & prepared["team"].astype(str).eq(str(team))
        ]
        mc_all = 0.0
        mc_rb = 0.0
        for _, p in team_prepared.iterrows():
            d = lookup(sims, p, "rush_att")
            if d is None:
                continue
            m = float(np.mean(np.asarray(d, float)))
            mc_all += m
            if pos_family(p.get("position")) in {"RB", "FB"}:
                mc_rb += m
        dg = detail.loc[
            detail["event_id"].astype(str).eq(str(event))
            & detail["team"].astype(str).eq(str(team))
        ]
        team_total = float(pd.to_numeric(tg["team_rush_total_mean"], errors="coerce").dropna().iloc[0])
        residual = float(pd.to_numeric(tg["residual_probability"], errors="coerce").dropna().iloc[0])
        team_rows.append({
            "event_id": str(event), "team": str(team),
            "team_mc_rush_total_mean": team_total,
            "sum_all_player_mc_rush_att": mc_all,
            "sum_rbfb_mc_rush_att": mc_rb,
            "sum_rbfb_generic_ensemble_rush_att": float(dg["ensemble_rush_att"].sum()) if len(dg) else 0.0,
            "simulator_residual_probability": residual,
            "mc_unallocated_rush_att_mean": team_total - mc_all,
            "rbfb_ensemble_minus_rbfb_mc_rush_att": float(dg["ensemble_rush_att"].sum() - mc_rb) if len(dg) else -mc_rb,
        })
    team = pd.DataFrame(team_rows)
    team.to_csv(out_dir / "team_finite_opportunity_summary.csv", index=False)

    # Reproduce the protected RB Rush+Receiving Conservation V2 identity.
    v2_metrics = pd.DataFrame(expanded_v2)
    v2_map, v2_payload = build_rb_rr_v2(v2_metrics, sims, weights)
    v2_rows = []
    for (event, pkey), meta in sorted(v2_map.items()):
        gap = float(meta["target_mean"] - (meta["rush_target_mean"] + meta["rec_target_mean"]))
        v2_rows.append({
            "event_id": event, "player_clean_key": pkey,
            "rush_target_mean": float(meta["rush_target_mean"]),
            "rec_target_mean": float(meta["rec_target_mean"]),
            "rush_rec_target_mean": float(meta["target_mean"]),
            "identity_gap": gap,
        })
    v2 = pd.DataFrame(v2_rows)
    if v2.empty:
        raise RuntimeError("protected RB Rush+Receiving V2 audit produced zero eligible rows")
    max_v2_gap = float(v2["identity_gap"].abs().max())
    if max_v2_gap > 1e-8:
        raise RuntimeError(f"protected RB Rush+Receiving V2 identity failed max_gap={max_v2_gap}")
    v2.to_csv(out_dir / "rb_rush_rec_v2_identity_audit.csv", index=False)

    summary = {
        "study": "POST_SPECIALIST_CROSS_MARKET_CONSISTENCY_V1",
        "disposition": "DIAGNOSTIC_COMPLETE_NO_REPAIR_AUTHORIZED",
        "source_full_slate_run": int(args.source_run),
        "source_full_slate_artifact": int(args.source_artifact),
        "source_full_slate_digest": str(args.source_digest),
        "production_source_sha": str(args.source_sha),
        "target_season": 2026,
        "target_week": 3,
        "simulation_iterations": int(sims.iterations),
        "rb_fb_rows": int(len(detail)),
        "teams": int(detail["team"].nunique()),
        "ensemble_minus_mc_implied_ypc": quantiles_abs(detail["ensemble_minus_mc_implied_ypc"]),
        "ensemble_minus_rules_ypc": quantiles_abs(detail["ensemble_minus_rules_ypc"]),
        "mc_minus_rules_ypc": quantiles_abs(detail["mc_minus_rules_ypc"]),
        "local_ensemble_implied_ypc_sensitivity_to_mc_opportunity_scale": quantiles_abs(
            detail["d_ensemble_implied_ypc_d_mc_opportunity_scale"]
        ),
        "mc_implied_ypc_sensitivity_to_proportional_opportunity_scale": 0.0,
        "rb_rush_rec_v2_rows": int(len(v2)),
        "rb_rush_rec_v2_max_identity_gap": max_v2_gap,
        "rb_rush_rec_v2_payload": v2_payload,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_read": 0,
        "new_candidate_variants_constructed": 0,
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
        "production_mutations": 0,
        "repair_authorized": False,
        "weight_audit": weight_audit,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Post-Specialist Cross-Market Consistency Audit V1",
        "",
        "Disposition: **DIAGNOSTIC_COMPLETE_NO_REPAIR_AUTHORIZED**",
        "",
        f"- RB/FB rows: **{len(detail)}**",
        f"- teams: **{detail['team'].nunique()}**",
        f"- simulator iterations: **{sims.iterations}**",
        "- sportsbook inputs: **0**",
        "- target-game outcomes read: **0**",
        "- new candidate variants constructed/scored: **0 / 0**",
        "- parameters fit: **0**",
        "",
        "## RB rushing cross-market",
        "",
        f"- abs ensemble-vs-MC implied YPC median / p90 / max: "
        f"**{summary['ensemble_minus_mc_implied_ypc']['median_abs']:.6f} / "
        f"{summary['ensemble_minus_mc_implied_ypc']['p90_abs']:.6f} / "
        f"{summary['ensemble_minus_mc_implied_ypc']['max_abs']:.6f}**",
        f"- abs local d(final implied YPC)/d(MC opportunity scale) median / p90 / max: "
        f"**{summary['local_ensemble_implied_ypc_sensitivity_to_mc_opportunity_scale']['median_abs']:.6f} / "
        f"{summary['local_ensemble_implied_ypc_sensitivity_to_mc_opportunity_scale']['p90_abs']:.6f} / "
        f"{summary['local_ensemble_implied_ypc_sensitivity_to_mc_opportunity_scale']['max_abs']:.6f}**",
        "- joint-MC implied-YPC derivative under proportional opportunity scaling: **0 by construction**",
        "",
        "## Protected RB Rush+Receiving Conservation V2",
        "",
        f"- eligible rows: **{len(v2)}**",
        f"- max identity gap: **{max_v2_gap:.3g}**",
        "",
        "This run is descriptive only. A separate repair hypothesis must be frozen before any scoring.",
    ]
    (out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
