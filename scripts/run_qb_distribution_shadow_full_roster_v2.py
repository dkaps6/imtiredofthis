#!/usr/bin/env python3
"""Real-slate QB-C2 shadow from the exact full-roster football simulation universe.

This is deployment evidence only. It never changes pricing outputs. It rebuilds
the canonical QB arrays from the sportsbook-independent 469-player universe,
then applies the frozen QB-C2 distribution candidate anchored to the promoted
M89/M90 mean. Sportsbook lines are read only after both football distributions
exist for audit-only probability deltas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.qb_distribution_state_v1 import load_artifact, select_c2
from scripts.simulation_c2_qb_candidate import apply_c2, lookup, simulate_with_states
from scripts.utils.player_identity_v3 import player_name_key


def _f(v, default=np.nan) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _key(v) -> str:
    try:
        return str(player_name_key(v, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(np.asarray(a, float), p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--priced", required=True)
    ap.add_argument("--state-context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    universe = pd.read_csv(args.universe, low_memory=False)
    priced = pd.read_csv(args.priced, low_memory=False)
    ctx = pd.read_csv(args.state_context, low_memory=False)
    for frame in (universe, priced, ctx):
        frame.columns = [str(c).lower() for c in frame.columns]

    if universe.empty or universe["team"].nunique() != 32:
        raise RuntimeError("full-roster shadow requires 32-team football universe")
    forbidden = {"line","source_line","over_odds","under_odds","book","book_title","vegas_line","vegas_odds","market_prob","team_wp"}
    leaked = sorted(forbidden & set(universe.columns))
    if leaked:
        raise RuntimeError(f"sportsbook fields leaked into football universe: {leaked}")

    universe["_canon_player"] = universe["player"].map(_key)
    if universe["_canon_player"].eq("").any():
        raise RuntimeError("blank canonical player identity in football universe")
    if universe.duplicated(["event_id","team","_canon_player"]).any():
        raise RuntimeError("duplicate canonical player identity in football universe")

    qb = priced.loc[priced["market"].astype(str).str.lower().eq("pass_yards")].copy()
    qb = qb.sort_values(["event_id","team","player_clean_key","side"]).drop_duplicates(
        ["event_id","team","player_clean_key"], keep="first"
    )
    if len(qb) != 32:
        raise RuntimeError(f"expected 32 priced primary QBs, found {len(qb)}")
    if not pd.to_numeric(qb.get("qb_synthesis_applied",0), errors="coerce").eq(1).all():
        raise RuntimeError("promoted M89/M90 mean missing on QB row")
    qb["_canon_player"] = qb["player"].map(_key)

    # Map provider event/player identity to the exact sportsbook-independent
    # canonical universe identity that production actually simulated.
    ident = universe[["event_id","team","opponent","season","week","player","_canon_player"]].copy()
    qmap = qb.merge(
        ident,
        on=["team","_canon_player"],
        how="left",
        suffixes=("_priced","_football"),
        validate="one_to_one",
    )
    if qmap["event_id_football"].isna().any():
        sample = qmap.loc[qmap["event_id_football"].isna(), ["player_priced","team"]].to_dict("records")
        raise RuntimeError(f"priced QB missing from football universe: {sample}")

    art = load_artifact()
    if int(art.get("sportsbook_inputs_used",1)) != 0:
        raise RuntimeError("QB selector artifact sportsbook leakage flag")
    required_ctx = ["team","pass_opportunity_spot","pass_efficiency_spot","rush_opportunity_spot","rush_efficiency_spot","sportsbook_inputs_used"]
    missing = [c for c in required_ctx if c not in ctx.columns]
    if missing:
        raise RuntimeError(f"state context missing columns: {missing}")
    if not pd.to_numeric(ctx["sportsbook_inputs_used"], errors="coerce").eq(0).all():
        raise RuntimeError("state context sportsbook leakage flag")
    if ctx.duplicated("team").any():
        raise RuntimeError("duplicate state-context team")
    ctx_by_team = ctx.set_index("team", drop=False)

    # IMPORTANT: full-roster production simulation is keyed from this universe,
    # not the offer-only pricing rows. The state-capturing simulator is a
    # byte-parity shadow of canonical simulation and does not consume sportsbook.
    base = simulate_with_states(universe.drop(columns=["_canon_player"]))

    anchors: dict[tuple[str,str],float] = {}
    for _, r in qmap.iterrows():
        game = str(r["event_id_football"])
        team = str(r["team"])
        mean = _f(r.get("model_proj"))
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid promoted QB mean {(game,team)}={mean}")
        anchors[(game,team)] = mean

    c2 = apply_c2(base, universe.drop(columns=["_canon_player"]), anchor_map=anchors, seed=5601)

    rows = []
    for _, r in qmap.iterrows():
        team = str(r["team"])
        cr = ctx_by_team.loc[team]
        if isinstance(cr, pd.DataFrame):
            raise RuntimeError(f"duplicate context rows team={team}")

        shadow_row = pd.Series({
            "event_id": r["event_id_football"],
            "player": r["player_football"],
            "player_clean_key": r["_canon_player"],
            "team": team,
            "opponent": r["opponent_football"],
        })
        raw = lookup(base, shadow_row, "pass_yards")
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"canonical full-roster QB array missing {team} {r['player_priced']}")
        raw = np.asarray(raw, float)
        conv = _f(r.get("qb_attempt_conversion"))
        share = float(np.clip(_f(r.get("qb_pass_att_share"),1.0),0.0,1.0))
        if not np.isfinite(conv) or not 0.50 <= conv <= 1.0:
            raise RuntimeError(f"invalid attempt conversion {team}={conv}")
        component = raw * conv * share
        rebuilt_mc = float(component.mean())
        priced_mc = _f(r.get("mc_proj"))
        final_mean = _f(r.get("model_proj"))
        if rebuilt_mc <= 0 or not np.isfinite(priced_mc) or not np.isfinite(final_mean):
            raise RuntimeError(f"invalid QB shadow means {team}")
        canonical = component * (final_mean / rebuilt_mc)

        features = {
            "pass_opportunity_spot": _f(cr.get("pass_opportunity_spot")),
            "pass_efficiency_spot": _f(cr.get("pass_efficiency_spot")),
            "rush_opportunity_spot": _f(cr.get("rush_opportunity_spot")),
            "rush_efficiency_spot": _f(cr.get("rush_efficiency_spot")),
            "pred_qb_attempts": _f(r.get("qb_pred_attempts")),
            "week": _f(r.get("week_priced"), _f(r.get("week_football"))),
        }
        selected, delta, selector_version = select_c2(features, art)
        c2_arr = lookup(c2, shadow_row, "pass_yards")
        if selected and (c2_arr is None or len(c2_arr) == 0):
            raise RuntimeError(f"selected C2 array missing {team}")
        chosen = np.asarray(c2_arr,float) if selected else canonical
        line = _f(r.get("vegas_line"))
        p0 = float(np.mean(canonical > line)) if np.isfinite(line) else np.nan
        p1 = float(np.mean(chosen > line)) if np.isfinite(line) else np.nan
        rows.append({
            "season": int(_f(r.get("season_priced"),_f(r.get("season_football"),0))),
            "week": int(_f(r.get("week_priced"),_f(r.get("week_football"),0))),
            "provider_event_id": r.get("event_id_priced"),
            "canonical_event_id": r.get("event_id_football"),
            "team": team,
            "opponent": r.get("opponent_priced"),
            "player": r.get("player_priced"),
            "selector_version": selector_version,
            "selector_delta_pass_attempts": float(delta),
            "selector_c2_selected": int(selected),
            "priced_mc_proj": priced_mc,
            "rebuilt_mc_proj": rebuilt_mc,
            "mc_rebuild_gap": rebuilt_mc-priced_mc,
            "promoted_mean": final_mean,
            "canonical_shadow_mean": float(canonical.mean()),
            "selected_shadow_mean": float(chosen.mean()),
            "canonical_mean_gap": float(canonical.mean()-final_mean),
            "selected_mean_gap": float(chosen.mean()-final_mean),
            "canonical_sd": float(np.std(canonical,ddof=1)),
            "selected_sd": float(np.std(chosen,ddof=1)),
            "canonical_p10": _q(canonical,.10),
            "canonical_p50": _q(canonical,.50),
            "canonical_p90": _q(canonical,.90),
            "selected_p10": _q(chosen,.10),
            "selected_p50": _q(chosen,.50),
            "selected_p90": _q(chosen,.90),
            "vegas_line_audit_only": line,
            "canonical_over_prob_audit_only": p0,
            "selected_over_prob_audit_only": p1,
            "over_prob_delta_audit_only": p1-p0 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
            "sportsbook_inputs_to_selector": 0,
        })

    out = pd.DataFrame(rows).sort_values(["canonical_event_id","team"]).reset_index(drop=True)
    selected_n = int(out["selector_c2_selected"].sum())
    max_mc = float(out["mc_rebuild_gap"].abs().max())
    max_canon = float(out["canonical_mean_gap"].abs().max())
    max_sel = float(out["selected_mean_gap"].abs().max())
    gates = {
        "real_slate_full_roster_mc_parity_le_1e8": bool(max_mc <= 1e-8),
        "canonical_mean_anchor_le_1e8": bool(max_canon <= 1e-8),
        "selected_mean_anchor_le_1e8": bool(max_sel <= 1e-8),
        "selector_selected_at_least_one_qb": bool(selected_n > 0),
        "selector_sportsbook_inputs_zero": bool(out["sportsbook_inputs_to_selector"].eq(0).all()),
        "all_32_qb_rows_covered": bool(len(out) == 32),
    }
    disposition = "QB_DISTRIBUTION_FULL_ROSTER_SHADOW_PASS" if all(gates.values()) else "QB_DISTRIBUTION_FULL_ROSTER_SHADOW_FAIL"
    result = {
        "disposition": disposition,
        "qb_rows": int(len(out)),
        "selected_qb_rows": selected_n,
        "selector_version": str(art["version"]),
        "max_mc_rebuild_gap": max_mc,
        "max_canonical_mean_gap": max_canon,
        "max_selected_mean_gap": max_sel,
        "mean_canonical_sd": float(out["canonical_sd"].mean()),
        "mean_selected_sd": float(out["selected_sd"].mean()),
        "mean_abs_over_prob_delta_selected": float(out.loc[out.selector_c2_selected.eq(1),"over_prob_delta_audit_only"].abs().mean()) if selected_n else np.nan,
        "football_universe_rows": int(len(universe)),
        "production_pricing_modified": 0,
        "sportsbook_inputs_to_selector": 0,
        "gates": gates,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out,index=False)
    Path(args.result).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if disposition.endswith("FAIL"):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
