#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 2130
EXPECTED_YPR_PLAYERS = 16
SIGNALS = {
    "PLAYER_EXP20_PER_TARGET_PRIOR8": ("player_exp20_per_target_prior8", "player"),
    "PLAYER_EXP40_PER_TARGET_PRIOR8": ("player_exp40_per_target_prior8", "player"),
    "PLAYER_YAC_PER_RECEPTION_PRIOR8": ("player_yac_per_reception_prior8", "player"),
    "PLAYER_AIR_PER_TARGET_PRIOR8": ("player_air_per_target_prior8", "player"),
    "DEF_EXP20_PER_ATT_ALLOWED_PRIOR8": ("def_exp20_per_att_allowed_prior8", "defense"),
    "DEF_EXP40_PER_ATT_ALLOWED_PRIOR8": ("def_exp40_per_att_allowed_prior8", "defense"),
    "DEF_YAC_PER_COMPLETION_ALLOWED_PRIOR8": ("def_yac_per_completion_allowed_prior8", "defense"),
    "DEF_AIR_PER_ATT_ALLOWED_PRIOR8": ("def_air_per_att_allowed_prior8", "defense"),
}


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(v):
    return pd.to_numeric(v, errors="coerce")


def spearman(a, b) -> float:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method="spearman"))


def gap(frame: pd.DataFrame, col: str, q25: float, q75: float, outcome: str) -> float:
    v = num(frame[col])
    y = num(frame[outcome])
    hi = v.notna() & v.ge(q75) & y.notna()
    lo = v.notna() & v.le(q25) & y.notna()
    if not hi.any() or not lo.any():
        return np.nan
    return float(y.loc[hi].mean() - y.loc[lo].mean())


def score_signal(frame: pd.DataFrame, name: str, col: str, side: str) -> dict:
    v = num(frame[col])
    valid = v.notna() & num(frame["ypr_component"]).notna()
    n_valid = int(valid.sum())
    coverage = float(valid.mean()) if len(frame) else 0.0
    vv = v.loc[valid]
    q25 = float(vv.quantile(.25)) if n_valid else np.nan
    q75 = float(vv.quantile(.75)) if n_valid else np.nan
    rho = spearman(v.loc[valid], frame.loc[valid, "ypr_component"])
    ypr_gap = gap(frame, col, q25, q75, "ypr_component") if n_valid else np.nan
    rec_gap = gap(frame, col, q25, q75, "yard_residual") if n_valid else np.nan

    hi = valid & v.ge(q75)
    tail = frame["under25"].astype(bool)
    overall_tail = float(tail.loc[valid].mean()) if n_valid else np.nan
    hi_tail = float(tail.loc[hi].mean()) if hi.any() else np.nan
    enrichment = float(hi_tail / overall_tail) if np.isfinite(hi_tail) and np.isfinite(overall_tail) and overall_tail > 0 else np.nan

    w2 = frame.loc[frame.week.between(2,18)]
    w13 = frame.loc[frame.week.between(13,18)]
    w2_gap = gap(w2, col, q25, q75, "ypr_component") if n_valid else np.nan
    w13_gap = gap(w13, col, q25, q75, "ypr_component") if n_valid else np.nan

    player_rhos = []
    for pk, g in frame.loc[valid].groupby("player_clean_key", sort=True):
        if len(g) < 8:
            continue
        r = spearman(g[col], g["ypr_component"])
        if np.isfinite(r):
            player_rhos.append((pk, r))
    positive_rate = float(np.mean([r > 0 for _, r in player_rhos])) if player_rhos else np.nan
    player_count = int(len(player_rhos))

    gates = {
        "valid_n_ge_120": bool(n_valid >= 120),
        "coverage_ge_0_75": bool(coverage >= 0.75),
        "spearman_ge_0_10": bool(np.isfinite(rho) and rho >= 0.10),
        "ypr_gap_ge_4": bool(np.isfinite(ypr_gap) and ypr_gap >= 4.0),
        "rec_residual_gap_ge_5": bool(np.isfinite(rec_gap) and rec_gap >= 5.0),
        "under25_enrichment_ge_1_20": bool(np.isfinite(enrichment) and enrichment >= 1.20),
        "w2_18_ypr_gap_positive": bool(np.isfinite(w2_gap) and w2_gap > 0),
        "w13_18_ypr_gap_positive": bool(np.isfinite(w13_gap) and w13_gap > 0),
        "player_count_ge_10": bool(player_count >= 10),
        "positive_player_spearman_rate_ge_0_55": bool(np.isfinite(positive_rate) and positive_rate >= 0.55),
    }
    return {
        "signal": name,
        "column": col,
        "side": side,
        "conditioned_rows": int(len(frame)),
        "valid_n": n_valid,
        "coverage": coverage,
        "q25": q25,
        "q75": q75,
        "spearman_ypr_component": rho,
        "q4_minus_q1_ypr_component_gap": ypr_gap,
        "q4_minus_q1_rec_yard_residual_gap": rec_gap,
        "under25_tail_overall": overall_tail,
        "under25_tail_q4": hi_tail,
        "under25_tail_enrichment": enrichment,
        "w2_18_ypr_component_gap": w2_gap,
        "w13_18_ypr_component_gap": w13_gap,
        "players_with_8plus_valid": player_count,
        "positive_player_spearman_rate": positive_rate,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-r5-root", type=Path, required=True)
    ap.add_argument("--nd6-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    r5 = pd.read_csv(one(a.wr_r5_root, "wr_r5_mechanism_casebook.csv"), low_memory=False)
    prof = pd.read_csv(one(a.wr_r5_root, "wr_r5_individual_mechanisms.csv"), low_memory=False)
    nd6 = pd.read_csv(one(a.nd6_root, "wr_nd6_casebook.csv"), low_memory=False)
    for d in (r5, prof, nd6):
        d.columns = [str(c).strip().lower() for c in d.columns]
    if len(r5) != EXPECTED_ROWS or len(nd6) != EXPECTED_ROWS:
        raise RuntimeError(f"casebook parity failed r5={len(r5)} nd6={len(nd6)}")
    if len(prof) != 133:
        raise RuntimeError(f"R5 qualifying profile drift expected=133 got={len(prof)}")
    ypr_players = prof.loc[prof.dominant_mechanism.astype(str).eq("YPR"), "player_key"].astype(str).tolist()
    if len(ypr_players) != EXPECTED_YPR_PLAYERS:
        raise RuntimeError(f"YPR player count drift expected={EXPECTED_YPR_PLAYERS} got={len(ypr_players)}")

    keys = ["season", "week", "team", "player_clean_key"]
    need_r5 = set(keys + ["ypr_component", "yard_residual"])
    need_nd6 = set(keys + [c for c,_ in SIGNALS.values()] + ["under25"])
    if need_r5-set(r5.columns):
        raise RuntimeError(f"R5 missing {sorted(need_r5-set(r5.columns))}")
    if need_nd6-set(nd6.columns):
        raise RuntimeError(f"ND6 missing {sorted(need_nd6-set(nd6.columns))}")

    left = r5[keys + ["ypr_component", "yard_residual"]].copy()
    right = nd6[keys + [c for c,_ in SIGNALS.values()] + ["under25"]].copy()
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise RuntimeError("duplicate WR keys in source evidence")
    x = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"R5/ND6 merge drift expected={EXPECTED_ROWS} got={len(x)}")
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    x["week"] = num(x["week"])
    y = x.loc[x.player_clean_key.isin(set(ypr_players))].copy()
    if y.empty:
        raise RuntimeError("empty YPR-conditioned population")

    rows = [score_signal(y, name, col, side) for name,(col,side) in SIGNALS.items()]
    passed = [r for r in rows if r["passed"]]
    player_side_passes = [r for r in passed if r["side"] == "player"]
    family_pass = bool(len(passed) >= 2 and len(player_side_passes) >= 1)
    disposition = "WR_YPR_MECHANISM_CONDITIONED_EFFICIENCY_DISCOVERY_PASS" if family_pass else "NO_ACTIONABLE_WR_YPR_CONDITIONED_EFFICIENCY_SIGNAL"

    result = {
        "migration": "WR_R7_YPR_MECHANISM_CONDITIONED_EFFICIENCY",
        "full_rows": int(len(x)),
        "qualifying_profiles": int(len(prof)),
        "ypr_dominant_players": int(len(ypr_players)),
        "ypr_conditioned_rows": int(len(y)),
        "passing_signals": [r["signal"] for r in passed],
        "passing_signal_count": int(len(passed)),
        "player_side_passing_signal_count": int(len(player_side_passes)),
        "family_gate": {"at_least_two_signals_pass": bool(len(passed)>=2), "at_least_one_player_side_pass": bool(len(player_side_passes)>=1)},
        "signals": {r["signal"]: {k:v for k,v in r.items() if k!="signal"} for r in rows},
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    y.to_csv(a.out_dir / "wr_r7_ypr_conditioned_casebook.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!="gates"} for r in rows]).to_csv(a.out_dir / "wr_r7_signal_summary.csv", index=False)
    (a.out_dir / "wr_r7_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(pd.DataFrame([{k:v for k,v in r.items() if k!="gates"} for r in rows]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
