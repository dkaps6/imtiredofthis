#!/usr/bin/env python3
"""Score frozen Rush Post-Ensemble Reconciliation V1.

Consumes only preserved historical baseline full-stack rows from run
36049144898. The old Rush Pool candidate columns are explicitly excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "RUSH_POST_ENSEMBLE_RECONCILIATION_V1"
SEASONS = (2024, 2025)
TEAM_KEYS = ["season", "week", "event_id", "team"]
PLAYER_KEYS = TEAM_KEYS + ["player_clean_key"]
SOURCE_COLS = [
    "event_id", "team", "player", "player_clean_key", "position", "market",
    "season", "week", "actual", "position_family",
    "baseline_mc_proj", "baseline_proj",
]


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing source: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = set(SOURCE_COLS) - set(x.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    # Deliberately discard every old candidate column before any computation.
    x = x[SOURCE_COLS].copy()
    for c in ("season", "week", "actual", "baseline_mc_proj", "baseline_proj"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if x[["season","week","actual","baseline_mc_proj","baseline_proj"]].isna().any().any():
        raise RuntimeError(f"non-finite required values in {path}")
    return x


def pivot_source(x: pd.DataFrame) -> pd.DataFrame:
    if not x["season"].isin(SEASONS).all():
        raise RuntimeError("unexpected season in source")
    if not x["week"].between(2, 18).all():
        raise RuntimeError("Week-1/out-of-scope rows present")
    if x.duplicated(PLAYER_KEYS + ["market"]).any():
        raise RuntimeError("duplicate player-game-market source identity")

    idx = PLAYER_KEYS + ["player", "position", "position_family"]
    p = x.pivot(index=idx, columns="market", values=["actual","baseline_mc_proj","baseline_proj"]).reset_index()
    p.columns = [
        "_".join(str(v) for v in c if str(v) != "") if isinstance(c, tuple) else str(c)
        for c in p.columns
    ]
    req = {
        "actual_rush_att", "actual_rush_yards", "actual_rush_rec_yards",
        "baseline_mc_proj_rush_att",
        "baseline_proj_rush_att", "baseline_proj_rush_yards",
        "baseline_proj_rush_rec_yards",
    }
    missing = req - set(p.columns)
    if missing:
        raise RuntimeError(f"pivot missing {sorted(missing)}")
    if p[list(req)].isna().any().any():
        raise RuntimeError("paired rushing source contains missing required values")
    if p.duplicated(PLAYER_KEYS).any():
        raise RuntimeError("paired player identity not unique")
    return p


def apply_candidate(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sums = (
        p.groupby(TEAM_KEYS, as_index=False)
        .agg(
            baseline_mc_player_carry_mass=("baseline_mc_proj_rush_att","sum"),
            baseline_final_player_carry_mass=("baseline_proj_rush_att","sum"),
        )
    )
    bad = sums.loc[
        sums["baseline_final_player_carry_mass"].abs().le(1e-12)
        & sums["baseline_mc_player_carry_mass"].abs().gt(1e-12)
    ]
    if len(bad):
        raise RuntimeError(f"zero final carry mass with positive MC mass: {bad.head().to_dict('records')}")

    sums["reconciliation_factor"] = np.where(
        sums["baseline_final_player_carry_mass"].abs().gt(1e-12),
        sums["baseline_mc_player_carry_mass"] / sums["baseline_final_player_carry_mass"],
        1.0,
    )
    out = p.merge(sums, on=TEAM_KEYS, how="left", validate="many_to_one")
    if not np.isfinite(out["reconciliation_factor"]).all() or (out["reconciliation_factor"] < 0).any():
        raise RuntimeError("invalid reconciliation factor")

    f = out["reconciliation_factor"]
    out["candidate_rush_att"] = out["baseline_proj_rush_att"] * f
    out["candidate_rush_yards"] = out["baseline_proj_rush_yards"] * f
    out["baseline_rec_component"] = (
        out["baseline_proj_rush_rec_yards"] - out["baseline_proj_rush_yards"]
    )
    out["candidate_rush_rec_yards"] = (
        out["candidate_rush_yards"] + out["baseline_rec_component"]
    )

    # Exact invariants.
    chk = (
        out.groupby(TEAM_KEYS, as_index=False)
        .agg(candidate_mass=("candidate_rush_att","sum"),
             mc_mass=("baseline_mc_proj_rush_att","sum"))
    )
    chk["mass_gap"] = chk["candidate_mass"] - chk["mc_mass"]
    max_mass_gap = float(chk["mass_gap"].abs().max()) if len(chk) else 0.0
    if max_mass_gap > 1e-10:
        raise RuntimeError(f"team player-carry conservation failed max_gap={max_mass_gap}")

    valid = (
        out["baseline_proj_rush_att"].gt(1e-10)
        & out["candidate_rush_att"].gt(1e-10)
    )
    base_ypc = out.loc[valid,"baseline_proj_rush_yards"] / out.loc[valid,"baseline_proj_rush_att"]
    cand_ypc = out.loc[valid,"candidate_rush_yards"] / out.loc[valid,"candidate_rush_att"]
    max_ypc_gap = float((base_ypc - cand_ypc).abs().max()) if len(base_ypc) else 0.0
    if max_ypc_gap > 1e-10:
        raise RuntimeError(f"player implied YPC changed max_gap={max_ypc_gap}")

    rec_before = out["baseline_rec_component"]
    rec_after = out["candidate_rush_rec_yards"] - out["candidate_rush_yards"]
    max_rec_gap = float((rec_before - rec_after).abs().max()) if len(out) else 0.0
    if max_rec_gap > 1e-10:
        raise RuntimeError(f"RB receiving component drift max_gap={max_rec_gap}")

    audit = sums.copy()
    audit["candidate_player_carry_mass"] = audit["baseline_mc_player_carry_mass"]
    return out, audit.assign(
        max_team_carry_mass_gap=max_mass_gap,
        max_player_implied_ypc_gap=max_ypc_gap,
        max_receiving_component_gap=max_rec_gap,
    )


def metric(part: pd.DataFrame, base: str, cand: str, actual: str, catastrophic: float) -> dict:
    if part.empty:
        return {"n": 0}
    b = pd.to_numeric(part[base], errors="coerce")
    c = pd.to_numeric(part[cand], errors="coerce")
    y = pd.to_numeric(part[actual], errors="coerce")
    usable = b.notna() & c.notna() & y.notna()
    b,c,y = b[usable],c[usable],y[usable]
    be = b-y
    ce = c-y
    bae = be.abs()
    cae = ce.abs()
    changed = (c-b).abs().gt(1e-12)
    closer = float((cae[changed] < bae[changed]).mean()) if changed.any() else np.nan
    return {
        "n": int(len(b)),
        "baseline_mae": float(bae.mean()),
        "candidate_mae": float(cae.mean()),
        "mae_delta_candidate_minus_baseline": float(cae.mean()-bae.mean()),
        "baseline_rmse": float(np.sqrt(np.mean(be**2))),
        "candidate_rmse": float(np.sqrt(np.mean(ce**2))),
        "baseline_bias": float(be.mean()),
        "candidate_bias": float(ce.mean()),
        "baseline_p90_abs_error": float(bae.quantile(0.90)),
        "candidate_p90_abs_error": float(cae.quantile(0.90)),
        "baseline_catastrophic_misses": int(bae.ge(catastrophic).sum()),
        "candidate_catastrophic_misses": int(cae.ge(catastrophic).sum()),
        "changed_rows": int(changed.sum()),
        "changed_row_candidate_closer_rate": closer,
    }


def family_frame(x: pd.DataFrame, family: str) -> pd.DataFrame:
    if family == "ALL":
        return x
    if family == "RB_FAMILY":
        return x.loc[x["position_family"].eq("RB_FAMILY")]
    if family == "QB":
        return x.loc[x["position_family"].eq("QB")]
    if family == "OTHER":
        return x.loc[x["position_family"].eq("OTHER")]
    raise ValueError(family)


def score_season(x: pd.DataFrame) -> dict:
    out = {}
    for fam in ("ALL","RB_FAMILY","QB","OTHER"):
        q = family_frame(x, fam)
        out[fam] = {
            "rush_att": metric(q, "baseline_proj_rush_att", "candidate_rush_att", "actual_rush_att", 10.0),
            "rush_yards": metric(q, "baseline_proj_rush_yards", "candidate_rush_yards", "actual_rush_yards", 30.0),
        }
    rb = family_frame(x, "RB_FAMILY")
    out["RB_FAMILY"]["rush_rec_yards"] = metric(
        rb, "baseline_proj_rush_rec_yards", "candidate_rush_rec_yards",
        "actual_rush_rec_yards", 30.0,
    )
    return out


def nonworse(a: float, b: float, tol: float=1e-12) -> bool:
    return b <= a + tol


def gates(scores: dict) -> dict:
    g = {}
    for season in ("2024","2025"):
        s = scores[season]
        g[f"{season}_all_att_mae_improves"] = s["ALL"]["rush_att"]["candidate_mae"] < s["ALL"]["rush_att"]["baseline_mae"]
        g[f"{season}_all_yards_mae_improves"] = s["ALL"]["rush_yards"]["candidate_mae"] < s["ALL"]["rush_yards"]["baseline_mae"]
        g[f"{season}_rb_att_mae_improves"] = s["RB_FAMILY"]["rush_att"]["candidate_mae"] < s["RB_FAMILY"]["rush_att"]["baseline_mae"]
        g[f"{season}_rb_yards_mae_improves"] = s["RB_FAMILY"]["rush_yards"]["candidate_mae"] < s["RB_FAMILY"]["rush_yards"]["baseline_mae"]
        for fam,label in (("QB","qb"),("OTHER","other")):
            for m,mlabel in (("rush_att","att"),("rush_yards","yards")):
                q=s[fam][m]
                g[f"{season}_{label}_{mlabel}_mae_nonworse"] = nonworse(q["baseline_mae"],q["candidate_mae"])
        for fam,label in (("ALL","all"),("RB_FAMILY","rb")):
            for m,mlabel in (("rush_att","att"),("rush_yards","yards")):
                q=s[fam][m]
                g[f"{season}_{label}_{mlabel}_p90_nonworse"] = nonworse(q["baseline_p90_abs_error"],q["candidate_p90_abs_error"])
        combo=s["RB_FAMILY"]["rush_rec_yards"]
        g[f"{season}_rb_combo_mae_nonworse"] = nonworse(combo["baseline_mae"],combo["candidate_mae"])
        g[f"{season}_rb_combo_p90_nonworse"] = nonworse(combo["baseline_p90_abs_error"],combo["candidate_p90_abs_error"])
        g[f"{season}_att_closer_gt_half"] = s["ALL"]["rush_att"]["changed_row_candidate_closer_rate"] > 0.5
        g[f"{season}_yards_closer_gt_half"] = s["ALL"]["rush_yards"]["changed_row_candidate_closer_rate"] > 0.5
    return g


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifact-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    scores={}
    all_detail=[]
    all_audit=[]
    for season in SEASONS:
        raw=read(args.artifact_dir / f"detail_{season}.csv")
        p=pivot_source(raw)
        cand,audit=apply_candidate(p)
        cand.to_csv(args.out_dir / f"detail_{season}.csv",index=False)
        audit.to_csv(args.out_dir / f"team_audit_{season}.csv",index=False)
        scores[str(season)]=score_season(cand)
        all_detail.append(cand)
        all_audit.append(audit.assign(season=season))

    gs=gates(scores)
    integrity={
        "sportsbook_inputs_used":0,
        "parameters_fit":0,
        "candidate_variants_scored":1,
        "old_rush_pool_candidate_columns_used":0,
        "week1_candidate_rows":0,
        "production_changed":False,
        "max_team_carry_mass_gap":float(max(a["max_team_carry_mass_gap"].max() for a in all_audit)),
        "max_player_implied_ypc_gap":float(max(a["max_player_implied_ypc_gap"].max() for a in all_audit)),
        "max_receiving_component_gap":float(max(a["max_receiving_component_gap"].max() for a in all_audit)),
    }
    integrity_pass=(
        integrity["max_team_carry_mass_gap"] <= 1e-10
        and integrity["max_player_implied_ypc_gap"] <= 1e-10
        and integrity["max_receiving_component_gap"] <= 1e-10
    )
    gs["integrity_gates_pass"]=bool(integrity_pass)
    qualified=all(bool(v) for v in gs.values())
    disposition=(
        f"{VERSION}_QUALIFIED" if qualified
        else f"{VERSION}_FAILED_CLOSED"
    )
    payload={
        "version":VERSION,
        "disposition":disposition,
        "qualified":bool(qualified),
        "historical_authority":{
            "run":36049144898,
            "job":107800031207,
            "artifact":10830277740,
            "digest":"sha256:b6a0715d38984fa91b8815561b4e72679684affcc142ee773a8b7428342412a5",
            "baseline_source_main":"b5b1816e7d11cadca412faf841fbfe5336798c2e",
            "current_production_source":"f7d2011b73950488ea209124ba895b92c401b2b1",
        },
        "integrity":integrity,
        "scores":scores,
        "gates":gs,
    }
    (args.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    lines=[
        "# Rush Post-Ensemble Reconciliation V1 — Result",
        "",
        f"Disposition: **{disposition}**",
        "",
        "## Integrity",
        f"- max team player-carry mass gap: **{integrity['max_team_carry_mass_gap']:.3g}**",
        f"- max player implied-YPC change: **{integrity['max_player_implied_ypc_gap']:.3g}**",
        f"- max preserved receiving-component gap: **{integrity['max_receiving_component_gap']:.3g}**",
        "- sportsbook inputs: **0**",
        "- parameters fit: **0**",
        "- candidate variants scored: **1**",
        "",
    ]
    for season in ("2024","2025"):
        lines += [f"## {season}", ""]
        s=scores[season]
        for fam in ("ALL","RB_FAMILY","QB","OTHER"):
            for market in ("rush_att","rush_yards"):
                q=s[fam][market]
                lines.append(
                    f"- {fam} {market} MAE: **{q['baseline_mae']:.6f} -> {q['candidate_mae']:.6f}**; "
                    f"p90 **{q['baseline_p90_abs_error']:.6f} -> {q['candidate_p90_abs_error']:.6f}**"
                )
        q=s["RB_FAMILY"]["rush_rec_yards"]
        lines.append(
            f"- RB_FAMILY rush_rec_yards MAE: **{q['baseline_mae']:.6f} -> {q['candidate_mae']:.6f}**; "
            f"p90 **{q['baseline_p90_abs_error']:.6f} -> {q['candidate_p90_abs_error']:.6f}**"
        )
        lines.append("")
    failed=[k for k,v in gs.items() if not v]
    lines += ["## Gates","",f"- passed: **{sum(bool(v) for v in gs.values())}/{len(gs)}**",f"- failed: **{len(failed)}**"]
    for k in failed:
        lines.append(f"  - {k}")
    (args.out_dir/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    print((args.out_dir/"RESULT.md").read_text())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
