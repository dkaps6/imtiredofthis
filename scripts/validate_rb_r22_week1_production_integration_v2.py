#!/usr/bin/env python3
"""R22 mechanical differential gate.

Compares V4 against a V3 control generated earlier in the SAME checkout from the
same immutable inputs and deterministic MC seed. This isolates R22 from unrelated
historical code drift between old replay artifacts and the current certified V3 stack.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/v3_same_checkout.csv")
CANDIDATE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/props_priced_clean.csv")
TRACE = Path("data/rb_receiving_tail_production_trace.csv")
ADAPTER_AUDIT = Path("data/rb_receiving_tail_production_audit.json")
PRICING_AUDIT = Path("data/rb_receiving_tail_pricing_lineage_audit.json")
TE_AUDIT = Path("data/te_r5p_full_slate_entitlement_audit.json")
WR_AUDIT = Path("data/wr_r15_full_slate_entitlement_audit.json")
QB_AUDIT = Path("data/qb_c2_production_integration_audit.json")
OUT_JSON = Path("data/rb_r22_week1_production_integration_audit_v2.json")
OUT_CSV = Path("data/rb_r22_week1_pricing_delta_audit_v2.csv")

KEYS = ["event_id", "player_clean_key", "market", "book", "side", "vegas_line", "vegas_odds"]


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing R22 validation input: {path}")
    return pd.read_csv(path, low_memory=False)


def _eq(a: pd.Series, b: pd.Series, tol: float = 1e-10) -> pd.Series:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    return (x.isna() & y.isna()) | (x.notna() & y.notna() & ((x-y).abs() <= tol))


def main() -> int:
    control = _read(CONTROL)
    candidate = _read(CANDIDATE)
    trace = _read(TRACE)
    adapter = json.loads(ADAPTER_AUDIT.read_text(encoding="utf-8"))
    pricing = json.loads(PRICING_AUDIT.read_text(encoding="utf-8"))
    te = json.loads(TE_AUDIT.read_text(encoding="utf-8"))
    wr = json.loads(WR_AUDIT.read_text(encoding="utf-8"))
    qb = json.loads(QB_AUDIT.read_text(encoding="utf-8"))

    for df, label in [(control,"control"),(candidate,"candidate")]:
        missing=[c for c in KEYS if c not in df.columns]
        if missing: raise RuntimeError(f"{label} missing comparison keys: {missing}")
        if df.duplicated(KEYS).any(): raise RuntimeError(f"{label} has duplicate pricing keys")

    merged = control.merge(candidate, on=KEYS, how="outer", suffixes=("_control","_candidate"), indicator=True, validate="one_to_one")
    row_universe_exact = bool(len(control)==len(candidate) and merged._merge.eq("both").all())

    adapted_keys = {(str(r.team).upper(), str(r.player_clean_key)) for r in trace.itertuples(index=False) if bool(r.rb_receiving_tail_applied)}
    team_col = "team_candidate" if "team_candidate" in merged.columns else "team"
    if team_col not in merged.columns:
        raise RuntimeError("R22 differential comparison lacks team identity")
    merged["adapted_rb"] = [(str(t).upper(), str(p)) in adapted_keys for t,p in zip(merged[team_col], merged.player_clean_key)]
    merged["allowed_distribution_change"] = merged.adapted_rb & merged.market.astype(str).isin(["rec_yards","rush_rec_yards"])

    mean_checks={}
    for col in ["model_proj","mc_proj","ensemble_proj","ml_proj","state_proj"]:
        a,b=f"{col}_control",f"{col}_candidate"
        if a in merged.columns and b in merged.columns:
            eq=_eq(merged[a],merged[b],1e-8)
            diff=(pd.to_numeric(merged[b],errors="coerce")-pd.to_numeric(merged[a],errors="coerce")).abs()
            mean_checks[col]={"all_equal":bool(eq.all()),"max_abs_delta":float(diff.max(skipna=True) if diff.notna().any() else 0.0)}
            merged[f"{col}_equal"]=eq

    unexpected = pd.Series(False,index=merged.index)
    allowed_changed = pd.Series(False,index=merged.index)
    for col in ["fair_prob","fair_odds","edge_pct","edge_abs"]:
        a,b=f"{col}_control",f"{col}_candidate"
        if a in merged.columns and b in merged.columns:
            eq=_eq(merged[a],merged[b],1e-10)
            merged[f"{col}_equal"]=eq
            unexpected |= (~eq) & (~merged.allowed_distribution_change)
            allowed_changed |= (~eq) & merged.allowed_distribution_change

    recmask = merged.adapted_rb & merged.market.astype(str).eq("receptions")
    receptions_exact=True
    for col in ["model_proj","mc_proj","fair_prob","fair_odds","edge_pct","edge_abs"]:
        a,b=f"{col}_control",f"{col}_candidate"
        if a in merged.columns and b in merged.columns and recmask.any():
            receptions_exact &= bool(_eq(merged.loc[recmask,a],merged.loc[recmask,b],1e-10).all())

    non_allowed = ~merged.allowed_distribution_change
    non_allowed_exact=True
    for col in ["model_proj","mc_proj","fair_prob","fair_odds","edge_pct","edge_abs"]:
        a,b=f"{col}_control",f"{col}_candidate"
        if a in merged.columns and b in merged.columns:
            non_allowed_exact &= bool(_eq(merged.loc[non_allowed,a],merged.loc[non_allowed,b],1e-10).all())

    te_valid=bool(te.get("disposition")=="TE_R5P_FULL_SLATE_ENTITLEMENT_READY" and te.get("team_te_pool_preserved") is True and te.get("non_te_entitlement_preserved") is True and te.get("sportsbook_inputs_used") is False)
    wr_valid=bool(wr.get("disposition")=="WR_R15_FULL_SLATE_ENTITLEMENT_READY" and wr.get("m38_wr1_anchor_preserved") is True and wr.get("wr2plus_pool_preserved") is True and wr.get("non_wr_entitlement_preserved") is True and wr.get("sportsbook_inputs_used") is False)
    qb_valid=bool(qb.get("disposition")=="QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS" and qb.get("all_changed_keys_are_selected_qb_pass_yards") is True and int(qb.get("sportsbook_inputs_to_selector",1))==0 and int(qb.get("sportsbook_inputs_to_c2_generation",1))==0 and qb.get("te_r5p_consumed_before_c2") is True and qb.get("wr_r15_consumed_before_c2") is True)

    gates={
        "row_universe_exact":row_universe_exact,
        "adapter_integration_valid":adapter.get("integration_valid") is True,
        "exact_94_rb_adapted":int(adapter.get("adapted_rb_rows",-1))==94,
        "adapter_mean_parity":float(adapter.get("max_mean_delta",1.0))<=1e-8,
        "adapter_rank_preservation":float(adapter.get("min_spearman",-1.0))>=0.9999,
        "adapter_non_rb_exact":adapter.get("gates",{}).get("non_rb_exact") is True,
        "adapter_receptions_exact":adapter.get("gates",{}).get("receptions_exact") is True,
        "adapter_rb_other_markets_exact":adapter.get("gates",{}).get("rb_nonreceiving_markets_exact") is True,
        "adapter_rush_rec_identity":adapter.get("gates",{}).get("rush_rec_identity") is True,
        "pricing_lineage_valid":pricing.get("integration_valid") is True,
        "pricing_reaches_rb_rec_yards":int(pricing.get("rec_yards_rows_stamped",0))>0,
        "priced_model_means_exact_vs_same_checkout_v3":mean_checks.get("model_proj",{}).get("all_equal",False),
        "priced_mc_means_exact_vs_same_checkout_v3":mean_checks.get("mc_proj",{}).get("all_equal",False),
        "priced_receptions_exact_vs_same_checkout_v3":receptions_exact,
        "all_nonallowed_pricing_rows_exact":non_allowed_exact,
        "no_unexpected_probability_changes":not bool(unexpected.any()),
        "rb_receiving_probability_change_reaches_pricing":bool(allowed_changed.any()),
        "existing_te_stack_valid":te_valid,
        "existing_wr_stack_valid":wr_valid,
        "existing_qb_stack_valid":qb_valid,
        "sportsbook_zero_to_adapter":int(adapter.get("sportsbook_inputs_added",1))==0,
        "outcomes_zero_to_adapter":int(adapter.get("current_or_future_outcomes_used",1))==0,
    }
    passed=bool(all(gates.values()))

    merged["unexpected_probability_change"]=unexpected
    merged["allowed_probability_change_observed"]=allowed_changed
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    keep=KEYS+[team_col,"adapted_rb","allowed_distribution_change","unexpected_probability_change","allowed_probability_change_observed"]
    for col in ["model_proj","mc_proj","fair_prob","fair_odds","edge_pct","edge_abs"]:
        for s in ["_control","_candidate"]:
            x=f"{col}{s}"
            if x in merged.columns: keep.append(x)
    merged[keep].to_csv(OUT_CSV,index=False)

    payload={
        "candidate":"RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_V1",
        "comparison_contract":"SAME_CHECKOUT_SAME_INPUTS_SAME_SEED_V3_VS_V4",
        "disposition":"RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS" if passed else "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_FAIL",
        "pass":passed,
        "control_rows":int(len(control)),"candidate_rows":int(len(candidate)),
        "adapted_rb_player_keys":int(len(adapted_keys)),
        "allowed_probability_change_rows":int(allowed_changed.sum()),
        "unexpected_probability_change_rows":int(unexpected.sum()),
        "mean_checks":mean_checks,"gates":gates,
        "adapter_disposition":adapter.get("disposition"),
        "pricing_lineage_disposition":pricing.get("disposition"),
        "governance_note":"This is a mechanical repair of the comparison method only; frozen R22 football/integration gates were not weakened.",
    }
    OUT_JSON.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if passed else 1

if __name__=="__main__":
    raise SystemExit(main())
