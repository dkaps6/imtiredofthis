#!/usr/bin/env python3
"""No-outcome Stage-1 structural ablation for receiving-rule semantic repairs.

Frozen cells:
A0B0 current production
A1B0 middle_open unit normalization only
A0B1 slot-alignment carry only
A1B1 both

No target-game outcomes, sportsbook data, fitted parameters, multiplier changes,
threshold changes, or production mutations.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import build_bayesian_baseline, apply_bayesian_to_metrics
import scripts.modeling.simulation_rules as sr
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement

DATA = Path("data")
OUT = Path("outputs/receiving_rule_semantics_integrity_v1_stage1")
BASELINE_AUTH = Path("/tmp/baseline_authority/availability-opportunity-rule-order-v1/target_entitlement_rows.csv")

NON_TARGET_RULE_COLS = [
    "rules_plays_est", "rules_pass_rate", "rules_rush_share",
    "rules_ypt", "rules_ypc", "rules_ypa", "rules_catch_rate",
    "rules_volatility_mult", "rules_pass_eff_mult", "rules_rush_eff_mult",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing required artifact: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _alignment_map(pf: pd.DataFrame) -> dict[tuple[str, str], str]:
    if "alignment_position" not in pf.columns:
        raise RuntimeError("PlayerForm lacks preserved alignment_position")
    source_key = "player_clean_key"
    if source_key not in pf.columns:
        raise RuntimeError("PlayerForm lacks player_clean_key")
    out = {}
    for _, r in pf.drop_duplicates(["team", source_key]).iterrows():
        out[(str(r["team"]).upper().strip(), str(r[source_key]))] = str(r.get("alignment_position") or "").upper().strip()
    return out


def _slot_preserving_labels(players, align):
    labels = {}
    by_team = {}
    for p in players:
        if sr._is_wr(p.position, p.role):
            by_team.setdefault(p.team, []).append(p)
    for team, group in by_team.items():
        slots = []
        for p in group:
            key = (team, sr._key(p.player))
            a = align.get(key, "")
            if a == "SWR" or str(p.position).upper() == "SWR" or "SLOT" in str(p.role).upper():
                slots.append(p)
        for p in slots:
            labels[(team, sr._key(p.player))] = "SLOT"
        perim = [p for p in group if p not in slots]
        perim.sort(key=lambda p: sr._num(p.features.get("tgt_share"), 0.0), reverse=True)
        if perim:
            labels[(team, sr._key(perim[0].player))] = "WR1"
        if len(perim) > 1:
            labels[(team, sr._key(perim[1].player))] = "WR1_5"
    return labels


def _normalized_middle_matchup(original):
    def wrapped(offense, defense):
        middle = sr._num(defense.middle_open_rate)
        if not np.isfinite(middle):
            raise RuntimeError("middle_open_rate is non-finite")
        if 0.0 <= middle <= 1.0:
            canon = float(middle)
        elif 1.0 < middle <= 100.0:
            canon = float(middle) / 100.0
        else:
            raise RuntimeError(f"middle_open_rate outside semantic contract: {middle}")
        return original(offense, replace(defense, middle_open_rate=canon))
    return wrapped


def _prepare_metrics(pf: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    bayes = build_bayesian_baseline(pf)
    metrics = apply_bayesian_to_metrics(pf, baseline=bayes)
    c = context.copy()
    c.columns = [str(x).strip().lower() for x in c.columns]
    if "player_clean_key" not in c.columns:
        pkeys = pf[["team", "player", "player_clean_key"]].drop_duplicates()
        c = c.merge(pkeys, on=["team", "player"], how="left", validate="one_to_one")
    identity = c[["team", "player_clean_key", "game_id"]].drop_duplicates().rename(columns={"game_id": "event_id"})
    metrics = metrics.merge(identity, on=["team", "player_clean_key"], how="left", validate="many_to_one")
    if metrics["event_id"].isna().any():
        raise RuntimeError("missing event_id after preserved context identity join")
    return metrics


def _run_cell(name, metrics, align, *, fix_middle: bool, fix_slot: bool, orig_matchup, orig_labels):
    sr.matchup_multipliers = _normalized_middle_matchup(orig_matchup) if fix_middle else orig_matchup
    if fix_slot:
        sr._wr_role_labels = lambda players: _slot_preserving_labels(players, align)
    else:
        sr._wr_role_labels = orig_labels
    rules = sr.apply_rules_to_metrics(metrics.copy())
    entitlement, trace = materialize_target_entitlement(rules.copy())
    rules["cell"] = name
    entitlement["cell"] = name
    trace["cell"] = name
    return rules, entitlement, trace


def _num_gap(a, b):
    aa = pd.to_numeric(a, errors="coerce").to_numpy(float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(float)
    both_nan = np.isnan(aa) & np.isnan(bb)
    one_nan = np.isnan(aa) ^ np.isnan(bb)
    diff = np.abs(np.nan_to_num(aa) - np.nan_to_num(bb))
    if one_nan.any():
        return float("inf")
    diff[both_nan] = 0.0
    return float(diff.max()) if len(diff) else 0.0


def main():
    pf = _read(DATA / "player_form_consensus.csv")
    context = _read(DATA / "model_context_bridge.csv")
    align = _alignment_map(pf)
    metrics = _prepare_metrics(pf, context)

    wr_pf = pf.loc[pf.get("position_group", pf.get("position")).astype(str).str.upper().eq("WR")].copy()
    swr_count = int(wr_pf["alignment_position"].astype(str).str.upper().eq("SWR").sum())

    team_context = _read(DATA / "team_context_v3.csv")
    mids = pd.to_numeric(team_context["middle_open_rate"], errors="coerce")
    middle_gt1 = int(mids.gt(1.0).sum())
    middle_invalid = int((mids.lt(0) | mids.gt(100) | mids.isna()).sum())

    orig_matchup = sr.matchup_multipliers
    orig_labels = sr._wr_role_labels
    cells = {}
    try:
        for name, fm, fs in [
            ("A0B0", False, False),
            ("A1B0", True, False),
            ("A0B1", False, True),
            ("A1B1", True, True),
        ]:
            cells[name] = _run_cell(
                name, metrics, align,
                fix_middle=fm, fix_slot=fs,
                orig_matchup=orig_matchup, orig_labels=orig_labels,
            )
    finally:
        sr.matchup_multipliers = orig_matchup
        sr._wr_role_labels = orig_labels

    base_rules, base_ent, _ = cells["A0B0"]

    # Exact A0 parity against previously accepted same-authority entitlement artifact.
    auth = _read(BASELINE_AUTH)
    key = ["event_id", "team", "player_clean_key"]
    parity = base_ent.merge(
        auth[key + ["entitlement_tgt_share"]],
        on=key, how="outer", suffixes=("_replay", "_authority"), indicator=True
    )
    if not parity["_merge"].eq("both").all():
        raise RuntimeError(f"A0 entitlement identity drift: {parity['_merge'].value_counts().to_dict()}")
    parity_gap = _num_gap(parity["entitlement_tgt_share_replay"], parity["entitlement_tgt_share_authority"])
    if parity_gap > 1e-12:
        raise RuntimeError(f"A0 entitlement replay drift: {parity_gap}")

    summaries = []
    deltas = []
    role_rows = []
    for name, (rules, ent, trace) in cells.items():
        wr = rules.loc[rules["position"].astype(str).str.upper().eq("WR")].copy()
        role_counts = wr["rules_role"].fillna("").astype(str).value_counts().to_dict()
        role_rows.append({
            "cell": name,
            "wr_rows": int(len(wr)),
            "wr1": int(role_counts.get("WR1", 0)),
            "wr1_5": int(role_counts.get("WR1_5", 0)),
            "slot": int(role_counts.get("SLOT", 0)),
            "unlabeled": int(role_counts.get("", 0)),
        })

        merged_r = base_rules[["team","player_clean_key","position","rules_tgt_share",*NON_TARGET_RULE_COLS]].merge(
            rules[["team","player_clean_key","rules_tgt_share",*NON_TARGET_RULE_COLS]],
            on=["team","player_clean_key"], how="inner", suffixes=("_base","_cell"), validate="one_to_one"
        )
        target_delta = (
            pd.to_numeric(merged_r["rules_tgt_share_cell"], errors="coerce")
            - pd.to_numeric(merged_r["rules_tgt_share_base"], errors="coerce")
        )
        abs_target = target_delta.abs()
        non_target_max = 0.0
        for c in NON_TARGET_RULE_COLS:
            non_target_max = max(non_target_max, _num_gap(merged_r[f"{c}_base"], merged_r[f"{c}_cell"]))

        merged_e = base_ent[key + ["position","entitlement_tgt_share"]].merge(
            ent[key + ["entitlement_tgt_share"]],
            on=key, how="inner", suffixes=("_base","_cell"), validate="one_to_one"
        )
        e_delta = (
            pd.to_numeric(merged_e["entitlement_tgt_share_cell"], errors="coerce")
            - pd.to_numeric(merged_e["entitlement_tgt_share_base"], errors="coerce")
        )
        abs_e = e_delta.abs()

        summaries.append({
            "cell": name,
            "wr_slot_labels": int(role_counts.get("SLOT", 0)),
            "rules_target_rows_changed": int(abs_target.gt(1e-12).sum()),
            "rules_target_median_abs_delta": float(abs_target[abs_target.gt(1e-12)].median()) if abs_target.gt(1e-12).any() else 0.0,
            "rules_target_max_abs_delta": float(abs_target.max()),
            "wr_rules_target_rows_changed": int((abs_target.gt(1e-12) & merged_r["position"].astype(str).str.upper().eq("WR")).sum()),
            "te_rules_target_rows_changed": int((abs_target.gt(1e-12) & merged_r["position"].astype(str).str.upper().eq("TE")).sum()),
            "entitlement_rows_changed": int(abs_e.gt(1e-12).sum()),
            "entitlement_median_abs_delta": float(abs_e[abs_e.gt(1e-12)].median()) if abs_e.gt(1e-12).any() else 0.0,
            "entitlement_max_abs_delta": float(abs_e.max()),
            "non_target_rule_max_abs_delta": float(non_target_max),
        })

        if name != "A0B0":
            tmp = merged_e.loc[abs_e.gt(1e-12), key + ["position"]].copy()
            tmp["cell"] = name
            tmp["baseline_entitlement"] = merged_e.loc[abs_e.gt(1e-12), "entitlement_tgt_share_base"].to_numpy()
            tmp["candidate_entitlement"] = merged_e.loc[abs_e.gt(1e-12), "entitlement_tgt_share_cell"].to_numpy()
            tmp["delta"] = e_delta.loc[abs_e.gt(1e-12)].to_numpy()
            deltas.append(tmp)

    summary_df = pd.DataFrame(summaries)
    roles_df = pd.DataFrame(role_rows)
    delta_df = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT / "cell_summary.csv", index=False)
    roles_df.to_csv(OUT / "wr_role_counts.csv", index=False)
    delta_df.sort_values(["cell","delta"], key=lambda s: s.abs() if s.name=="delta" else s, ascending=False).to_csv(
        OUT / "entitlement_deltas.csv", index=False
    )
    for name, (rules, ent, trace) in cells.items():
        rules.to_csv(OUT / f"rules_{name}.csv", index=False)
        ent.to_csv(OUT / f"entitlement_{name}.csv", index=False)
        trace.to_csv(OUT / f"trace_{name}.csv", index=False)

    payload = {
        "disposition": "STRUCTURAL_ABLATION_COMPLETE_NO_OUTCOMES",
        "stage": 1,
        "season": 2026,
        "week": 3,
        "candidate_cells": ["A1B0","A0B1","A1B1"],
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_read": 0,
        "production_mutations": 0,
        "playerform_wr_rows": int(len(wr_pf)),
        "playerform_swr_rows": swr_count,
        "team_context_rows": int(len(team_context)),
        "middle_open_values_gt_1": middle_gt1,
        "middle_open_invalid_values": middle_invalid,
        "a0_entitlement_authority_max_abs_gap": parity_gap,
        "cells": summary_df.to_dict("records"),
        "wr_roles": roles_df.to_dict("records"),
    }

    def default(v):
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    rendered = json.dumps(payload, indent=2, sort_keys=True, default=default)
    (OUT / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
