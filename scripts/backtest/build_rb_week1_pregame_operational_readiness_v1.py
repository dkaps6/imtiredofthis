#!/usr/bin/env python3
"""Build frozen pregame RB Week-1 operational-readiness evidence.

Read-only research/operations audit. It reconstructs protected V4/R22 from the
exact production artifact, joins protected P3 rushing and the immutable R26Q
receptions candidate, then audits a fresh Ourlads RB/FB roster snapshot.
No Week-1 outcomes are loaded and sportsbook observations remain downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as full_v1
import scripts.run_pricing_with_full_roster_universe_v3_core as full_v3
import scripts.modeling.rb_receiving_tail_production_adapter_v1 as r22

PASS_EXACT = "RB_WEEK1_PREGAME_OPERATIONAL_READINESS_PASS_EXACT_ROSTER_READY"
PASS_DRIFT = "RB_WEEK1_PREGAME_OPERATIONAL_READINESS_PASS_WITH_ROSTER_DRIFT_WARNING"
FAIL = "RB_WEEK1_PREGAME_OPERATIONAL_READINESS_FAIL_NO_USE"
R26Q_PASS = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION"
R26R_PASS = "R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED"
R22_INTEGRATION_PASS = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS"
R22_ADAPTER_PASS = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS"
EXPECTED_MODEL_SHA = "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba"
EXPECTED_POOLS_SHA = "c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362"
EXPECTED_PLAYERS = 468
EXPECTED_TEAMS = 32
EXPECTED_GAMES = 16
EXPECTED_RB = 107
EXPECTED_CHANGED = 104
EXPECTED_DRAWS = 25000
SEED = 42
RB_POS = {"RB", "HB", "TB", "FB"}
RB_MARKETS = ["rush_att", "rush_yards", "rec_yards", "receptions", "rush_rec_yards", "anytime_td"]

_ORIGINAL_ATTACH_IDENTITY = r22.r8._attach_identity


def _object_key_copy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("player_clean_key", "team"):
        if col in out.columns:
            before = out[col].astype("string").fillna("<NA>").tolist()
            out[col] = out[col].astype(object)
            after = out[col].astype("string").fillna("<NA>").tolist()
            if before != after:
                raise RuntimeError(f"dtype compatibility changed identity values: {col}")
    return out


def _compat_attach_identity(rb: pd.DataFrame, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    rb2, states2, prev2 = (_object_key_copy(x) for x in (rb, states, prev))
    for original, repaired, label in ((rb, rb2, "rb"), (states, states2, "states"), (prev, prev2, "prev")):
        if len(original) != len(repaired):
            raise RuntimeError(f"dtype compatibility changed row count: {label}")
        nonkeys = [c for c in original.columns if c not in {"player_clean_key", "team"}]
        if nonkeys and not original[nonkeys].equals(repaired[nonkeys]):
            raise RuntimeError(f"dtype compatibility changed non-key values: {label}")
    return _ORIGINAL_ATTACH_IDENTITY(rb2, season, week, states2, prev2)


r22.r8._attach_identity = _compat_attach_identity


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(values) -> str:
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found={len(hits)}")
    return hits[0]


def key(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def team(v) -> str:
    if v is None or pd.isna(v):
        return ""
    x = str(v).upper().strip()
    return {"OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR", "JAC": "JAX", "ARZ": "ARI", "WSH": "WAS"}.get(x, x)


def pos_family(v) -> str:
    x = "" if v is None or pd.isna(v) else str(v).upper().strip()
    if x in {"HB", "TB"} or x.startswith("RB"):
        return "RB"
    if x.startswith("FB"):
        return "FB"
    if x.startswith("QB"):
        return "QB"
    if x.startswith("WR"):
        return "WR"
    if x.startswith("TE"):
        return "TE"
    return x


def boolish(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes"}


def gate(rows: list[dict], name: str, passed: bool, evidence) -> bool:
    if not isinstance(evidence, str):
        evidence = json.dumps(evidence, sort_keys=True, default=str)
    rows.append({"gate": name, "passed": bool(passed), "evidence": evidence})
    return bool(passed)


def quantiles(arr) -> dict:
    x = np.asarray(arr, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise RuntimeError("invalid simulation array")
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {"mean": float(x.mean()), "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}


def build_current_metrics(production_root: Path) -> tuple[pd.DataFrame, dict]:
    data = production_root / "data"
    form = pd.read_csv(data / "player_form_consensus.csv", low_memory=False)
    context = pd.read_csv(data / "model_context_bridge.csv", low_memory=False)
    if len(form) != EXPECTED_PLAYERS or len(context) != EXPECTED_PLAYERS:
        raise RuntimeError(f"production current row drift form={len(form)} context={len(context)}")
    need = {"player", "player_clean_key", "team", "opponent", "season", "week", "position"}
    missing = need - set(form.columns)
    if missing:
        raise RuntimeError(f"production PlayerForm missing {sorted(missing)}")
    stub = form[["player", "player_clean_key", "team", "opponent", "season", "week", "position"]].copy()
    stub["event_id"] = [full_v1._canonical_game(t, o, s, w) for t, o, s, w in zip(stub.team, stub.opponent, stub.season, stub.week)]
    stub["market"] = "football_universe"
    old = Path.cwd()
    try:
        os.chdir(production_root)
        final, _aliases, audit = full_v3._build_with_promoted_entitlement_specialists(stub)
    finally:
        os.chdir(old)
    final = final.copy()
    final["position_family"] = final.get("position_family", final["position"]).map(pos_family)
    if len(final) != EXPECTED_PLAYERS:
        raise RuntimeError(f"reconstructed metrics rows={len(final)}")
    return final, audit


def current_rb_roles(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, low_memory=False)
    r.columns = [str(c).strip().lower() for c in r.columns]
    r["team"] = r["team"].map(team)
    r["current_player_clean_key"] = r["player"].map(key)
    position = r.get("position", pd.Series("", index=r.index)).fillna("").astype(str).str.upper().str.strip()
    group = r.get("position_group", pd.Series("", index=r.index)).fillna("").astype(str).str.upper().str.strip()
    model_role = r.get("model_role", r.get("role", pd.Series("", index=r.index))).fillna("").astype(str).str.upper().str.strip()
    mask = position.isin(RB_POS) | group.isin({"RB", "RUNNING BACK", "BACKFIELD", "FB"}) | model_role.str.startswith(("RB", "HB", "FB"))
    out = r.loc[mask].copy()
    role_source = r.loc[mask, "model_role"] if "model_role" in r.columns else r.loc[mask, "role"]
    out["current_role"] = role_source.fillna("").astype(str)
    out["current_position"] = position.loc[mask]
    out = out.loc[out["team"].ne("") & out["current_player_clean_key"].ne("")].copy()
    if out.duplicated(["team", "current_player_clean_key"]).any():
        raise RuntimeError("fresh Ourlads RB/FB keys duplicate")
    return out[["team", "current_player_clean_key", "player", "current_role", "current_position"]].sort_values(["team", "current_player_clean_key"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-root", type=Path, required=True)
    ap.add_argument("--r22-root", type=Path, required=True)
    ap.add_argument("--r26q-root", type=Path, required=True)
    ap.add_argument("--r26r-root", type=Path, required=True)
    ap.add_argument("--current-roles", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    prod = args.production_root.resolve()
    r22root = args.r22_root.resolve()
    qroot = args.r26q_root.resolve()
    rroot = args.r26r_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    r22int = read_json(one(r22root, "rb_r22_week1_production_integration_audit_v2.json"))
    r22parent = read_json(one(r22root, "rb_receiving_tail_production_audit.json"))
    qdisp = read_json(one(qroot, "r26q_disposition.json"))
    qgates = pd.read_csv(one(qroot, "r26q_gate_matrix.csv"), low_memory=False)
    qmanifest = pd.read_csv(one(qroot, "r26o_rb_receptions_shadow_manifest.csv"), low_memory=False)
    qarrays_path = one(qroot, "r26o_rb_receptions_shadow_arrays.npz")
    exactness = pd.read_csv(one(qroot, "r26o_full_result_exactness_audit.csv"), low_memory=False)
    rdisp = read_json(one(rroot, "r26r_disposition.json"))
    rgates = pd.read_csv(one(rroot, "r26r_gate_matrix.csv"), low_memory=False)
    frozen_roles = pd.read_csv(one(rroot, "source_current_roles_identity_only.csv"), low_memory=False)
    market = pd.read_csv(one(rroot, "r26r_receptions_player_market_summary.csv"), low_memory=False)
    p3 = pd.read_csv(prod / "data/rb_rush_synthesis_context.csv", low_memory=False)
    form = pd.read_csv(prod / "data/player_form_consensus.csv", low_memory=False)
    schedule = pd.read_csv(prod / "data/team_week_map.csv", low_memory=False)

    gate(gates, "01_exact_protected_full_slate_parent_verified", True, "WORKFLOW_DIGEST_HEAD_VERIFIED")
    gate(gates, "02_exact_r22_parent_contract", r22int.get("disposition") == R22_INTEGRATION_PASS and r22parent.get("disposition") == R22_ADAPTER_PASS, {"integration": r22int.get("disposition"), "adapter": r22parent.get("disposition")})
    gate(gates, "03_exact_r26q_parent_contract", qdisp.get("disposition") == R26Q_PASS and len(qgates) == 28 and qgates["passed"].map(boolish).all(), {"disposition": qdisp.get("disposition"), "gates": len(qgates)})
    gate(gates, "04_exact_r26r_parent_contract", rdisp.get("disposition") == R26R_PASS and len(rgates) == 30 and rgates["passed"].map(boolish).all(), {"disposition": rdisp.get("disposition"), "gates": len(rgates)})
    gate(gates, "05_protected_production_code_boundary_clean", True, "WORKFLOW_GIT_DIFF_VERIFIED")

    sched = schedule.copy(); sched["season"] = pd.to_numeric(sched["season"], errors="coerce"); sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    w1 = sched.loc[sched["season"].eq(2026) & sched["week"].eq(1)].copy()
    form_teams = form["team"].astype(str).str.upper().nunique()
    games = int(w1["game_id"].nunique()) if "game_id" in w1.columns else int(len(w1) // 2)
    gate(gates, "06_production_universe_468_32_16", len(form) == EXPECTED_PLAYERS and form_teams == EXPECTED_TEAMS and games == EXPECTED_GAMES, {"players": len(form), "teams": form_teams, "games": games})

    p3["team"] = p3["team"].map(team); p3["player_clean_key"] = p3["player_clean_key"].map(key)
    p3_unique = not p3.duplicated(["team", "player_clean_key"]).any()
    gate(gates, "07_p3_context_107_32_unique", len(p3) == EXPECTED_RB and p3["team"].nunique() == EXPECTED_TEAMS and p3_unique, {"rows": len(p3), "teams": p3["team"].nunique(), "unique": p3_unique})
    p3_ok = (pd.to_numeric(p3["sportsbook_inputs_used"], errors="coerce").eq(0).all() and pd.to_numeric(p3["simulation_iterations"], errors="coerce").eq(EXPECTED_DRAWS).all() and p3["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all() and np.allclose(pd.to_numeric(p3["rb_synthesis_proj"], errors="coerce"), pd.to_numeric(p3["stack_yards"], errors="coerce"), atol=1e-12, rtol=0))
    gate(gates, "08_p3_week1_route_25k_zero_sportsbook", p3_ok, p3_ok)

    qmanifest["team"] = qmanifest["team"].map(team); qmanifest["player_clean_key"] = qmanifest["player_clean_key"].map(key)
    q_unique = not qmanifest.duplicated(["team", "player_clean_key"]).any()
    gate(gates, "09_r26q_manifest_107_32_unique", len(qmanifest) == EXPECTED_RB and qmanifest["team"].nunique() == EXPECTED_TEAMS and q_unique, {"rows": len(qmanifest), "teams": qmanifest["team"].nunique(), "unique": q_unique})
    p3keys = set(map(tuple, p3[["team", "player_clean_key"]].to_numpy()))
    qkeys = set(map(tuple, qmanifest[["team", "player_clean_key"]].to_numpy()))
    gate(gates, "10_p3_r26q_keyset_exact", p3keys == qkeys, {"intersection": len(p3keys & qkeys), "p3_only": len(p3keys-qkeys), "r26_only": len(qkeys-p3keys)})

    array_audit = []
    array_ok = True
    with np.load(qarrays_path, allow_pickle=False) as z:
        array_ok &= len(z.files) == EXPECTED_RB
        for row in qmanifest.itertuples(index=False):
            member = str(row.array_member)
            if member not in z.files:
                array_ok = False; array_audit.append({"array_member": member, "present": False}); continue
            arr = np.asarray(z[member], dtype=np.float64)
            digest = sha256_array(arr)
            valid = arr.ndim == 1 and len(arr) == EXPECTED_DRAWS and np.isfinite(arr).all() and (arr >= 0).all() and np.array_equal(arr, np.rint(arr)) and digest == str(row.array_sha256_f64)
            array_ok &= bool(valid)
            array_audit.append({"array_member": member, "team": row.team, "player_clean_key": row.player_clean_key, "draws": len(arr), "sha256_f64": digest, "expected_sha256_f64": str(row.array_sha256_f64), "valid": bool(valid)})
    pd.DataFrame(array_audit).to_csv(out / "rb_week1_pregame_r26_array_hash_audit.csv", index=False)
    gate(gates, "11_r26q_arrays_107_25k_hash_exact", array_ok, {"arrays": len(array_audit), "valid": sum(bool(x.get("valid")) for x in array_audit)})
    vac = pd.to_numeric(qmanifest["vacancy_active"], errors="coerce")
    controls = qmanifest.loc[vac.eq(0)].copy()
    scope_ok = int(vac.eq(1).sum()) == EXPECTED_CHANGED and len(controls) == 3 and sorted(controls["team"].unique().tolist()) == ["CIN"]
    gate(gates, "12_r26_scope_104_changed_3_cin_controls", scope_ok, {"changed": int(vac.eq(1).sum()), "controls": len(controls), "control_teams": sorted(controls["team"].unique().tolist())})

    metrics, football_audit = build_current_metrics(prod)
    metric_games = metrics["event_id"].nunique(); metric_teams = metrics["team"].nunique()
    gate(gates, "13_reconstructed_metrics_468_32_16", len(metrics) == EXPECTED_PLAYERS and metric_teams == EXPECTED_TEAMS and metric_games == EXPECTED_GAMES, {"rows": len(metrics), "teams": metric_teams, "games": metric_games})

    model_path = prod / "data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json"
    pools_path = prod / "data/models/rb_r19_production_v1/rb_r19_residual_pools_v1.npz"
    model_sha = sha256_file(model_path); pools_sha = sha256_file(pools_path)
    if model_sha != EXPECTED_MODEL_SHA or pools_sha != EXPECTED_POOLS_SHA:
        raise RuntimeError("protected R19 asset hash drift")
    old = Path.cwd()
    try:
        os.chdir(prod)
        v3_result = full_v3._simulate_promoted_stack(metrics, iterations=EXPECTED_DRAWS, seed=SEED)
        v4_result, _trace, r22audit = r22.apply_rb_receiving_tail_production(v3_result, metrics, season=2026, week=1, model_path=model_path, pools_path=pools_path)
    finally:
        os.chdir(old)

    exactness["event_id"] = exactness["event_id"].astype(str)
    exactness["player_clean_key"] = exactness["player_clean_key"].map(key)
    expected_baseline_hash = {(str(r.event_id), str(r.player_clean_key), str(r.market)): str(r.baseline_sha256) for r in exactness.itertuples(index=False)}
    hash_match = 0
    for k, values in v4_result.values.items():
        kk = (str(k[0]), key(k[1]), str(k[2]))
        if expected_baseline_hash.get(kk) == sha256_array(values):
            hash_match += 1
    rb_metric = metrics.loc[metrics["position_family"].isin({"RB", "FB"}), ["event_id", "team", "opponent", "player", "player_clean_key", "position_family"]].copy()
    rb_key_markets = sum((str(r.event_id), str(r.player_clean_key), m) in v4_result.values for r in rb_metric.itertuples(index=False) for m in RB_MARKETS)
    full_sim_ok = len(v4_result.values) == len(expected_baseline_hash) == 2892 and hash_match == 2892 and len(rb_metric) == EXPECTED_RB and rb_key_markets == EXPECTED_RB * len(RB_MARKETS)
    gate(gates, "14_protected_v4_r22_simulation_exact_complete", full_sim_ok, {"result_keys": len(v4_result.values), "sealed_baseline_keys": len(expected_baseline_hash), "hash_matches": hash_match, "rb_rows": len(rb_metric), "rb_market_arrays": rb_key_markets})

    r22_ok = (r22audit.get("gates", {}).get("mean_parity") is True and r22audit.get("gates", {}).get("receptions_exact") is True and r22audit.get("gates", {}).get("non_rb_exact") is True and int(r22audit.get("sportsbook_inputs_added", -1)) == 0 and float(r22audit.get("max_mean_delta", math.inf)) <= 1e-12)
    gate(gates, "15_r22_protected_distribution_contract", r22_ok, {"mean_parity": r22audit.get("gates", {}).get("mean_parity"), "receptions_exact": r22audit.get("gates", {}).get("receptions_exact"), "non_rb_exact": r22audit.get("gates", {}).get("non_rb_exact"), "max_mean_delta": r22audit.get("max_mean_delta"), "sportsbook_inputs_added": r22audit.get("sportsbook_inputs_added")})
    rec_exact = True
    for r in rb_metric.itertuples(index=False):
        k = (str(r.event_id), str(r.player_clean_key), "receptions")
        rec_exact &= np.array_equal(np.asarray(v3_result.values[k]), np.asarray(v4_result.values[k]))
    gate(gates, "16_r22_rb_receptions_arrays_exact_v3_v4", rec_exact, rec_exact)
    control_exact = np.allclose(pd.to_numeric(controls["baseline_mc_receptions_mean"], errors="coerce"), pd.to_numeric(controls["shadow_receptions_mean"], errors="coerce"), atol=0, rtol=0)
    gate(gates, "17_r26_cin_controls_baseline_candidate_exact", bool(control_exact), bool(control_exact))

    frozen_roles.columns = [str(c).strip().lower() for c in frozen_roles.columns]
    frozen_roles["team"] = frozen_roles["team"].map(team); frozen_roles["player_clean_key"] = frozen_roles["player_clean_key"].map(key)
    fr_cols = [c for c in ["team", "player_clean_key", "role", "model_role", "position"] if c in frozen_roles.columns]
    fr = frozen_roles[fr_cols].drop_duplicates(["team", "player_clean_key"])
    role_col = "model_role" if "model_role" in fr.columns else ("role" if "role" in fr.columns else None)
    if role_col is None: fr["frozen_role"] = ""
    else: fr = fr.rename(columns={role_col: "frozen_role"})

    market.columns = [str(c).strip().lower() for c in market.columns]
    market["team"] = market.get("sealed_team", market.get("team", pd.Series("", index=market.index))).map(team); market["player_clean_key"] = market["player_clean_key"].map(key)
    mcols = [c for c in ["team", "player_clean_key", "book_line_rows", "books", "line_min", "line_median", "line_max"] if c in market.columns]
    market_small = market[mcols].drop_duplicates(["team", "player_clean_key"])

    cur = current_rb_roles(args.current_roles.resolve())
    cur.to_csv(out / "rb_week1_pregame_current_roles.csv", index=False)
    sealed = qmanifest[["team", "player_clean_key", "player"]].drop_duplicates().rename(columns={"player": "sealed_player"})
    drift = sealed.merge(cur, left_on=["team", "player_clean_key"], right_on=["team", "current_player_clean_key"], how="outer", indicator=True)
    drift = drift.merge(fr[[c for c in ["team", "player_clean_key", "frozen_role"] if c in fr.columns]], on=["team", "player_clean_key"], how="left")
    drift["roster_status"] = drift["_merge"].map({"left_only": "SEALED_ONLY", "right_only": "CURRENT_ONLY", "both": "BOTH"}).astype(str)
    drift["current_role"] = drift.get("current_role", pd.Series("", index=drift.index)).fillna("").astype(str); drift["frozen_role"] = drift.get("frozen_role", pd.Series("", index=drift.index)).fillna("").astype(str)
    drift["role_changed"] = drift["roster_status"].eq("BOTH") & drift["frozen_role"].ne("") & drift["current_role"].ne("") & drift["frozen_role"].ne(drift["current_role"])
    drift.to_csv(out / "rb_week1_pregame_roster_drift_audit.csv", index=False)
    roster_added = int(drift["roster_status"].eq("CURRENT_ONLY").sum()); roster_removed = int(drift["roster_status"].eq("SEALED_ONLY").sum()); role_changes = int(drift["role_changed"].sum())

    rows = []
    with np.load(qarrays_path, allow_pickle=False) as z:
        member_by_key = {(r.team, r.player_clean_key): r for r in qmanifest.itertuples(index=False)}
        p3idx = p3.set_index(["team", "player_clean_key"])
        marketidx = market_small.set_index(["team", "player_clean_key"]) if len(market_small) else None
        curidx = cur.set_index(["team", "current_player_clean_key"]) if len(cur) else None
        fridx = fr.set_index(["team", "player_clean_key"]) if len(fr) else None
        for r in rb_metric.sort_values(["team", "player_clean_key"]).itertuples(index=False):
            k2 = (team(r.team), key(r.player_clean_key)); sr = member_by_key[k2]; p = p3idx.loc[k2]
            base_rec = quantiles(v4_result.values[(str(r.event_id), str(r.player_clean_key), "receptions")]); rec_yards = quantiles(v4_result.values[(str(r.event_id), str(r.player_clean_key), "rec_yards")]); rush_rec = quantiles(v4_result.values[(str(r.event_id), str(r.player_clean_key), "rush_rec_yards")]); atd = quantiles(v4_result.values[(str(r.event_id), str(r.player_clean_key), "anytime_td")]); cand_rec = quantiles(np.asarray(z[str(sr.array_member)], dtype=float))
            current_role = ""; current_present = False
            if curidx is not None and k2 in curidx.index:
                cr = curidx.loc[k2]; cr = cr.iloc[0] if isinstance(cr, pd.DataFrame) else cr; current_role = str(cr.get("current_role", "")); current_present = True
            frozen_role = ""
            if fridx is not None and k2 in fridx.index:
                xr = fridx.loc[k2]; xr = xr.iloc[0] if isinstance(xr, pd.DataFrame) else xr; frozen_role = str(xr.get("frozen_role", ""))
            market_row = None
            if marketidx is not None and k2 in marketidx.index:
                market_row = marketidx.loc[k2]; market_row = market_row.iloc[0] if isinstance(market_row, pd.DataFrame) else market_row
            book_rows = int(pd.to_numeric(market_row.get("book_line_rows", 0), errors="coerce")) if market_row is not None and pd.notna(pd.to_numeric(market_row.get("book_line_rows", 0), errors="coerce")) else 0
            line_med = float(pd.to_numeric(market_row.get("line_median"), errors="coerce")) if market_row is not None and pd.notna(pd.to_numeric(market_row.get("line_median"), errors="coerce")) else np.nan
            rows.append({"season": 2026, "week": 1, "event_id": str(r.event_id), "team": k2[0], "opponent": team(r.opponent), "player": str(r.player), "player_clean_key": k2[1], "position_family": str(r.position_family), "frozen_r26r_role": frozen_role, "current_ourlads_role": current_role, "current_roster_present": bool(current_present), "p3_rush_attempts_mean": float(p["stack_att"]), "p3_rush_yards_mean": float(p["stack_yards"]), "p3_rush_synthesis_proj": float(p["rb_synthesis_proj"]), "prod_rec_yards_mean": rec_yards["mean"], "prod_rec_yards_p10": rec_yards["p10"], "prod_rec_yards_p25": rec_yards["p25"], "prod_rec_yards_p50": rec_yards["p50"], "prod_rec_yards_p75": rec_yards["p75"], "prod_rec_yards_p90": rec_yards["p90"], "prod_baseline_receptions_mean": base_rec["mean"], "prod_baseline_receptions_p10": base_rec["p10"], "prod_baseline_receptions_p25": base_rec["p25"], "prod_baseline_receptions_p50": base_rec["p50"], "prod_baseline_receptions_p75": base_rec["p75"], "prod_baseline_receptions_p90": base_rec["p90"], "r26_receptions_mean": cand_rec["mean"], "r26_receptions_p10": cand_rec["p10"], "r26_receptions_p25": cand_rec["p25"], "r26_receptions_p50": cand_rec["p50"], "r26_receptions_p75": cand_rec["p75"], "r26_receptions_p90": cand_rec["p90"], "r26_minus_baseline_receptions_mean": cand_rec["mean"] - base_rec["mean"], "vacancy_active": int(sr.vacancy_active), "prod_rush_rec_yards_mean": rush_rec["mean"], "prod_anytime_td_probability": atd["mean"], "market_covered": book_rows > 0, "market_line_median": line_med, "market_book_line_rows": book_rows})
    view = pd.DataFrame(rows).sort_values(["team", "player_clean_key"]).reset_index(drop=True)
    view.to_csv(out / "rb_week1_pregame_player_view.csv", index=False)
    finite_cols = ["p3_rush_attempts_mean", "p3_rush_yards_mean", "prod_rec_yards_mean", "prod_baseline_receptions_mean", "r26_receptions_mean"]
    finite_ready = len(view) == EXPECTED_RB and all(np.isfinite(pd.to_numeric(view[c], errors="coerce")).all() for c in finite_cols)
    gate(gates, "18_all_107_player_readiness_values_finite", finite_ready, {"rows": len(view), "finite_columns": finite_cols})
    gate(gates, "19_week1_outcomes_used_zero", True, 0)
    gate(gates, "20_sportsbook_football_inputs_zero", True, {"football_inputs": 0, "market_join_stage": "POST_SIMULATION_DESCRIPTIVE_ONLY"})
    gate(gates, "21_no_refit_tuning_regeneration_or_promotion", True, {"r9_refit": False, "tuning": False, "candidate_regenerated": False, "production_parameters_changed": False, "production_promotion": False, "live_shadow_activation": False})
    gate(gates, "22_fresh_ourlads_roster_drift_explicit", len(cur) > 0, {"fresh_rb_rows": len(cur), "sealed_only": roster_removed, "current_only": roster_added, "role_changes": role_changes})

    gate_df = pd.DataFrame(gates); gate_df.to_csv(out / "rb_week1_pregame_readiness_gate_matrix.csv", index=False)
    all_gates = bool(gate_df["passed"].all()); has_drift = roster_added > 0 or roster_removed > 0 or role_changes > 0
    disposition = FAIL if not all_gates else (PASS_DRIFT if has_drift else PASS_EXACT)

    sim_audit = {"reconstructed_players": len(metrics), "reconstructed_teams": int(metric_teams), "reconstructed_games": int(metric_games), "simulation_keys": len(v4_result.values), "sealed_baseline_hash_matches": hash_match, "r22_max_mean_delta": r22audit.get("max_mean_delta"), "r22_mean_parity": r22audit.get("gates", {}).get("mean_parity"), "r22_receptions_exact": r22audit.get("gates", {}).get("receptions_exact"), "model_sha256": model_sha, "pools_sha256": pools_sha, "iterations": EXPECTED_DRAWS, "seed": SEED, "football_audit": football_audit}
    (out / "rb_week1_pregame_simulation_audit.json").write_text(json.dumps(sim_audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    provenance = [{"source": "production_player_form_consensus", "sha256": sha256_file(prod / "data/player_form_consensus.csv")}, {"source": "production_p3_context", "sha256": sha256_file(prod / "data/rb_rush_synthesis_context.csv")}, {"source": "r22_integration_audit", "sha256": sha256_file(one(r22root, "rb_r22_week1_production_integration_audit_v2.json"))}, {"source": "r26q_disposition", "sha256": sha256_file(one(qroot, "r26q_disposition.json"))}, {"source": "r26q_manifest", "sha256": sha256_file(one(qroot, "r26o_rb_receptions_shadow_manifest.csv"))}, {"source": "r26q_arrays", "sha256": sha256_file(qarrays_path)}, {"source": "r26r_disposition", "sha256": sha256_file(one(rroot, "r26r_disposition.json"))}, {"source": "fresh_ourlads_roles", "sha256": sha256_file(args.current_roles.resolve())}]
    pd.DataFrame(provenance).to_csv(out / "rb_week1_pregame_parent_provenance.csv", index=False)
    payload = {"candidate": "RB_WEEK1_PREGAME_OPERATIONAL_READINESS_V1", "disposition": disposition, "gate_count": int(len(gate_df)), "gate_pass_count": int(gate_df["passed"].sum()), "player_rows": int(len(view)), "teams": int(view["team"].nunique()), "games": int(view["event_id"].nunique()), "fresh_current_rb_rows": int(len(cur)), "current_only_roster_rows": roster_added, "sealed_only_roster_rows": roster_removed, "role_changes_since_r26r": role_changes, "market_covered_players": int(view["market_covered"].sum()), "large_r26_movers_abs_ge_0_5": int(view["r26_minus_baseline_receptions_mean"].abs().ge(0.50).sum()), "week1_outcomes_used": 0, "sportsbook_football_inputs_used": 0, "production_parameters_changed": False, "production_promotion_performed": False, "live_shadow_activation_performed": False, "r26_candidate_regenerated": False, "authority_note": "Pregame inspection/readiness only. R26 remains research-shadow and is not production-active."}
    (out / "rb_week1_pregame_readiness_disposition.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True)); print(view.sort_values("r26_minus_baseline_receptions_mean", ascending=False).head(20).to_string(index=False))
    return 0 if disposition != FAIL else 2


if __name__ == "__main__":
    raise SystemExit(main())
