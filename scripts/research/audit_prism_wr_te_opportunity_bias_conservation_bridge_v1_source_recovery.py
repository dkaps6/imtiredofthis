#!/usr/bin/env python3
"""Recover and audit source topology for PRISM WR/TE opportunity-bias bridge V1.

Research-only. No bridge intervention or outcome improvement is scored.

WR authority is an exact deterministic rerun of the original WR-R15 source
commit and must match the historical predictions CSV SHA-256 byte-for-byte.
TE authority comes from the later preserved opportunity-authority artifact
which explicitly certifies TE-R5P was applied for 2024-2025.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_RECOVERY_AUDIT"
FREEZE_COMMIT = "d0fd57c3f10de74cbdd8c5dc7e027464d9e67d6c"

WR_SOURCE_COMMIT = "02c3dd1a681d4ab2953683039e39830554f9ec9f"
WR_EXPECTED_PREDICTIONS_SHA256 = "91d108ee7981935d78179257b204a0204e9a4f8061e6eba4d6810845ed057d05"
WR_ORIGINAL_RUN = 34238301577
WR_ORIGINAL_ARTIFACT = 10061328722
WR_TOPOLOGY_RUN = 36203881629
WR_TOPOLOGY_ARTIFACT = 10893281689
WR_TOPOLOGY_DIGEST = "sha256:33b12fd521196838054f4ee4240d8cea8f66c2fce5b5bfc9eb902bc67d730adb"

PRESERVED_RUN = 36332720457
PRESERVED_ARTIFACT = 10935999653
PRESERVED_DIGEST = "sha256:4fde0a72c3aef7ae2412580333a7fefcd3078e53a31f0e9ebb945fd6a6ed1bb5"

SIGNAL_RUN = 39109423266
SIGNAL_ARTIFACT = 12128983909
SIGNAL_DIGEST = "sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490"

SUPPORT = {
    ("WR", 2023): {"rows": 250, "players": 50},
    ("WR", 2024): {"rows": 250, "players": 50},
    ("TE", 2024): {"rows": 250, "players": 50},
    ("TE", 2025): {"rows": 250, "players": 50},
}
SEASONS = {"WR": (2023, 2024), "TE": (2024, 2025)}
TOL = 1e-9


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty JSON: {path}")
    return json.loads(path.read_text())


def canon_identity(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    for c in ("team", "player_clean_key"):
        if c in x:
            x[c] = x[c].astype("string").fillna("").str.strip()
    for c in ("season", "week"):
        if c in x:
            x[c] = num(x[c]).astype("Int64")
    return x


def load_signal(path: Path, result_path: Path) -> pd.DataFrame:
    result = load_json(result_path)
    if result.get("version") != "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL":
        raise RuntimeError("wrong signed-bias signal result version")
    expected = {
        "WR_OPPORTUNITY": "WR_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED",
        "TE_OPPORTUNITY": "TE_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED",
    }
    for cell, disp in expected.items():
        if result.get("cell_dispositions", {}).get(cell) != disp:
            raise RuntimeError(f"{cell} is not qualified in signal authority")
    for cell in ("WR_EFFICIENCY", "TE_EFFICIENCY"):
        if not str(result.get("cell_dispositions", {}).get(cell, "")).endswith("_CLOSED"):
            raise RuntimeError(f"{cell} unexpectedly not closed")

    x = canon_identity(pd.read_csv(path, low_memory=False))
    required = {
        "position", "component", "season", "week", "team", "player_clean_key",
        "source_eligible", "hierarchical_signed_bias",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"signal rows missing: {sorted(missing)}")
    x["position"] = x["position"].astype("string").str.upper().str.strip()
    x["component"] = x["component"].astype("string").str.upper().str.strip()
    if x["source_eligible"].dtype != bool:
        x["source_eligible"] = (
            x["source_eligible"].astype("string").str.lower().str.strip()
            .map({"true": True, "false": False}).fillna(False).astype(bool)
        )
    x["hierarchical_signed_bias"] = num(x["hierarchical_signed_bias"])
    x = x.loc[
        x["component"].eq("OPPORTUNITY")
        & x["position"].isin(["WR", "TE"])
        & x["source_eligible"].eq(True)
    ].copy()
    keys = ["position", "season", "week", "team", "player_clean_key"]
    if x.empty or x.duplicated(keys).any():
        raise RuntimeError("signal opportunity identity invalid")
    if not np.isfinite(x["hierarchical_signed_bias"]).all():
        raise RuntimeError("signal bias contains nonfinite values")
    return x


def load_wr_predictions(path: Path) -> tuple[pd.DataFrame, dict]:
    got_hash = sha256(path)
    if got_hash != WR_EXPECTED_PREDICTIONS_SHA256:
        raise RuntimeError(
            f"WR rebuilt predictions hash mismatch expected={WR_EXPECTED_PREDICTIONS_SHA256} got={got_hash}"
        )
    x = canon_identity(pd.read_csv(path, low_memory=False))
    required = {
        "variant", "season", "week", "team", "player_clean_key",
        "pred_targets", "mc_rec_yards",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"WR predictions missing: {sorted(missing)}")
    x["variant"] = x["variant"].astype("string").fillna("").str.strip()
    x = x.loc[
        x["variant"].eq("WR_R15_WR1_ANCHORED_PARTICIPATION")
        & x["season"].isin(SEASONS["WR"])
    ].copy()
    x["pred_targets"] = num(x["pred_targets"])
    x["pred_rec_yards"] = num(x["mc_rec_yards"])
    keys = ["season", "week", "team", "player_clean_key"]
    if x.empty or x.duplicated(keys).any():
        raise RuntimeError("WR candidate authority identity invalid")
    if not np.isfinite(x[["pred_targets", "pred_rec_yards"]].to_numpy(float)).all():
        raise RuntimeError("WR candidate authority nonfinite projection")
    return x, {"rebuilt_predictions_sha256": got_hash, "candidate_rows": int(len(x))}


def load_wr_topology(detail_path: Path, audit_path: Path) -> tuple[pd.DataFrame, dict]:
    audit = load_json(audit_path)
    if audit.get("version") != "WR_ANCHOR_ROLE_TRANSMISSION_AUDIT_V1":
        raise RuntimeError("wrong WR topology audit version")
    if int(audit.get("source_audit", {}).get("frozen_full_wr_rows", -1)) != 6409:
        raise RuntimeError("WR topology full-row count drift")
    # source_audit.json stores these mechanical certifications at top level.
    if float(audit.get("max_frozen_wr_room_mass_gap", math.inf)) > 1e-12:
        raise RuntimeError("WR topology room-mass conservation failed")
    if float(audit.get("max_frozen_anchor_entitlement_delta", math.inf)) > 1e-12:
        raise RuntimeError("WR topology anchor conservation failed")
    if int(audit.get("rank1_anchor_identity_mismatches", -1)) != 0:
        raise RuntimeError("WR topology anchor identity mismatch")
    if float(audit.get("secondary_bridge_coverage", 0.0)) != 1.0:
        raise RuntimeError("WR topology secondary bridge coverage drift")
    if int(audit.get("strict_prior_future_violations", -1)) != 0:
        raise RuntimeError("WR topology strict-prior violation")

    x = canon_identity(pd.read_csv(detail_path, low_memory=False))
    required = {
        "season", "week", "team", "player_clean_key",
        "candidate_entitlement_tgt_share", "authority_row_type",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"WR topology missing: {sorted(missing)}")
    x = x.loc[x["season"].isin(SEASONS["WR"])].copy()
    x["candidate_entitlement_tgt_share"] = num(x["candidate_entitlement_tgt_share"])
    keys = ["season", "week", "team", "player_clean_key"]
    if x.empty or x.duplicated(keys).any():
        raise RuntimeError("WR full-room topology identity invalid")
    if not np.isfinite(x["candidate_entitlement_tgt_share"]).all():
        raise RuntimeError("WR full-room topology contains nonfinite entitlement")
    x["room_size"] = x.groupby(["season", "week", "team"], sort=False)["player_clean_key"].transform("size")
    x["room_entitlement_mass"] = x.groupby(["season", "week", "team"], sort=False)["candidate_entitlement_tgt_share"].transform("sum")
    return x, audit


def load_preserved_opportunity(
    direct_path: Path, market_path: Path, result_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result = load_json(result_path)
    if result.get("version") != "OPPORTUNITY_AUTHORITY_PRIORITY_V1":
        raise RuntimeError("wrong preserved opportunity authority version")
    integ = result.get("integrity", {})
    gates = result.get("gates", {})
    integrity_required_true = (
        "te_r5p_2024_2025_applied",
        "wr_r15_2024_applied_all_weeks",
        "specialist_assets_invariant",
    )
    gate_required_true = (
        "source_priority_boundary_exact",
        "specialist_assets_invariant",
        "team_volume_inputs_invariant",
        "zero_sportsbook_inputs",
        "zero_target_future_feature_rows",
    )
    bad_integrity = [k for k in integrity_required_true if integ.get(k) is not True]
    bad_gates = [k for k in gate_required_true if gates.get(k) is not True]
    if bad_integrity or bad_gates:
        raise RuntimeError(
            "preserved opportunity authority integrity failed: "
            f"integrity={bad_integrity} gates={bad_gates}"
        )
    if int(integ.get("sportsbook_inputs_used", -1)) != 0:
        raise RuntimeError("preserved opportunity authority used sportsbook inputs")
    if int(integ.get("target_future_feature_rows", -1)) != 0:
        raise RuntimeError("preserved opportunity authority used target/future features")

    d = canon_identity(pd.read_csv(direct_path, low_memory=False))
    m = canon_identity(pd.read_csv(market_path, low_memory=False))
    for req, name in (
        ({"family", "season", "week", "team", "player_clean_key", "baseline_pred"}, "direct"),
        ({"position", "market", "season", "week", "team", "player_clean_key", "baseline_proj"}, "market"),
    ):
        missing = req - (set(d.columns) if name == "direct" else set(m.columns))
        if missing:
            raise RuntimeError(f"preserved {name} missing: {sorted(missing)}")
    d["family"] = d["family"].astype("string").fillna("").str.strip().str.upper()
    d["baseline_pred"] = num(d["baseline_pred"])
    m["position"] = m["position"].astype("string").fillna("").str.strip().str.upper()
    m["market"] = m["market"].astype("string").fillna("").str.strip().str.lower()
    m["baseline_proj"] = num(m["baseline_proj"])
    return d, m, result


def quantiles(s: pd.Series) -> dict:
    v = num(s).dropna()
    if v.empty:
        return {k: math.nan for k in ("min", "p01", "p10", "median", "p90", "p99", "max")}
    return {
        "min": float(v.min()),
        "p01": float(v.quantile(.01)),
        "p10": float(v.quantile(.10)),
        "median": float(v.quantile(.50)),
        "p90": float(v.quantile(.90)),
        "p99": float(v.quantile(.99)),
        "max": float(v.max()),
    }


def audit_wr(
    season: int,
    signal: pd.DataFrame,
    pred: pd.DataFrame,
    topo: pd.DataFrame,
    direct: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    keys = ["season", "week", "team", "player_clean_key"]
    s = signal.loc[(signal["position"].eq("WR")) & signal["season"].eq(season)].copy()
    p = pred.loc[pred["season"].eq(season), keys + ["pred_targets", "pred_rec_yards"]].copy()
    t = topo.loc[
        topo["season"].eq(season),
        keys + ["candidate_entitlement_tgt_share", "room_size", "room_entitlement_mass", "authority_row_type"],
    ].copy()

    z = (
        s.merge(p, on=keys, how="left", validate="one_to_one", indicator="pred_join")
         .merge(t, on=keys, how="left", validate="one_to_one", indicator="topology_join")
    )
    z["pred_join_ok"] = z["pred_join"].eq("both")
    z["topology_join_ok"] = z["topology_join"].eq("both")
    z["pred_ypt"] = z["pred_rec_yards"] / z["pred_targets"]
    z["bridge_topology_eligible"] = (
        z["pred_join_ok"] & z["topology_join_ok"]
        & z["pred_targets"].gt(0) & z["pred_rec_yards"].gt(0)
        & np.isfinite(z["pred_ypt"]) & z["pred_ypt"].gt(0)
        & z["room_size"].ge(2)
    )
    e = z.loc[z["bridge_topology_eligible"]].copy()
    e["bias_factor"] = np.exp(-e["hierarchical_signed_bias"] / e["pred_rec_yards"])

    parity_max_gap = math.nan
    parity_rows = 0
    if season == 2024:
        wd = direct.loc[
            direct["family"].eq("WR_TARGET_OPPORTUNITY") & direct["season"].eq(2024),
            keys + ["baseline_pred"],
        ].copy()
        q = e.merge(wd, on=keys, how="left", validate="one_to_one")
        if q["baseline_pred"].isna().any():
            parity_max_gap = math.inf
            parity_rows = int(q["baseline_pred"].notna().sum())
        else:
            parity_rows = int(len(q))
            parity_max_gap = float((q["pred_targets"] - q["baseline_pred"]).abs().max()) if len(q) else math.inf

    req = SUPPORT[("WR", season)]
    gates = {
        "eligible_rows": int(len(e)) >= req["rows"],
        "eligible_players": int(e["player_clean_key"].nunique()) >= req["players"],
        "prediction_join_100pct": bool(len(z) and z["pred_join_ok"].all()),
        "topology_join_100pct": bool(len(z) and z["topology_join_ok"].all()),
        "positive_pred_targets": bool(len(z) and z["pred_targets"].gt(0).all()),
        "positive_pred_ypt": bool(len(z) and np.isfinite(z["pred_ypt"]).all() and z["pred_ypt"].gt(0).all()),
        "multi_player_room": bool(len(e) == len(z)),
    }
    if season == 2024:
        gates["preserved_wr2024_target_parity"] = bool(
            parity_rows == len(e) and np.isfinite(parity_max_gap) and parity_max_gap <= TOL
        )

    room_sizes = topo.loc[topo["season"].eq(season)].drop_duplicates(["season", "week", "team"])["room_size"]
    rec = {
        "position": "WR",
        "season": season,
        "signal_rows": int(len(z)),
        "eligible_rows": int(len(e)),
        "eligible_players": int(e["player_clean_key"].nunique()),
        "full_topology_rows": int((topo["season"].eq(season)).sum()),
        "full_rooms": int(topo.loc[topo["season"].eq(season), ["week", "team"]].drop_duplicates().shape[0]),
        "room_size_p10": float(room_sizes.quantile(.10)),
        "room_size_median": float(room_sizes.quantile(.50)),
        "room_size_p90": float(room_sizes.quantile(.90)),
        "room_size_max": float(room_sizes.max()),
        "wr2024_target_parity_rows": parity_rows,
        "wr2024_target_parity_max_gap": parity_max_gap,
        "factor_quantiles": quantiles(e["bias_factor"]),
        "source_gates": gates,
        "source_ready": bool(all(gates.values())),
    }
    e["position"] = "WR"
    return rec, e


def audit_te(
    season: int,
    signal: pd.DataFrame,
    direct: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    keys = ["season", "week", "team", "player_clean_key"]
    s = signal.loc[(signal["position"].eq("TE")) & signal["season"].eq(season)].copy()
    d = direct.loc[
        direct["family"].eq("TE_TARGET_OPPORTUNITY") & direct["season"].eq(season),
        keys + ["baseline_pred"],
    ].copy()
    m = market.loc[
        market["position"].eq("TE") & market["market"].eq("rec_yards") & market["season"].eq(season),
        keys + ["baseline_proj"],
    ].copy()
    if d.duplicated(keys).any() or m.duplicated(keys).any():
        raise RuntimeError(f"TE {season} preserved authority duplicate identity")
    full = d.merge(m, on=keys, how="inner", validate="one_to_one")
    if len(full) != len(d) or len(full) != len(m):
        raise RuntimeError(f"TE {season} target/yard authority universe mismatch")
    full = full.rename(columns={"baseline_pred": "pred_targets", "baseline_proj": "pred_rec_yards"})
    full["room_size"] = full.groupby(["season", "week", "team"], sort=False)["player_clean_key"].transform("size")
    full["room_target_sum"] = full.groupby(["season", "week", "team"], sort=False)["pred_targets"].transform("sum")

    z = s.merge(
        full[keys + ["pred_targets", "pred_rec_yards", "room_size", "room_target_sum"]],
        on=keys, how="left", validate="one_to_one", indicator=True
    )
    z["join_ok"] = z["_merge"].eq("both")
    z["pred_ypt"] = z["pred_rec_yards"] / z["pred_targets"]
    z["bridge_topology_eligible"] = (
        z["join_ok"] & z["pred_targets"].gt(0) & z["pred_rec_yards"].gt(0)
        & np.isfinite(z["pred_ypt"]) & z["pred_ypt"].gt(0)
        & z["room_size"].ge(2)
    )
    e = z.loc[z["bridge_topology_eligible"]].copy()
    e["bias_factor"] = np.exp(-e["hierarchical_signed_bias"] / e["pred_rec_yards"])
    req = SUPPORT[("TE", season)]
    gates = {
        "eligible_rows": int(len(e)) >= req["rows"],
        "eligible_players": int(e["player_clean_key"].nunique()) >= req["players"],
        "join_100pct": bool(len(z) and z["join_ok"].all()),
        "positive_pred_targets": bool(len(z) and z["pred_targets"].gt(0).all()),
        "positive_pred_ypt": bool(len(z) and np.isfinite(z["pred_ypt"]).all() and z["pred_ypt"].gt(0).all()),
    }
    room_sizes = full.drop_duplicates(["season", "week", "team"])["room_size"]
    rec = {
        "position": "TE",
        "season": season,
        "signal_rows": int(len(z)),
        "eligible_rows": int(len(e)),
        "eligible_players": int(e["player_clean_key"].nunique()),
        "single_player_signal_rows": int((z["room_size"] < 2).sum()),
        "full_authority_rows": int(len(full)),
        "full_rooms": int(full[["week", "team"]].drop_duplicates().shape[0]),
        "room_size_p10": float(room_sizes.quantile(.10)),
        "room_size_median": float(room_sizes.quantile(.50)),
        "room_size_p90": float(room_sizes.quantile(.90)),
        "room_size_max": float(room_sizes.max()),
        "factor_quantiles": quantiles(e["bias_factor"]),
        "source_gates": gates,
        "source_ready": bool(all(gates.values())),
    }
    e["position"] = "TE"
    return rec, e, full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-predictions", required=True, type=Path)
    ap.add_argument("--wr-topology-detail", required=True, type=Path)
    ap.add_argument("--wr-topology-audit", required=True, type=Path)
    ap.add_argument("--signal-rows", required=True, type=Path)
    ap.add_argument("--signal-result", required=True, type=Path)
    ap.add_argument("--opportunity-direct", required=True, type=Path)
    ap.add_argument("--opportunity-market", required=True, type=Path)
    ap.add_argument("--opportunity-result", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        signal = load_signal(args.signal_rows, args.signal_result)
        wr_pred, wr_recovery = load_wr_predictions(args.wr_predictions)
        wr_topo, wr_topo_audit = load_wr_topology(args.wr_topology_detail, args.wr_topology_audit)
        direct, market, opp_result = load_preserved_opportunity(
            args.opportunity_direct, args.opportunity_market, args.opportunity_result
        )

        summaries: dict[str, dict] = {"WR": {}, "TE": {}}
        supported_parts = []
        te_full_parts = []
        overall = True

        for season in SEASONS["WR"]:
            rec, e = audit_wr(season, signal, wr_pred, wr_topo, direct)
            summaries["WR"][str(season)] = rec
            supported_parts.append(e)
            overall = overall and rec["source_ready"]

        for season in SEASONS["TE"]:
            rec, e, full = audit_te(season, signal, direct, market)
            summaries["TE"][str(season)] = rec
            supported_parts.append(e)
            te_full_parts.append(full)
            overall = overall and rec["source_ready"]

        supported = pd.concat(supported_parts, ignore_index=True, sort=False)
        supported.to_csv(
            args.out_dir / "wr_te_opportunity_bias_bridge_supported_signal_rows.csv",
            index=False,
        )
        wr_topo.to_csv(
            args.out_dir / "wr_r15_full_room_topology_recovered.csv", index=False
        )
        pd.concat(te_full_parts, ignore_index=True, sort=False).to_csv(
            args.out_dir / "te_r5p_full_room_projection_topology_recovered.csv", index=False
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "wr_source_commit": WR_SOURCE_COMMIT,
            "wr_expected_predictions_sha256": WR_EXPECTED_PREDICTIONS_SHA256,
            "wr_recovery": wr_recovery,
            "wr_original_run": WR_ORIGINAL_RUN,
            "wr_original_artifact": WR_ORIGINAL_ARTIFACT,
            "wr_topology_run": WR_TOPOLOGY_RUN,
            "wr_topology_artifact": WR_TOPOLOGY_ARTIFACT,
            "wr_topology_digest": WR_TOPOLOGY_DIGEST,
            "preserved_opportunity_run": PRESERVED_RUN,
            "preserved_opportunity_artifact": PRESERVED_ARTIFACT,
            "preserved_opportunity_digest": PRESERVED_DIGEST,
            "signal_run": SIGNAL_RUN,
            "signal_artifact": SIGNAL_ARTIFACT,
            "signal_digest": SIGNAL_DIGEST,
            "summaries": summaries,
            "disposition": (
                "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERED_READY_ALL"
                if overall else
                "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERY_BLOCKED"
            ),
            "bridge_outcome_rows_scored": 0,
            "target_corrections_applied": 0,
            "room_renormalizations_applied": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERY_INTEGRITY_FAILURE",
            "error": str(exc),
            "bridge_outcome_rows_scored": 0,
            "target_corrections_applied": 0,
            "room_renormalizations_applied": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        out = args.out_dir / "wr_te_opportunity_bias_bridge_source_recovery_result.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    out = args.out_dir / "wr_te_opportunity_bias_bridge_source_recovery_result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
