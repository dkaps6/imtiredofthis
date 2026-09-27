#!/usr/bin/env python3
"""Audit non-target Monte Carlo invariance across TE-R5P and WR-R15 stages.

Read-only systems audit for:
docs/research/WEEK3_SPECIALIST_NONTARGET_MC_INVARIANCE_V1_PLAN.md

No simulation is rerun and no production file is mutated.  The script consumes
the already-emitted entitlement traces and stage-to-stage simulation deltas from
one preserved Full Slate artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TOL = 1e-12
UNRELATED = {"pass_yards", "rush_att", "rush_yards"}
RECEIVING_LINKED = {"receptions", "rec_yards", "rush_rec_yards"}
ATD = {"anytime_td"}

REQ_TARGET = {
    "event_id", "team", "player_clean_key", "position",
    "m38_explicit_entitlement_tgt_share", "entitlement_tgt_share",
    "wr_r15_anchor",
}
REQ_TE_TRACE = {
    "event_id", "team", "player_clean_key", "te_r5p_entitlement_tgt_share"
}
REQ_WR_TRACE = {
    "event_id", "team", "player_clean_key", "wr_r15_entitlement_tgt_share"
}
REQ_DELTA = {
    "event_id", "player_clean_key", "position", "market",
    "mean_delta", "abs_mean_delta", "max_element_gap",
}


def _read_csv(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")
    return df


def _read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(20).to_dict("records")
        raise RuntimeError(f"{label} duplicate keys={keys}: {sample}")


def _finite_numeric(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.isna().any() or not np.isfinite(x.to_numpy(float)).all():
            raise RuntimeError(f"{label} contains non-finite {col}")


def build_entitlement_states(
    target: pd.DataFrame,
    te_trace: pd.DataFrame,
) -> pd.DataFrame:
    """Return exact baseline -> TE-only -> final entitlement state per player."""
    keys = ["event_id", "team", "player_clean_key"]
    _assert_unique(target, keys, "target entitlement trace")
    _assert_unique(te_trace, keys, "TE-R5P trace")

    state = target[
        keys + [
            "position",
            "m38_explicit_entitlement_tgt_share",
            "entitlement_tgt_share",
            "wr_r15_anchor",
        ]
    ].copy()
    state = state.rename(columns={
        "m38_explicit_entitlement_tgt_share": "m38_entitlement",
        "entitlement_tgt_share": "final_entitlement",
    })
    state["m38_entitlement"] = pd.to_numeric(state["m38_entitlement"], errors="raise").astype(float)
    state["final_entitlement"] = pd.to_numeric(state["final_entitlement"], errors="raise").astype(float)

    t = te_trace[keys + ["te_r5p_entitlement_tgt_share"]].copy()
    t["te_r5p_entitlement_tgt_share"] = pd.to_numeric(
        t["te_r5p_entitlement_tgt_share"], errors="raise"
    ).astype(float)
    state = state.merge(t, on=keys, how="left", validate="one_to_one")

    # TE-R5P changes only TE rows. All other players retain M38 entitlement.
    state["te_only_entitlement"] = state["te_r5p_entitlement_tgt_share"].combine_first(
        state["m38_entitlement"]
    )
    state["te_entitlement_delta"] = state["te_only_entitlement"] - state["m38_entitlement"]
    state["wr_entitlement_delta"] = state["final_entitlement"] - state["te_only_entitlement"]
    state["te_protected"] = state["te_entitlement_delta"].abs().le(TOL)
    state["wr_protected"] = state["wr_entitlement_delta"].abs().le(TOL)

    if state[["m38_entitlement", "te_only_entitlement", "final_entitlement"]].isna().any().any():
        raise RuntimeError("entitlement state contains missing numeric values")
    return state


def _semantic_class(market: object) -> str:
    m = str(market)
    if m in UNRELATED:
        return "SEMANTICALLY_UNRELATED"
    if m in RECEIVING_LINKED:
        return "RECEIVING_LINKED"
    if m in ATD:
        return "ATD_DESCRIPTIVE"
    return "OTHER"


def attach_stage(
    delta: pd.DataFrame,
    state: pd.DataFrame,
    *,
    stage: str,
) -> pd.DataFrame:
    keys = ["event_id", "player_clean_key"]
    _assert_unique(delta, keys + ["market"], f"{stage} simulation delta")
    s = state[
        [
            "event_id", "team", "player_clean_key", "position",
            "m38_entitlement", "te_only_entitlement", "final_entitlement",
            "te_entitlement_delta", "wr_entitlement_delta",
            "te_protected", "wr_protected", "wr_r15_anchor",
        ]
    ].copy()
    out = delta.merge(
        s,
        on=["event_id", "player_clean_key"],
        how="left",
        validate="many_to_one",
        suffixes=("_delta", "_state"),
    )
    if out["team"].isna().any():
        sample = out.loc[out["team"].isna(), ["event_id", "player_clean_key"]].head(20).to_dict("records")
        raise RuntimeError(f"{stage} delta could not join entitlement state: {sample}")

    if "position_delta" in out.columns and "position_state" in out.columns:
        a = out["position_delta"].fillna("").astype(str).str.upper().str.strip()
        b = out["position_state"].fillna("").astype(str).str.upper().str.strip()
        mismatch = a.ne(b) & a.ne("") & b.ne("")
        if mismatch.any():
            raise RuntimeError(f"{stage} position mismatch after entitlement join")
        out["position"] = b.where(b.ne(""), a)
    elif "position_delta" in out.columns:
        out["position"] = out["position_delta"]
    elif "position_state" in out.columns:
        out["position"] = out["position_state"]

    if stage == "TE_R5P":
        out["entitlement_delta"] = out["te_entitlement_delta"].astype(float)
        out["protected"] = out["te_protected"].astype(bool)
    elif stage == "WR_R15":
        out["entitlement_delta"] = out["wr_entitlement_delta"].astype(float)
        out["protected"] = out["wr_protected"].astype(bool)
    else:
        raise ValueError(stage)

    _finite_numeric(out, ["mean_delta", "abs_mean_delta", "max_element_gap", "entitlement_delta"], stage)
    out["semantic_class"] = out["market"].map(_semantic_class)
    out["mean_drift"] = pd.to_numeric(out["abs_mean_delta"], errors="raise").gt(TOL)
    out["element_drift"] = pd.to_numeric(out["max_element_gap"], errors="raise").gt(TOL)
    out["stage"] = stage
    return out


def _q(x: pd.Series, q: float) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(q)) if len(x) else float("nan")


def summarize_group(
    frame: pd.DataFrame,
    *,
    stage: str,
    group_type: str,
    group_value: str,
) -> dict:
    g = frame.copy()
    n = int(len(g))
    absd = pd.to_numeric(g["abs_mean_delta"], errors="coerce") if n else pd.Series(dtype=float)
    signed = pd.to_numeric(g["mean_delta"], errors="coerce") if n else pd.Series(dtype=float)
    elem = pd.to_numeric(g["max_element_gap"], errors="coerce") if n else pd.Series(dtype=float)
    return {
        "stage": stage,
        "group_type": group_type,
        "group_value": group_value,
        "protected_players": int(g[["event_id", "player_clean_key"]].drop_duplicates().shape[0]) if n else 0,
        "simulation_keys": n,
        "mean_drift_keys": int(g["mean_drift"].sum()) if n else 0,
        "mean_drift_rate": float(g["mean_drift"].mean()) if n else float("nan"),
        "element_drift_keys": int(g["element_drift"].sum()) if n else 0,
        "element_drift_rate": float(g["element_drift"].mean()) if n else float("nan"),
        "signed_mean_delta": float(signed.mean()) if n else float("nan"),
        "mean_abs_mean_delta": float(absd.mean()) if n else float("nan"),
        "median_abs_mean_delta": float(absd.median()) if n else float("nan"),
        "p90_abs_mean_delta": _q(absd, .90),
        "p95_abs_mean_delta": _q(absd, .95),
        "p99_abs_mean_delta": _q(absd, .99),
        "max_abs_mean_delta": float(absd.max()) if n else float("nan"),
        "max_element_gap": float(elem.max()) if n else float("nan"),
    }


def summarize(stage_frame: pd.DataFrame) -> pd.DataFrame:
    stage = str(stage_frame["stage"].iloc[0])
    p = stage_frame.loc[stage_frame["protected"]].copy()
    rows = [
        summarize_group(p, stage=stage, group_type="protected_scope", group_value="ALL"),
    ]
    for cls in ("SEMANTICALLY_UNRELATED", "RECEIVING_LINKED", "ATD_DESCRIPTIVE", "OTHER"):
        rows.append(
            summarize_group(
                p.loc[p["semantic_class"].eq(cls)],
                stage=stage,
                group_type="semantic_class",
                group_value=cls,
            )
        )
    for market in sorted(p["market"].astype(str).unique()):
        rows.append(
            summarize_group(
                p.loc[p["market"].astype(str).eq(market)],
                stage=stage,
                group_type="market",
                group_value=market,
            )
        )
    for pos in sorted(p["position"].fillna("").astype(str).unique()):
        rows.append(
            summarize_group(
                p.loc[p["position"].fillna("").astype(str).eq(pos)],
                stage=stage,
                group_type="position",
                group_value=pos or "BLANK",
            )
        )
    changed = stage_frame.loc[~stage_frame["protected"]].copy()
    rows.append(
        summarize_group(
            changed,
            stage=stage,
            group_type="intentional_specialist_scope",
            group_value="ENTITLEMENT_CHANGED",
        )
    )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("data/research/week3_specialist_nontarget_mc_invariance_v1"))
    p.add_argument("--source-run-id", default="36293274478")
    p.add_argument("--source-artifact-id", default="10923570170")
    p.add_argument(
        "--source-artifact-digest",
        default="sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480",
    )
    a = p.parse_args()
    root = a.root
    data = root / "data"

    target = _read_csv(data / "target_entitlement_v1_trace.csv", REQ_TARGET, "target entitlement trace")
    te_trace = _read_csv(data / "te_r5p_full_slate_entitlement_trace.csv", REQ_TE_TRACE, "TE-R5P trace")
    wr_trace = _read_csv(data / "wr_r15_full_slate_entitlement_trace.csv", REQ_WR_TRACE, "WR-R15 trace")
    te_delta = _read_csv(data / "te_r5p_full_slate_simulation_delta.csv", REQ_DELTA, "TE simulation delta")
    wr_delta = _read_csv(data / "wr_r15_full_slate_simulation_delta.csv", REQ_DELTA, "WR simulation delta")

    ent_audit = _read_json(data / "target_entitlement_v1_audit.json", "target entitlement audit")
    te_audit = _read_json(data / "te_r5p_full_slate_entitlement_audit.json", "TE-R5P audit")
    wr_audit = _read_json(data / "wr_r15_full_slate_entitlement_audit.json", "WR-R15 audit")

    integrity: dict[str, bool] = {}
    integrity["target_trace_nonempty"] = bool(len(target))
    integrity["te_trace_nonempty"] = bool(len(te_trace))
    integrity["wr_trace_nonempty"] = bool(len(wr_trace))

    state = build_entitlement_states(target, te_trace)

    # The WR specialist trace is an independent source of the final WR2+ values.
    keys = ["event_id", "team", "player_clean_key"]
    _assert_unique(wr_trace, keys, "WR-R15 trace")
    wr_chk = wr_trace[keys + ["wr_r15_entitlement_tgt_share"]].merge(
        state[keys + ["final_entitlement"]], on=keys, how="left", validate="one_to_one"
    )
    if wr_chk["final_entitlement"].isna().any():
        raise RuntimeError("WR trace could not join final entitlement state")
    wr_gap = (
        pd.to_numeric(wr_chk["wr_r15_entitlement_tgt_share"], errors="raise")
        - pd.to_numeric(wr_chk["final_entitlement"], errors="raise")
    ).abs()
    integrity["wr_trace_matches_final_entitlement"] = bool(float(wr_gap.max()) <= TOL)

    te_keys = set(map(tuple, te_delta[["event_id", "player_clean_key", "market"]].astype(str).to_numpy()))
    wr_keys = set(map(tuple, wr_delta[["event_id", "player_clean_key", "market"]].astype(str).to_numpy()))
    integrity["simulation_delta_key_universe_equal"] = te_keys == wr_keys

    integrity["te_audit_ready"] = te_audit.get("disposition") == "TE_R5P_FULL_SLATE_ENTITLEMENT_READY"
    integrity["te_non_te_entitlement_preserved"] = te_audit.get("non_te_entitlement_preserved") is True
    integrity["te_team_pool_preserved"] = te_audit.get("team_te_pool_preserved") is True
    integrity["te_team_total_preserved"] = te_audit.get("team_total_player_entitlement_preserved") is True
    integrity["te_sportsbook_inputs_zero"] = te_audit.get("sportsbook_inputs_used") is False
    integrity["te_future_outcomes_zero"] = te_audit.get("current_or_future_outcomes_used") is False

    integrity["wr_audit_ready"] = wr_audit.get("disposition") == "WR_R15_FULL_SLATE_ENTITLEMENT_READY"
    integrity["wr_non_wr_entitlement_preserved"] = wr_audit.get("non_wr_entitlement_preserved") is True
    integrity["wr_wr1_anchor_preserved"] = wr_audit.get("m38_wr1_anchor_preserved") is True
    integrity["wr_wr2plus_pool_preserved"] = wr_audit.get("wr2plus_pool_preserved") is True
    integrity["wr_room_mass_preserved"] = wr_audit.get("wr_room_mass_preserved") is True
    integrity["wr_team_total_preserved"] = wr_audit.get("team_total_player_entitlement_preserved") is True
    integrity["wr_sportsbook_inputs_zero"] = wr_audit.get("sportsbook_inputs_used") is False
    integrity["wr_future_outcomes_zero"] = wr_audit.get("current_or_future_outcomes_used") is False

    integrity["entitlement_audit_sportsbook_inputs_zero"] = ent_audit.get("sportsbook_inputs_used") is False
    integrity["entitlement_audit_te_non_te_preserved"] = ent_audit.get("te_r5p_non_te_entitlement_preserved") is True
    integrity["entitlement_audit_wr_non_wr_preserved"] = ent_audit.get("wr_r15_non_wr_entitlement_preserved") is True

    # Re-prove the protected position contracts numerically from the actual trace.
    pos = state["position"].fillna("").astype(str).str.upper().str.strip()
    non_te = ~pos.eq("TE")
    non_wr = ~pos.eq("WR")
    wr_anchor = state["wr_r15_anchor"].fillna(False).astype(bool)
    integrity["numeric_te_non_te_delta_zero"] = bool(
        state.loc[non_te, "te_entitlement_delta"].abs().le(TOL).all()
    )
    integrity["numeric_wr_non_wr_delta_zero"] = bool(
        state.loc[non_wr, "wr_entitlement_delta"].abs().le(TOL).all()
    )
    integrity["numeric_wr_anchor_delta_zero"] = bool(
        state.loc[wr_anchor, "wr_entitlement_delta"].abs().le(TOL).all()
    )

    if not all(integrity.values()):
        disposition = "SPECIALIST_NONTARGET_MC_INVARIANCE_INTEGRITY_FAILURE"
        payload = {
            "version": "WEEK3_SPECIALIST_NONTARGET_MC_INVARIANCE_V1",
            "disposition": disposition,
            "source_run_id": str(a.source_run_id),
            "source_artifact_id": str(a.source_artifact_id),
            "source_artifact_digest": str(a.source_artifact_digest),
            "integrity": integrity,
            "production_changed": False,
            "sportsbook_inputs_used_to_define_protection": False,
            "week3_outcomes_used": False,
        }
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"{disposition}: {[k for k,v in integrity.items() if not v]}")

    te = attach_stage(te_delta, state, stage="TE_R5P")
    wr = attach_stage(wr_delta, state, stage="WR_R15")
    detail = pd.concat([te, wr], ignore_index=True)

    summary = pd.concat([summarize(te), summarize(wr)], ignore_index=True)

    protected_unrelated = detail.loc[
        detail["protected"] & detail["semantic_class"].eq("SEMANTICALLY_UNRELATED")
    ].copy()
    drift = protected_unrelated["mean_drift"] | protected_unrelated["element_drift"]
    disposition = (
        "SPECIALIST_NONTARGET_MC_PATH_DRIFT_CONFIRMED"
        if bool(drift.any())
        else "SPECIALIST_NONTARGET_MC_PATH_INVARIANCE_HOLDS"
    )

    tops = (
        detail.loc[detail["protected"]]
        .sort_values(["stage", "abs_mean_delta"], ascending=[True, False])
        .groupby("stage", sort=False, group_keys=False)
        .head(25)
        .reset_index(drop=True)
    )

    stage_payload = {}
    for stage, g in detail.groupby("stage", sort=True):
        pscope = g.loc[g["protected"]]
        punrel = pscope.loc[pscope["semantic_class"].eq("SEMANTICALLY_UNRELATED")]
        stage_payload[stage] = {
            "protected_players": int(pscope[["event_id", "player_clean_key"]].drop_duplicates().shape[0]),
            "protected_keys": int(len(pscope)),
            "protected_mean_drift_keys": int(pscope["mean_drift"].sum()),
            "protected_element_drift_keys": int(pscope["element_drift"].sum()),
            "protected_unrelated_keys": int(len(punrel)),
            "protected_unrelated_mean_drift_keys": int(punrel["mean_drift"].sum()),
            "protected_unrelated_element_drift_keys": int(punrel["element_drift"].sum()),
            "protected_unrelated_max_abs_mean_delta": float(punrel["abs_mean_delta"].max()) if len(punrel) else 0.0,
            "protected_unrelated_max_element_gap": float(punrel["max_element_gap"].max()) if len(punrel) else 0.0,
            "intentional_entitlement_changed_players": int(
                g.loc[~g["protected"], ["event_id", "player_clean_key"]].drop_duplicates().shape[0]
            ),
        }

    payload = {
        "version": "WEEK3_SPECIALIST_NONTARGET_MC_INVARIANCE_V1",
        "disposition": disposition,
        "source_run_id": str(a.source_run_id),
        "source_artifact_id": str(a.source_artifact_id),
        "source_artifact_digest": str(a.source_artifact_digest),
        "tolerance": TOL,
        "primary_semantically_unrelated_markets": sorted(UNRELATED),
        "receiving_linked_markets": sorted(RECEIVING_LINKED),
        "integrity": integrity,
        "stages": stage_payload,
        "production_changed": False,
        "sportsbook_inputs_used_to_define_protection": False,
        "week3_outcomes_used": False,
        "repair_authorized": False,
        "next_if_confirmed": "SEPARATELY_FROZEN_DOWNSTREAM_PROBABILITY_AND_BOARD_MATERIALITY_AUDIT",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "detail.csv", index=False)
    summary.to_csv(a.out_dir / "summary.csv", index=False)
    tops.to_csv(a.out_dir / "top_protected_drifts.csv", index=False)
    (a.out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
