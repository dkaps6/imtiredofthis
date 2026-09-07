#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def n(v) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a / b)


def slice_summary(name: str, g: pd.DataFrame) -> dict:
    if g.empty:
        return {
            "slice": name,
            "rows": 0,
            "carry_mae": np.nan,
            "carry_bias_actual_minus_pred": np.nan,
            "room_volume_component_abs": np.nan,
            "individual_allocation_component_abs": np.nan,
            "allocation_over_volume_ratio": np.nan,
            "volume_over_allocation_ratio": np.nan,
        }
    v = float(g["room_volume_component"].abs().mean())
    a = float(g["individual_allocation_component"].abs().mean())
    carry_resid = g["actual_att"] - g["pred_att"]
    return {
        "slice": name,
        "rows": int(len(g)),
        "carry_mae": float(carry_resid.abs().mean()),
        "carry_bias_actual_minus_pred": float(carry_resid.mean()),
        "room_volume_component_abs": v,
        "individual_allocation_component_abs": a,
        "allocation_over_volume_ratio": safe_ratio(a, v),
        "volume_over_allocation_ratio": safe_ratio(v, a),
    }


def classify(volume_abs: float, allocation_abs: float) -> str:
    if volume_abs >= 1.25 * allocation_abs:
        return "ROOM_VOLUME"
    if allocation_abs >= 1.25 * volume_abs:
        return "INDIVIDUAL_ALLOCATION"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    casebook = pd.read_csv(one(a.rb_root, "rb_mechanism_casebook.csv"), low_memory=False)
    profiles = pd.read_csv(one(a.rb_root, "rb_individual_mechanisms.csv"), low_memory=False)
    source_result = json.loads(one(a.rb_root, "rb_mechanism_result.json").read_text(encoding="utf-8"))
    casebook.columns = [str(c).strip().lower() for c in casebook.columns]
    profiles.columns = [str(c).strip().lower() for c in profiles.columns]

    expected_source = int(source_result.get("source_rows", -1))
    expected_scoreable = int(source_result.get("scoreable_rows", -1))
    expected_qualifying = int(source_result.get("qualifying_players", -1))
    if expected_source != 1393:
        raise RuntimeError(f"source row drift expected 1393 got {expected_source}")
    if len(casebook) != expected_scoreable:
        raise RuntimeError(f"scoreable casebook drift expected={expected_scoreable} got={len(casebook)}")
    if len(profiles) != expected_qualifying or expected_qualifying != 86:
        raise RuntimeError(f"qualifying profile drift expected=86 source={expected_qualifying} file={len(profiles)}")

    required_case = {"season", "week", "team", "player_key", "player", "pred_att", "actual_att"}
    missing = required_case - set(casebook.columns)
    if missing:
        raise RuntimeError(f"casebook missing columns {sorted(missing)}")
    required_prof = {"player_key", "games", "dominant_mechanism"}
    missing_prof = required_prof - set(profiles.columns)
    if missing_prof:
        raise RuntimeError(f"profiles missing columns {sorted(missing_prof)}")

    x = casebook.copy()
    x["pred_att"] = n(x["pred_att"])
    x["actual_att"] = n(x["actual_att"])
    if x[["pred_att", "actual_att"]].isna().any().any():
        raise RuntimeError("missing carry values in frozen scoreable casebook")

    room_keys = ["season", "week", "team"]
    x["pred_room_att"] = x.groupby(room_keys)["pred_att"].transform("sum")
    x["actual_room_att"] = x.groupby(room_keys)["actual_att"].transform("sum")
    x["pred_room_share"] = np.where(x["pred_room_att"].gt(0), x["pred_att"] / x["pred_room_att"], 0.0)
    x["actual_room_share"] = np.where(x["actual_room_att"].gt(0), x["actual_att"] / x["actual_room_att"], 0.0)

    dv = x["actual_room_att"] - x["pred_room_att"]
    ds = x["actual_room_share"] - x["pred_room_share"]
    x["room_volume_component"] = dv * (x["pred_room_share"] + x["actual_room_share"]) / 2.0
    x["individual_allocation_component"] = ds * (x["pred_room_att"] + x["actual_room_att"]) / 2.0
    x["carry_residual_actual_minus_pred"] = x["actual_att"] - x["pred_att"]
    x["shapley_reconstructed_carry_residual"] = x["room_volume_component"] + x["individual_allocation_component"]
    recon_err = float((x["carry_residual_actual_minus_pred"] - x["shapley_reconstructed_carry_residual"]).abs().max())
    if recon_err > 1e-9:
        raise RuntimeError(f"RB-R1 Shapley reconciliation failed max_abs_error={recon_err}")

    p = profiles[["player_key", "player", "games", "dominant_mechanism"]].copy()
    p = p.rename(columns={"player": "profile_player", "games": "profile_games", "dominant_mechanism": "parent_dominant_mechanism"})
    if p["player_key"].duplicated().any():
        raise RuntimeError("duplicate qualifying player_key in source profiles")
    x = x.merge(p, on="player_key", how="left", validate="many_to_one")

    summaries = [
        slice_summary("ALL_SCOREABLE_ROWS", x),
        slice_summary("CARRIES_DOMINANT_PLAYERS", x.loc[x["parent_dominant_mechanism"].eq("CARRIES")]),
        slice_summary("YPC_DOMINANT_PLAYERS", x.loc[x["parent_dominant_mechanism"].eq("YPC")]),
        slice_summary("MIXED_PLAYERS", x.loc[x["parent_dominant_mechanism"].eq("MIXED")]),
    ]
    summary_frame = pd.DataFrame(summaries)

    player_rows = []
    qualifying_keys = set(profiles["player_key"].astype(str))
    q = x.loc[x["player_key"].astype(str).isin(qualifying_keys)].copy()
    for pk, g in q.groupby("player_key", sort=True):
        parent = str(g["parent_dominant_mechanism"].dropna().iloc[0])
        volume_abs = float(g["room_volume_component"].abs().mean())
        allocation_abs = float(g["individual_allocation_component"].abs().mean())
        carry_resid = g["carry_residual_actual_minus_pred"]
        player_rows.append({
            "player_key": pk,
            "player": g["profile_player"].dropna().iloc[0] if g["profile_player"].notna().any() else g["player"].iloc[0],
            "games": int(len(g)),
            "parent_dominant_mechanism": parent,
            "carry_mae": float(carry_resid.abs().mean()),
            "carry_bias_actual_minus_pred": float(carry_resid.mean()),
            "room_volume_component_abs": volume_abs,
            "individual_allocation_component_abs": allocation_abs,
            "allocation_over_volume_ratio": safe_ratio(allocation_abs, volume_abs),
            "volume_over_allocation_ratio": safe_ratio(volume_abs, allocation_abs),
            "carry_submechanism": classify(volume_abs, allocation_abs),
        })
    player_frame = pd.DataFrame(player_rows)
    if len(player_frame) != 86:
        raise RuntimeError(f"qualified player profile reconstruction drift expected=86 got={len(player_frame)}")

    carry_players = player_frame.loc[player_frame["parent_dominant_mechanism"].eq("CARRIES")].copy()
    if len(carry_players) != 35:
        raise RuntimeError(f"CARRIES-dominant parent count drift expected=35 got={len(carry_players)}")
    carry_summary = next(r for r in summaries if r["slice"] == "CARRIES_DOMINANT_PLAYERS")
    alloc_player_rate = float(carry_players["carry_submechanism"].eq("INDIVIDUAL_ALLOCATION").mean())
    volume_player_rate = float(carry_players["carry_submechanism"].eq("ROOM_VOLUME").mean())

    allocation_route = bool(
        len(carry_players) >= 20
        and np.isfinite(carry_summary["allocation_over_volume_ratio"])
        and carry_summary["allocation_over_volume_ratio"] >= 1.20
        and alloc_player_rate >= 0.50
    )
    volume_route = bool(
        len(carry_players) >= 20
        and np.isfinite(carry_summary["volume_over_allocation_ratio"])
        and carry_summary["volume_over_allocation_ratio"] >= 1.20
        and volume_player_rate >= 0.50
    )
    if allocation_route and volume_route:
        raise RuntimeError("mutually exclusive RB-R1 routing gates both passed")
    if allocation_route:
        disposition = "RB_CARRY_ERRORS_ROUTE_TO_INDIVIDUAL_ALLOCATION"
    elif volume_route:
        disposition = "RB_CARRY_ERRORS_ROUTE_TO_ROOM_VOLUME"
    else:
        disposition = "RB_CARRY_ERRORS_REMAIN_MIXED_ROOM_AND_ALLOCATION"

    result = {
        "migration": "RB_R1_ROOM_VOLUME_VS_INDIVIDUAL_ALLOCATION",
        "source_rows": expected_source,
        "scoreable_rows": int(len(x)),
        "qualifying_players": int(len(player_frame)),
        "carry_dominant_players": int(len(carry_players)),
        "shapley_reconciliation_max_abs_error": recon_err,
        "carry_dominant_slice": carry_summary,
        "carry_dominant_player_submechanism_counts": {str(k): int(v) for k, v in carry_players["carry_submechanism"].value_counts().to_dict().items()},
        "carry_dominant_allocation_player_rate": alloc_player_rate,
        "carry_dominant_room_volume_player_rate": volume_player_rate,
        "routing_gates": {
            "carry_players_ge_20": bool(len(carry_players) >= 20),
            "allocation_slice_ratio_ge_1_20": bool(np.isfinite(carry_summary["allocation_over_volume_ratio"]) and carry_summary["allocation_over_volume_ratio"] >= 1.20),
            "allocation_player_rate_ge_0_50": bool(alloc_player_rate >= 0.50),
            "volume_slice_ratio_ge_1_20": bool(np.isfinite(carry_summary["volume_over_allocation_ratio"]) and carry_summary["volume_over_allocation_ratio"] >= 1.20),
            "volume_player_rate_ge_0_50": bool(volume_player_rate >= 0.50),
        },
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "rb_r1_room_allocation_casebook.csv", index=False)
    summary_frame.to_csv(a.out_dir / "rb_r1_slice_summary.csv", index=False)
    player_frame.sort_values(["parent_dominant_mechanism", "carry_mae"], ascending=[True, False]).to_csv(
        a.out_dir / "rb_r1_player_submechanisms.csv", index=False
    )
    (a.out_dir / "rb_r1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\n=== SLICE SUMMARY ===")
    print(summary_frame.to_string(index=False))
    print("\n=== CARRIES-DOMINANT PLAYER SUBMECHANISMS ===")
    print(carry_players.sort_values("carry_mae", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
