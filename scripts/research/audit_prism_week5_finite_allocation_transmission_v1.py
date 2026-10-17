#!/usr/bin/env python3
"""PRISM Week-5 finite-opportunity transmission audit V1.

Diagnostic only. Quantifies deterministic finite-pool overbooking/compression in
preserved pregame player-state rows and validates exact production attenuation
on the frozen Week-5 top-failure casebook. The cap is treated as a conservation
safeguard, not as the error itself. No candidate projection is fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_V1_AUDIT"
CAP = 0.95
WR_POS = {"WR", "LWR", "RWR", "SWR"}
RUSH_MARKETS = {"rush_yards", "rush_rec_yards"}
RECV_MARKETS = {"rec_yards", "receptions", "rush_rec_yards"}


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "over_odds", "under_odds", "moneyline",
            "closing_line", "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in {label}: {forbidden}")
    return x


def num(s):
    return pd.to_numeric(s, errors="coerce")


def top_n(shares: np.ndarray, n: int = 5) -> np.ndarray:
    clean = np.clip(
        np.nan_to_num(np.asarray(shares, float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        CAP,
    )
    if len(clean) <= n:
        return clean
    order = np.argsort(-clean, kind="stable")
    out = np.zeros_like(clean)
    out[order[:n]] = clean[order[:n]]
    return out


def wr_sharpen(position: pd.Series, shares: np.ndarray) -> np.ndarray:
    """Mirror the simulation WR transform; total WR target mass is conserved."""
    clean = np.clip(
        np.nan_to_num(np.asarray(shares, float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        CAP,
    )
    pos = position.fillna("").astype(str).str.upper().to_numpy()
    idx = np.flatnonzero(np.isin(pos, list(WR_POS)))
    if len(idx) <= 1:
        return clean
    wr = clean[idx].copy()
    total = float(wr.sum())
    if total <= 0:
        return clean
    mults = np.array([1.40, 1.14, 0.91, 0.78], float)
    order = np.argsort(-wr, kind="stable")
    mult = np.ones(len(wr), float)
    for rank, j in enumerate(order):
        mult[j] = mults[min(rank, len(mults) - 1)]
    sharpened = wr * mult
    if sharpened.sum() <= 0:
        return clean
    sharpened *= total / float(sharpened.sum())
    out = clean.copy()
    out[idx] = sharpened
    return out


def global_cap(shares: np.ndarray) -> tuple[np.ndarray, float, float]:
    clean = np.clip(
        np.nan_to_num(np.asarray(shares, float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        CAP,
    )
    mass = float(clean.sum())
    factor = 1.0
    if mass > CAP:
        factor = CAP / mass
        clean = clean * factor
    return clean, mass, float(factor)


def team_state(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "event_id", "team", "player", "player_clean_key", "position",
        "rules_tgt_share", "rules_rush_share",
    }
    missing = required - set(state.columns)
    if missing:
        raise RuntimeError(f"state missing {sorted(missing)}")

    room_rows = []
    player_rows = []
    for (event_id, team), g0 in state.groupby(["event_id", "team"], sort=True):
        g = g0.copy().reset_index(drop=True)

        raw_tgt = num(g["rules_tgt_share"]).fillna(0).to_numpy(float)
        sharpened_tgt = wr_sharpen(g["position"], raw_tgt)
        final_tgt, tgt_mass, tgt_factor = global_cap(sharpened_tgt)
        clipped_raw_tgt = np.clip(
            np.nan_to_num(raw_tgt, nan=0.0, posinf=0.0, neginf=0.0), 0.0, CAP
        )
        if abs(float(sharpened_tgt.sum()) - float(clipped_raw_tgt.sum())) > 1e-10:
            raise RuntimeError(
                f"WR hierarchy changed total target mass {event_id} {team}"
            )

        raw_rush = num(g["rules_rush_share"]).fillna(0).to_numpy(float)
        rush_top = top_n(raw_rush, 5)
        final_rush, rush_mass, rush_factor = global_cap(rush_top)

        room_rows.append({
            "event_id": event_id,
            "team": team,
            "players": len(g),
            "raw_target_mass": float(clipped_raw_tgt.sum()),
            "post_wr_hierarchy_target_mass": float(sharpened_tgt.sum()),
            "target_global_scale_factor": tgt_factor,
            "target_global_cap_applied": bool(tgt_factor < 1.0 - 1e-12),
            "raw_rush_mass_all_players": float(
                np.clip(
                    np.nan_to_num(raw_rush, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                    CAP,
                ).sum()
            ),
            "rush_top5_retained_mass": rush_mass,
            "rush_global_scale_factor": rush_factor,
            "rush_global_cap_applied": bool(rush_factor < 1.0 - 1e-12),
        })

        for i, row in g.iterrows():
            player_rows.append({
                "event_id": event_id,
                "team": team,
                "player": row["player"],
                "player_clean_key": row["player_clean_key"],
                "position": row["position"],
                "rules_tgt_share": raw_tgt[i],
                "post_wr_hierarchy_tgt_share": sharpened_tgt[i],
                "final_target_probability_from_state": final_tgt[i],
                "target_global_scale_factor": tgt_factor,
                "rules_rush_share": raw_rush[i],
                "post_top5_rush_share": rush_top[i],
                "final_rush_probability_from_state": final_rush[i],
                "rush_global_scale_factor": rush_factor,
            })

    return pd.DataFrame(room_rows), pd.DataFrame(player_rows)


def exact_casebook(casebook: pd.DataFrame) -> pd.DataFrame:
    required = {
        "case_rank", "player", "team", "position", "market",
        "rules_tgt_share", "final_target_probability",
        "rules_rush_share", "final_rush_probability",
        "model_proj", "actual", "component_label",
    }
    missing = required - set(casebook.columns)
    if missing:
        raise RuntimeError(f"casebook missing {sorted(missing)}")

    x = casebook.copy()
    for c in (
        "rules_tgt_share", "final_target_probability",
        "rules_rush_share", "final_rush_probability",
        "model_proj", "actual",
    ):
        x[c] = num(x[c])

    x["exact_target_ratio"] = (
        x["final_target_probability"] / x["rules_tgt_share"]
    )
    x["exact_rush_ratio"] = (
        x["final_rush_probability"] / x["rules_rush_share"]
    )

    x["target_path_interpretation"] = "NOT_TARGET_MARKET_OR_UNRESOLVED"
    recv = (
        x["market"].astype(str).isin(RECV_MARKETS)
        & x["rules_tgt_share"].gt(0)
        & x["final_target_probability"].notna()
    )
    non_wr = ~x["position"].astype(str).str.upper().isin(WR_POS)
    x.loc[
        recv & non_wr, "target_path_interpretation"
    ] = "CLEAN_NON_WR_GLOBAL_CAP_PATH"
    x.loc[
        recv & ~non_wr, "target_path_interpretation"
    ] = "WR_HIERARCHY_PLUS_GLOBAL_CAP_COMPOSITE"
    x["implied_target_mass_before_cap_if_clean"] = np.where(
        recv
        & non_wr
        & x["exact_target_ratio"].gt(0)
        & x["exact_target_ratio"].lt(1 - 1e-12),
        CAP / x["exact_target_ratio"],
        np.nan,
    )

    rush = (
        x["market"].astype(str).isin(RUSH_MARKETS)
        & x["rules_rush_share"].gt(0)
        & x["final_rush_probability"].notna()
    )
    x["rush_path_interpretation"] = "NOT_RUSH_MARKET_OR_UNRESOLVED"
    x.loc[
        rush & x["final_rush_probability"].eq(0),
        "rush_path_interpretation",
    ] = "TOP5_TRUNCATED"
    x.loc[
        rush
        & x["final_rush_probability"].gt(0)
        & x["exact_rush_ratio"].lt(1 - 1e-12),
        "rush_path_interpretation",
    ] = "RETAINED_TOP5_GLOBAL_CAP_PATH"
    x.loc[
        rush
        & x["final_rush_probability"].gt(0)
        & x["exact_rush_ratio"].ge(1 - 1e-12),
        "rush_path_interpretation",
    ] = "RETAINED_TOP5_NO_GLOBAL_CAP"
    x["implied_rush_top5_mass_before_cap_if_clean"] = np.where(
        rush
        & x["final_rush_probability"].gt(0)
        & x["exact_rush_ratio"].gt(0)
        & x["exact_rush_ratio"].lt(1 - 1e-12),
        CAP / x["exact_rush_ratio"],
        np.nan,
    )

    x["projection_error"] = x["actual"] - x["model_proj"]
    keep = [
        "case_rank", "player", "team", "position", "market",
        "model_proj", "actual", "projection_error", "component_label",
        "rules_tgt_share", "final_target_probability", "exact_target_ratio",
        "target_path_interpretation",
        "implied_target_mass_before_cap_if_clean",
        "rules_rush_share", "final_rush_probability", "exact_rush_ratio",
        "rush_path_interpretation",
        "implied_rush_top5_mass_before_cap_if_clean",
    ]
    return x[keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player-state", type=Path, required=True)
    ap.add_argument("--casebook", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = read_csv(args.player_state, "Week-5 preserved player state")
    casebook = read_csv(args.casebook, "Week-5 exact casebook")
    rooms, players = team_state(state)
    exact = exact_casebook(casebook)

    if len(rooms) != 30:
        raise RuntimeError(f"expected 30 Week-5 team rooms, got {len(rooms)}")
    if not rooms["target_global_cap_applied"].all():
        raise RuntimeError("expected all preserved Week-5 target rooms to exceed cap")
    if not rooms["rush_global_cap_applied"].all():
        raise RuntimeError(
            "expected all preserved Week-5 top-five rush rooms to exceed cap"
        )

    bowers = exact.loc[
        exact["player"].astype(str).eq("Brock Bowers")
        & exact["market"].eq("rec_yards")
    ]
    if len(bowers) != 1:
        raise RuntimeError("exact Brock Bowers case missing")
    b = bowers.iloc[0]
    if b["target_path_interpretation"] != "CLEAN_NON_WR_GLOBAL_CAP_PATH":
        raise RuntimeError("Bowers target path is not clean non-WR cap path")
    if abs(float(b["rules_tgt_share"]) - 0.27766642611) > 1e-9:
        raise RuntimeError("Bowers rules target share drift")
    if abs(float(b["final_target_probability"]) - 0.206264830495) > 1e-9:
        raise RuntimeError("Bowers final target probability drift")

    clean_target = exact.loc[
        exact["target_path_interpretation"].eq("CLEAN_NON_WR_GLOBAL_CAP_PATH")
    ].drop_duplicates(["team", "player"])
    clean_rush = exact.loc[
        exact["rush_path_interpretation"].isin([
            "RETAINED_TOP5_GLOBAL_CAP_PATH",
            "RETAINED_TOP5_NO_GLOBAL_CAP",
        ])
    ].drop_duplicates(["team", "player"])

    result = {
        "version": VERSION,
        "disposition": (
            "PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_AUDIT_COMPLETE__"
            "ARCHITECTURE_GAP_CONFIRMED__NO_MODEL_CHANGE"
        ),
        "preserved_state_team_rooms": int(len(rooms)),
        "preserved_state_player_rows": int(len(players)),
        "target_rooms_global_cap_applied": int(
            rooms["target_global_cap_applied"].sum()
        ),
        "target_rooms_global_cap_rate": float(
            rooms["target_global_cap_applied"].mean()
        ),
        "target_raw_mass_mean": float(rooms["raw_target_mass"].mean()),
        "target_raw_mass_median": float(rooms["raw_target_mass"].median()),
        "target_raw_mass_min": float(rooms["raw_target_mass"].min()),
        "target_raw_mass_max": float(rooms["raw_target_mass"].max()),
        "target_scale_factor_mean": float(
            rooms["target_global_scale_factor"].mean()
        ),
        "target_scale_factor_min": float(
            rooms["target_global_scale_factor"].min()
        ),
        "target_scale_factor_max": float(
            rooms["target_global_scale_factor"].max()
        ),
        "rush_rooms_global_cap_applied": int(
            rooms["rush_global_cap_applied"].sum()
        ),
        "rush_rooms_global_cap_rate": float(
            rooms["rush_global_cap_applied"].mean()
        ),
        "rush_top5_mass_mean": float(rooms["rush_top5_retained_mass"].mean()),
        "rush_top5_mass_median": float(
            rooms["rush_top5_retained_mass"].median()
        ),
        "rush_top5_mass_min": float(rooms["rush_top5_retained_mass"].min()),
        "rush_top5_mass_max": float(rooms["rush_top5_retained_mass"].max()),
        "rush_scale_factor_mean": float(
            rooms["rush_global_scale_factor"].mean()
        ),
        "rush_scale_factor_min": float(
            rooms["rush_global_scale_factor"].min()
        ),
        "rush_scale_factor_max": float(
            rooms["rush_global_scale_factor"].max()
        ),
        "exact_casebook_clean_nonwr_target_identities": int(len(clean_target)),
        "exact_casebook_clean_nonwr_target_compressed": int(
            (clean_target["exact_target_ratio"] < 1 - 1e-12).sum()
        ),
        "exact_casebook_clean_rush_identities": int(len(clean_rush)),
        "exact_casebook_clean_rush_compressed": int(
            (clean_rush["exact_rush_ratio"] < 1 - 1e-12).sum()
        ),
        "bowers_rules_tgt_share": float(b["rules_tgt_share"]),
        "bowers_final_target_probability": float(
            b["final_target_probability"]
        ),
        "bowers_global_scale_factor": float(b["exact_target_ratio"]),
        "bowers_relative_attenuation": float(1.0 - b["exact_target_ratio"]),
        "bowers_implied_team_target_mass_before_global_cap": float(
            b["implied_target_mass_before_cap_if_clean"]
        ),
        "scientific_interpretation": (
            "Independent player share estimates overbook finite opportunity; "
            "proportional cap normalization can attenuate an accurate named-player "
            "role signal. The cap is a conservation safeguard, not itself a "
            "fitted player-state mechanism."
        ),
        "stage2_transport_week4_5_status": (
            "SOURCE_BLOCKED_EXACT_PREGAME_Q_AND_AVAILABILITY_ARTIFACTS_"
            "EXPIRED__DO_NOT_APPROXIMATE"
        ),
        "qualified_stage2_cells_preserved": [
            "RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING"
        ],
        "qb_stage2_status": "CLOSED",
        "wr_stage2_status": "SOURCE_BLOCKED_NOT_SCORED",
        "raw_recent_share_replacement_reopened": False,
        "new_model_fit": 0,
        "threshold_searches": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "week6_outcomes_used": 0,
        "week6_locks_changed": False,
        "production_changed": False,
    }

    rooms.to_csv(
        args.out_dir / "prism_week5_finite_allocation_team_rooms.csv",
        index=False,
        float_format="%.12f",
    )
    players.to_csv(
        args.out_dir / "prism_week5_finite_allocation_player_transmission.csv",
        index=False,
        float_format="%.12f",
    )
    exact.to_csv(
        args.out_dir / "prism_week5_exact_casebook_transmission.csv",
        index=False,
        float_format="%.12f",
    )
    (
        args.out_dir / "prism_week5_finite_allocation_transmission_result.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
