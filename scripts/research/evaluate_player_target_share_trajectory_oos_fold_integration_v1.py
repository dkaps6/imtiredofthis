#!/usr/bin/env python3
"""Exact frozen Target Share Trajectory V1 integration on OOS specialist folds.

Research only. Candidate construction is deliberately separated from outcome
columns. No provider fetches, sportsbook inputs, fitting, thresholds, or 2026
outcomes are permitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_OOS_FOLD_INTEGRATION_V1"
WR_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
BOOT_REPS = 5000
BOOT_SEED = 20261008
TOL = 1e-12

KEYS = ["season", "week", "team", "player_clean_key"]
ROOM_KEYS = ["season", "week", "event_id", "team"]
TRAJ_COLS = KEYS + ["position_group", "trajectory_delta", "feature_max_week"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _team(v):
    s = "" if pd.isna(v) else str(v).strip().upper()
    return "WAS" if s == "WSH" else s


def _find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def _load_trajectory(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in set(TRAJ_COLS))
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = set(TRAJ_COLS) - set(x.columns)
    if missing:
        raise RuntimeError(f"trajectory artifact missing columns: {sorted(missing)}")
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    x["position_group"] = x["position_group"].astype(str).str.upper()
    x["trajectory_delta"] = _num(x["trajectory_delta"])
    x["feature_max_week"] = _num(x["feature_max_week"])
    if x.duplicated(KEYS + ["position_group"]).any():
        raise RuntimeError("duplicate trajectory identity rows")
    if (x["feature_max_week"] >= x["week"]).any():
        raise RuntimeError("same/future trajectory feature row detected")
    return x


def _attach_trajectory(base: pd.DataFrame, traj: pd.DataFrame, position: str) -> pd.DataFrame:
    t = traj.loc[traj["position_group"].eq(position), TRAJ_COLS].copy()
    out = base.merge(
        t,
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_trajectory"),
    )
    if "position_group" in out.columns:
        bad = out["position_group"].notna() & ~out["position_group"].eq(position)
        if bad.any():
            raise RuntimeError(f"{position} trajectory position mismatch")
    out["trajectory_available"] = out["trajectory_delta"].notna()
    out["trajectory_delta"] = _num(out["trajectory_delta"]).fillna(0.0)
    chronology = out.loc[out["trajectory_available"], "feature_max_week"] >= out.loc[out["trajectory_available"], "week"]
    if chronology.any():
        raise RuntimeError(f"{position} same/future trajectory violation")
    out["position_group"] = position
    return out


def _renormalize(values: np.ndarray, deltas: np.ndarray, pool: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(deltas).all():
        raise RuntimeError("non-finite room input")
    if (values < -TOL).any():
        raise RuntimeError("negative baseline target opportunity")
    if pool <= TOL:
        return np.zeros(len(values), dtype=float)
    weights = values * np.exp(deltas)
    denom = float(weights.sum())
    if denom <= 0 or not np.isfinite(denom):
        raise RuntimeError("invalid trajectory room weights")
    cand = pool * weights / denom
    # Force exact floating conservation onto the largest weight.
    gap = float(pool) - float(cand.sum())
    cand[int(np.argmax(weights))] += gap
    return cand


def apply_wr_trajectory(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = base.copy()
    out["shadow_targets"] = out["baseline_targets"].astype(float)
    audits = []
    for room, idx in out.groupby(ROOM_KEYS, sort=False).groups.items():
        g = out.loc[idx]
        anchors = g["is_wr1"].astype(bool)
        if int(anchors.sum()) != 1:
            raise RuntimeError(f"WR room must contain exactly one frozen WR1 anchor: {room}")
        sec_idx = g.index[~anchors]
        sec_pool = float(g.loc[~anchors, "baseline_targets"].sum())
        if len(sec_idx):
            cand = _renormalize(
                g.loc[~anchors, "baseline_targets"].to_numpy(float),
                g.loc[~anchors, "trajectory_delta"].to_numpy(float),
                sec_pool,
            )
            out.loc[sec_idx, "shadow_targets"] = cand
        anchor_gap = float(
            (out.loc[g.index[anchors], "shadow_targets"] - g.loc[g.index[anchors], "baseline_targets"]).abs().max()
        )
        sec_gap = abs(float(out.loc[sec_idx, "shadow_targets"].sum()) - sec_pool) if len(sec_idx) else 0.0
        total_gap = abs(float(out.loc[g.index, "shadow_targets"].sum()) - float(g["baseline_targets"].sum()))
        audits.append({
            **dict(zip(ROOM_KEYS, room)),
            "position_group": "WR",
            "wr1_max_abs_gap": anchor_gap,
            "protected_pool_max_abs_gap": sec_gap,
            "room_pool_max_abs_gap": total_gap,
        })
    return out, pd.DataFrame(audits)


def apply_te_trajectory(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = base.copy()
    out["shadow_targets"] = out["baseline_targets"].astype(float)
    audits = []
    for room, idx in out.groupby(ROOM_KEYS, sort=False).groups.items():
        g = out.loc[idx]
        pool = float(g["baseline_targets"].sum())
        cand = _renormalize(
            g["baseline_targets"].to_numpy(float),
            g["trajectory_delta"].to_numpy(float),
            pool,
        )
        out.loc[g.index, "shadow_targets"] = cand
        gap = abs(float(out.loc[g.index, "shadow_targets"].sum()) - pool)
        audits.append({
            **dict(zip(ROOM_KEYS, room)),
            "position_group": "TE",
            "wr1_max_abs_gap": 0.0,
            "protected_pool_max_abs_gap": gap,
            "room_pool_max_abs_gap": gap,
        })
    return out, pd.DataFrame(audits)


def _prepare_wr(path: Path, traj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "variant", "event_id", "team", "player_clean_key", "player", "wr_rank",
        "entitlement_tgt_share", "pred_targets", "mc_rec_yards",
        "season", "train_season", "week", "actual_targets", "actual_rec_yards",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"WR parent missing columns: {sorted(missing)}")
    x = x.loc[x["variant"].astype(str).eq(WR_VARIANT)].copy()
    if len(x) != 4193:
        raise RuntimeError(f"WR candidate parent row drift: {len(x)}")
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate WR parent identity")
    primary = x.loc[x["season"].eq(2024) & x["week"].ge(5)].copy()
    if sorted(_num(primary["train_season"]).dropna().astype(int).unique().tolist()) != [2023]:
        raise RuntimeError("WR 2024 fold is not exact train-2023 -> test-2024")
    # Candidate construction sees no realized target-game outcomes.
    base = primary[[
        "season", "week", "event_id", "team", "player_clean_key", "player",
        "wr_rank", "entitlement_tgt_share", "pred_targets", "mc_rec_yards",
    ]].copy()
    base = base.rename(columns={
        "entitlement_tgt_share": "baseline_entitlement_tgt_share",
        "pred_targets": "baseline_targets",
        "mc_rec_yards": "baseline_rec_yards",
    })
    base["baseline_entitlement_tgt_share"] = _num(base["baseline_entitlement_tgt_share"])
    base["baseline_targets"] = _num(base["baseline_targets"])
    base["baseline_rec_yards"] = _num(base["baseline_rec_yards"])
    # Exact original WR-R15 OOS-fold semantics: M38/WR-R15 anchor is the
    # max baseline entitlement row in each event/team room, NOT historical
    # descriptive wr_rank. This reproduces apply_wr_fold() exactly.
    base["is_wr1"] = False
    for _, idx in base.groupby(ROOM_KEYS, sort=False).groups.items():
        anchor = base.loc[idx, "baseline_entitlement_tgt_share"].idxmax()
        base.loc[anchor, "is_wr1"] = True
    if not base.groupby(ROOM_KEYS)["is_wr1"].sum().eq(1).all():
        raise RuntimeError("WR anchor reconstruction failed")
    base = _attach_trajectory(base, traj, "WR")
    candidate, audit = apply_wr_trajectory(base)
    outcome = primary[KEYS + ["actual_targets", "actual_rec_yards"]].copy()
    outcome["actual_targets"] = _num(outcome["actual_targets"])
    outcome["actual_rec_yards"] = _num(outcome["actual_rec_yards"])
    candidate = candidate.merge(outcome, on=KEYS, how="left", validate="one_to_one")
    return candidate, audit


def _prepare_te(path: Path, folds_path: Path, traj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(folds_path)
    folds.columns = [str(c).strip().lower() for c in folds.columns]
    f24 = folds.loc[_num(folds["test_season"]).eq(2024)]
    f25 = folds.loc[_num(folds["test_season"]).eq(2025)]
    if len(f24) != 1 or str(f24.iloc[0]["train_seasons"]) != "2022,2023":
        raise RuntimeError("TE 2024 OOS fold lineage drift")
    if len(f25) != 1 or str(f25.iloc[0]["train_seasons"]) != "2022,2023,2024":
        raise RuntimeError("TE 2025 OOS fold lineage drift")

    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "event_id", "team", "player_clean_key", "player",
        "candidate_targets_r5p", "candidate_rec_yards_r5p",
        "targets", "rec_yards",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"TE parent missing columns: {sorted(missing)}")
    if len(x) != 3214:
        raise RuntimeError(f"TE parent row drift: {len(x)}")
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate TE parent identity")
    use = x.loc[x["season"].isin([2024, 2025]) & x["week"].ge(5)].copy()
    base = use[[
        "season", "week", "event_id", "team", "player_clean_key", "player",
        "candidate_targets_r5p", "candidate_rec_yards_r5p",
    ]].copy()
    base = base.rename(columns={
        "candidate_targets_r5p": "baseline_targets",
        "candidate_rec_yards_r5p": "baseline_rec_yards",
    })
    base["baseline_targets"] = _num(base["baseline_targets"])
    base["baseline_rec_yards"] = _num(base["baseline_rec_yards"])
    base["is_wr1"] = False
    base = _attach_trajectory(base, traj, "TE")
    candidate, audit = apply_te_trajectory(base)
    outcome = use[KEYS + ["targets", "rec_yards"]].copy().rename(
        columns={"targets": "actual_targets", "rec_yards": "actual_rec_yards"}
    )
    outcome["actual_targets"] = _num(outcome["actual_targets"])
    outcome["actual_rec_yards"] = _num(outcome["actual_rec_yards"])
    candidate = candidate.merge(outcome, on=KEYS, how="left", validate="one_to_one")
    return candidate, audit


def _translate_efficiency(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    if out[["baseline_targets", "baseline_rec_yards", "shadow_targets"]].isna().any().any():
        raise RuntimeError("missing baseline candidate values")
    bad_zero = out["baseline_targets"].le(0) & out["baseline_rec_yards"].abs().gt(TOL)
    if bad_zero.any():
        raise RuntimeError("nonzero receiving mean with zero baseline targets")
    ypt = np.where(
        out["baseline_targets"].gt(0),
        out["baseline_rec_yards"] / out["baseline_targets"],
        0.0,
    )
    out["baseline_ypt"] = ypt
    out["shadow_rec_yards"] = out["shadow_targets"] * out["baseline_ypt"]
    return out


def _room_shares(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    base_pool = out.groupby(ROOM_KEYS)["baseline_targets"].transform("sum")
    actual_pool = out.groupby(ROOM_KEYS)["actual_targets"].transform("sum")
    out["baseline_room_share"] = np.where(base_pool.gt(0), out["baseline_targets"] / base_pool, np.nan)
    out["shadow_room_share"] = np.where(base_pool.gt(0), out["shadow_targets"] / base_pool, np.nan)
    out["actual_room_share"] = np.where(actual_pool.gt(0), out["actual_targets"] / actual_pool, np.nan)
    return out


def _metrics(x: pd.DataFrame) -> dict:
    z = x.dropna(subset=["actual_targets", "actual_rec_yards"]).copy()
    bt = z["baseline_targets"] - z["actual_targets"]
    st = z["shadow_targets"] - z["actual_targets"]
    by = z["baseline_rec_yards"] - z["actual_rec_yards"]
    sy = z["shadow_rec_yards"] - z["actual_rec_yards"]
    room = z.dropna(subset=["actual_room_share"]).copy()
    br = (room["baseline_room_share"] - room["actual_room_share"]).abs()
    sr = (room["shadow_room_share"] - room["actual_room_share"]).abs()
    bae = bt.abs()
    sae = st.abs()
    return {
        "rows": int(len(z)),
        "players": int(z["player_clean_key"].nunique()),
        "rooms": int(z[ROOM_KEYS].drop_duplicates().shape[0]),
        "trajectory_available_rows": int(z["trajectory_available"].sum()),
        "changed_rows": int((z["shadow_targets"] - z["baseline_targets"]).abs().gt(TOL).sum()),
        "baseline_target_mae": float(bae.mean()),
        "shadow_target_mae": float(sae.mean()),
        "target_mae_improvement": float(bae.mean() - sae.mean()),
        "baseline_target_rmse": float(np.sqrt(np.mean(np.square(bt)))),
        "shadow_target_rmse": float(np.sqrt(np.mean(np.square(st)))),
        "baseline_target_bias": float(bt.mean()),
        "shadow_target_bias": float(st.mean()),
        "baseline_rec_yards_mae": float(by.abs().mean()),
        "shadow_rec_yards_mae": float(sy.abs().mean()),
        "rec_yards_mae_improvement": float(by.abs().mean() - sy.abs().mean()),
        "baseline_rec_yards_rmse": float(np.sqrt(np.mean(np.square(by)))),
        "shadow_rec_yards_rmse": float(np.sqrt(np.mean(np.square(sy)))),
        "baseline_room_share_mae": float(br.mean()),
        "shadow_room_share_mae": float(sr.mean()),
        "room_share_mae_improvement": float(br.mean() - sr.mean()),
        "shadow_closer_target": int((sae < bae - TOL).sum()),
        "baseline_closer_target": int((bae < sae - TOL).sum()),
        "target_ties": int((bae - sae).abs().le(TOL).sum()),
    }


def _bootstrap_target_improvement(x: pd.DataFrame) -> dict:
    z = x.dropna(subset=["actual_targets"]).copy()
    z["improvement"] = (
        (z["baseline_targets"] - z["actual_targets"]).abs()
        - (z["shadow_targets"] - z["actual_targets"]).abs()
    )
    z["cluster"] = z["position_group"].astype(str) + "|" + z["player_clean_key"].astype(str)
    g = z.groupby("cluster")["improvement"].agg(["sum", "count"]).reset_index()
    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(int)
    if len(g) < 2:
        raise RuntimeError("insufficient player clusters for bootstrap")
    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        draw = rng.integers(0, len(g), size=len(g))
        vals[i] = float(sums[draw].sum() / counts[draw].sum())
    return {
        "reps": BOOT_REPS,
        "seed": BOOT_SEED,
        "player_clusters": int(len(g)),
        "mean_improvement": float(z["improvement"].mean()),
        "p_improvement_gt_0": float((vals > 0).mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents-root", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    wr_path = _find_one(a.parents_root, "wr_r15_confirmation_predictions.csv")
    te_path = _find_one(a.parents_root, "te_r5p_oos_player_casebook.csv")
    folds_path = _find_one(a.parents_root, "te_r5p_folds.csv")
    traj = _load_trajectory(a.trajectory)

    wr, wr_audit = _prepare_wr(wr_path, traj)
    te, te_audit = _prepare_te(te_path, folds_path, traj)
    rows = pd.concat([wr, te], ignore_index=True, sort=False)
    rows = _translate_efficiency(rows)
    rows = _room_shares(rows)

    audit = pd.concat([wr_audit, te_audit], ignore_index=True, sort=False)
    max_wr1 = float(audit["wr1_max_abs_gap"].max()) if len(audit) else 0.0
    max_protected = float(audit["protected_pool_max_abs_gap"].max()) if len(audit) else 0.0
    max_room = float(audit["room_pool_max_abs_gap"].max()) if len(audit) else 0.0
    if max(max_wr1, max_protected, max_room) > TOL:
        raise RuntimeError(
            f"conservation failed wr1={max_wr1} protected={max_protected} room={max_room}"
        )

    primary = rows.loc[rows["season"].eq(2024)].copy()
    primary_all = _metrics(primary)
    primary_wr = _metrics(primary.loc[primary["position_group"].eq("WR")])
    primary_te = _metrics(primary.loc[primary["position_group"].eq("TE")])
    secondary_te_2025 = _metrics(
        rows.loc[rows["season"].eq(2025) & rows["position_group"].eq("TE")]
    )
    boot = _bootstrap_target_improvement(primary)

    weekly = []
    for (season, position, week), g in rows.groupby(["season", "position_group", "week"], sort=True):
        m = _metrics(g)
        weekly.append({
            "season": int(season), "position_group": str(position), "week": int(week),
            "rows": m["rows"],
            "baseline_target_mae": m["baseline_target_mae"],
            "shadow_target_mae": m["shadow_target_mae"],
            "target_mae_improvement": m["target_mae_improvement"],
        })
    weekly = pd.DataFrame(weekly)

    gates = {
        "primary_pooled_target_mae_improves":
            primary_all["shadow_target_mae"] < primary_all["baseline_target_mae"],
        "primary_wr_target_mae_improves":
            primary_wr["shadow_target_mae"] < primary_wr["baseline_target_mae"],
        "primary_te_target_mae_improves":
            primary_te["shadow_target_mae"] < primary_te["baseline_target_mae"],
        "primary_pooled_rec_yards_nonworse":
            primary_all["shadow_rec_yards_mae"] <= primary_all["baseline_rec_yards_mae"],
        "primary_wr_rec_yards_nonworse":
            primary_wr["shadow_rec_yards_mae"] <= primary_wr["baseline_rec_yards_mae"],
        "primary_te_rec_yards_nonworse":
            primary_te["shadow_rec_yards_mae"] <= primary_te["baseline_rec_yards_mae"],
        "bootstrap_p_ge_0p80": boot["p_improvement_gt_0"] >= 0.80,
        "primary_room_share_mae_nonworse":
            primary_all["shadow_room_share_mae"] <= primary_all["baseline_room_share_mae"],
        "conservation_pass": max(max_wr1, max_protected, max_room) <= TOL,
    }
    supported = all(bool(v) for v in gates.values())
    disposition = (
        "HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED"
        if supported else
        "HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_NOT_CONFIRMED"
    )

    result = {
        "version": VERSION,
        "disposition": disposition,
        "primary_population": "2024_W5PLUS_WR_TE_OOS_FOLD_PARENTS",
        "secondary_population": "2025_W5PLUS_TE_OOS_FOLD_PARENT_DISCLOSURE_ONLY",
        "primary_all": primary_all,
        "primary_wr": primary_wr,
        "primary_te": primary_te,
        "secondary_te_2025": secondary_te_2025,
        "bootstrap_primary_target_mae": boot,
        "gates": gates,
        "max_wr1_abs_gap": max_wr1,
        "max_protected_pool_abs_gap": max_protected,
        "max_room_pool_abs_gap": max_room,
        "trajectory_rule": "baseline_targets * exp(trajectory_delta), protected-room renormalization",
        "trajectory_minimum_prior_team_games": 4,
        "trajectory_missing_state_action": "ZERO_DELTA_NO_CHANGE",
        "candidate_models_fit": 0,
        "threshold_searches": 0,
        "fresh_historical_provider_rebuild": False,
        "sportsbook_inputs_used": False,
        "paid_oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "production_changed": False,
        "prospective_week5_contract_unchanged": True,
        "automatic_production_promotion": False,
    }

    rows.to_csv(a.out_dir / "trajectory_oos_fold_integration_rows.csv", index=False)
    audit.to_csv(a.out_dir / "trajectory_oos_fold_conservation_audit.csv", index=False)
    weekly.to_csv(a.out_dir / "trajectory_oos_fold_weekly_summary.csv", index=False)
    (a.out_dir / "trajectory_oos_fold_integration_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
