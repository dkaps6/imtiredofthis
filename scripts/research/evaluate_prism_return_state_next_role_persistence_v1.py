#!/usr/bin/env python3
"""PRISM Return-State Next-Role Persistence V1.

Scores whether a source-certified restored return is followed by a clean
same-team participation observation closer to the named player's own pre-injury
participation role than to the interruption/continued-decline state.

Participation state only. No targets, carries, receptions, yards, touchdowns,
sportsbook inputs, or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1"
FREEZE_COMMIT = "c25d5f3d19cab3f68a0c0ad0482c79bf53a6730e"
SEASONS = (2024, 2025)
FAMILIES = {"QB", "RB", "WR", "TE"}

RETURN_RESTORED = "RETURN_PARTICIPATION_RESTORED"
START_COLLAPSE = "INJURY_CORROBORATED_PARTICIPATION_COLLAPSE"
START_NO_SNAPS = "REPORTED_INJURY_NO_OFFENSIVE_SNAPS"
TARGET_CLEAN = "NO_REPORT_PARTICIPATED"

N_BOOT = 5000
BOOT_SEED = 20261018
TOL = 1e-12

EP_REQ = {
    "season","team","stable_id","family","episode_number","episode_start_week",
    "episode_start_type","pre_episode_snap_pct","return_week",
    "return_offense_pct","return_state",
}
HEALTH_REQ = {
    "season","week","team","stable_id","family","offense_pct",
    "positive_participation","observation_state",
}


def text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def read_episodes(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = EP_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"episode source missing {sorted(miss)}")
    for c in ("season","episode_number","episode_start_week","return_week"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in ("pre_episode_snap_pct","return_offense_pct"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["stable_id"] = x["stable_id"].map(text)
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["episode_start_type"] = x["episode_start_type"].astype("string").fillna("").str.strip()
    x["return_state"] = x["return_state"].astype("string").fillna("").str.strip()
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["stable_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["episode_number"] = x["episode_number"].astype(int)
    x["episode_start_week"] = x["episode_start_week"].astype(int)
    return x.sort_values(
        ["season","team","stable_id","episode_start_week","episode_number"],
        kind="mergesort"
    ).reset_index(drop=True)


def read_health(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = HEALTH_REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"health source missing {sorted(miss)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["offense_pct"] = pd.to_numeric(x["offense_pct"], errors="coerce")
    x["stable_id"] = x["stable_id"].map(text)
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["observation_state"] = x["observation_state"].astype("string").fillna("").str.strip()
    x["positive_participation"] = as_bool(x["positive_participation"])
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["stable_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season","week","team","stable_id"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate health identities: {bad}")
    return x.sort_values(["season","team","stable_id","week"], kind="mergesort").reset_index(drop=True)


def build_scored_rows(
    episodes: pd.DataFrame,
    health: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    restored = episodes.loc[episodes["return_state"].eq(RETURN_RESTORED)].copy()

    dup_keys = ["season","team","stable_id","episode_number"]
    duplicate_episode_identities = int(restored.duplicated(dup_keys).sum())

    health_groups = {
        (int(s), str(t), str(pid)): g.sort_values("week", kind="mergesort").reset_index(drop=True)
        for (s,t,pid), g in health.groupby(["season","team","stable_id"], sort=False)
    }

    rows = []
    exclusions: dict[str,int] = {}
    chronology_violations = 0
    start_rows_missing = 0
    return_rows_missing = 0

    def exclude(reason: str):
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for e in restored.to_dict("records"):
        season = int(e["season"])
        team = str(e["team"])
        pid = str(e["stable_id"])
        family = str(e["family"])
        start_week = int(e["episode_start_week"])
        if pd.isna(e["return_week"]):
            exclude("missing_return_week")
            continue
        return_week = int(e["return_week"])
        pre_anchor = float(e["pre_episode_snap_pct"])

        if not (return_week > start_week):
            chronology_violations += 1
            exclude("return_not_after_start")
            continue
        if not np.isfinite(pre_anchor) or pre_anchor <= 0:
            exclude("invalid_pre_role_anchor")
            continue

        g = health_groups.get((season, team, pid))
        if g is None or g.empty:
            exclude("player_health_history_missing")
            continue

        start = g.loc[g["week"].eq(start_week)]
        if len(start) != 1:
            start_rows_missing += 1
            exclude("episode_start_health_row_not_unique")
            continue
        start = start.iloc[0]

        ret = g.loc[g["week"].eq(return_week)]
        if len(ret) != 1:
            return_rows_missing += 1
            exclude("return_health_row_not_unique")
            continue
        ret = ret.iloc[0]

        start_type = str(e["episode_start_type"])
        if start_type == START_COLLAPSE:
            start_pct = float(start["offense_pct"]) if pd.notna(start["offense_pct"]) else np.nan
            if (
                not bool(start["positive_participation"])
                or not np.isfinite(start_pct)
                or start_pct <= 0
            ):
                exclude("invalid_collapse_decline_anchor")
                continue
            decline_anchor = start_pct
        elif start_type == START_NO_SNAPS:
            if bool(start["positive_participation"]):
                exclude("no_snap_start_has_positive_participation")
                continue
            decline_anchor = 0.0
        else:
            exclude("unknown_episode_start_type")
            continue

        future = g.loc[g["week"].gt(return_week)].sort_values("week", kind="mergesort")
        if future.empty:
            exclude("no_immediate_next_observation")
            continue

        target = future.iloc[0]
        target_week = int(target["week"])
        if not (target_week > return_week > start_week):
            chronology_violations += 1
            exclude("target_chronology_violation")
            continue

        target_state = str(target["observation_state"])
        if target_state != TARGET_CLEAN:
            exclude(f"immediate_next_state__{target_state or 'BLANK'}")
            continue

        actual = float(target["offense_pct"]) if pd.notna(target["offense_pct"]) else np.nan
        if (
            not bool(target["positive_participation"])
            or not np.isfinite(actual)
            or actual <= 0
        ):
            exclude("clean_target_invalid_participation")
            continue

        gain = abs(decline_anchor - actual) - abs(pre_anchor - actual)

        rows.append({
            "season": season,
            "team": team,
            "stable_id": pid,
            "family": family,
            "episode_number": int(e["episode_number"]),
            "episode_start_week": start_week,
            "episode_start_type": start_type,
            "return_week": return_week,
            "target_week": target_week,
            "target_observation_state": target_state,
            "pre_role_anchor": pre_anchor,
            "decline_anchor": float(decline_anchor),
            "return_offense_pct": float(e["return_offense_pct"]),
            "actual_next_offense_pct": actual,
            "pre_role_closeness_gain": float(gain),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["season","target_week","team","stable_id","episode_number"],
            kind="mergesort"
        ).reset_index(drop=True)

    audit = {
        "restored_source_episodes": int(len(restored)),
        "scored_episodes": int(len(out)),
        "duplicate_episode_identities": duplicate_episode_identities,
        "chronology_violations": int(chronology_violations),
        "start_rows_missing_or_nonunique": int(start_rows_missing),
        "return_rows_missing_or_nonunique": int(return_rows_missing),
        "exclusions": dict(sorted(exclusions.items())),
    }
    return out, audit


def player_summary(s: pd.DataFrame) -> pd.Series:
    return (
        s.groupby(["season","stable_id"], sort=False)["pre_role_closeness_gain"]
        .mean()
        .astype(float)
    )


def bootstrap_players(s: pd.DataFrame) -> dict:
    p = (
        s.groupby("stable_id", sort=False)["pre_role_closeness_gain"]
        .mean()
        .astype(float)
    )
    vals = p.to_numpy(float)
    if len(vals) < 2:
        return {"reps": 0, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        boots[i] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "reps": N_BOOT,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def summarize(s: pd.DataFrame, season: int, bootstrap: bool) -> dict:
    z = s.loc[s["season"].eq(season)].copy()
    if z.empty:
        return {
            "season": season,
            "scored_episodes": 0,
            "named_players": 0,
            "equal_player_average_gain": np.nan,
            "median_player_gain": np.nan,
            "improved_player_share": np.nan,
            "tied_player_share": np.nan,
            "worsened_player_share": np.nan,
            "p10_player_gain": np.nan,
            "p25_player_gain": np.nan,
            "p75_player_gain": np.nan,
            "p90_player_gain": np.nan,
            "family_diagnostics": [],
            **({"player_bootstrap": {"reps":0,"ci_low":np.nan,"ci_high":np.nan}} if bootstrap else {}),
        }

    p = z.groupby("stable_id")["pre_role_closeness_gain"].mean().astype(float)
    improved = p > TOL
    worsened = p < -TOL
    tied = ~(improved | worsened)

    fam = []
    for family, g in z.groupby("family", sort=True):
        pg = g.groupby("stable_id")["pre_role_closeness_gain"].mean().astype(float)
        fi = pg > TOL
        fw = pg < -TOL
        ft = ~(fi | fw)
        fam.append({
            "family": str(family),
            "scored_episodes": int(len(g)),
            "named_players": int(len(pg)),
            "equal_player_average_gain": float(pg.mean()),
            "median_player_gain": float(pg.median()),
            "improved_player_share": float(fi.mean()),
            "tied_player_share": float(ft.mean()),
            "worsened_player_share": float(fw.mean()),
        })

    out = {
        "season": season,
        "scored_episodes": int(len(z)),
        "named_players": int(len(p)),
        "equal_player_average_gain": float(p.mean()),
        "median_player_gain": float(p.median()),
        "improved_player_share": float(improved.mean()),
        "tied_player_share": float(tied.mean()),
        "worsened_player_share": float(worsened.mean()),
        "p10_player_gain": float(p.quantile(0.10)),
        "p25_player_gain": float(p.quantile(0.25)),
        "p75_player_gain": float(p.quantile(0.75)),
        "p90_player_gain": float(p.quantile(0.90)),
        "family_diagnostics": fam,
    }
    if bootstrap:
        out["player_bootstrap"] = bootstrap_players(z)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--health", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    episodes = read_episodes(a.episodes)
    health = read_health(a.health)
    rows, audit = build_scored_rows(episodes, health)

    dev = summarize(rows, 2024, bootstrap=False)
    conf = summarize(rows, 2025, bootstrap=True)

    support_gates = {
        "duplicate_episode_identities_zero": audit["duplicate_episode_identities"] == 0,
        "chronology_violations_zero": audit["chronology_violations"] == 0,
        "development_scored_episodes_ge_80": dev["scored_episodes"] >= 80,
        "development_named_players_ge_70": dev["named_players"] >= 70,
        "confirmation_scored_episodes_ge_80": conf["scored_episodes"] >= 80,
        "confirmation_named_players_ge_70": conf["named_players"] >= 70,
        "all_targets_immediate_clean_state": bool(
            len(rows) > 0 and rows["target_observation_state"].eq(TARGET_CLEAN).all()
        ),
        "all_target_offense_pct_finite_positive": bool(
            len(rows) > 0
            and np.isfinite(rows["actual_next_offense_pct"]).all()
            and rows["actual_next_offense_pct"].gt(0).all()
        ),
        "all_pre_role_anchor_finite_positive": bool(
            len(rows) > 0
            and np.isfinite(rows["pre_role_anchor"]).all()
            and rows["pre_role_anchor"].gt(0).all()
        ),
        "all_decline_anchor_finite_nonnegative": bool(
            len(rows) > 0
            and np.isfinite(rows["decline_anchor"]).all()
            and rows["decline_anchor"].ge(0).all()
        ),
        "outcomes_2026_zero": True,
        "sportsbook_oddsapi_zero": True,
        "coefficients_fit_zero": True,
        "thresholds_searched_zero": True,
        "production_changed_false": True,
    }

    qualification_gates = {
        "development_equal_player_positive": bool(
            np.isfinite(dev["equal_player_average_gain"])
            and dev["equal_player_average_gain"] > 0
        ),
        "development_median_player_nonnegative": bool(
            np.isfinite(dev["median_player_gain"])
            and dev["median_player_gain"] >= 0
        ),
        "development_improved_share_gt_worsened": bool(
            np.isfinite(dev["improved_player_share"])
            and dev["improved_player_share"] > dev["worsened_player_share"]
        ),
        "confirmation_equal_player_positive": bool(
            np.isfinite(conf["equal_player_average_gain"])
            and conf["equal_player_average_gain"] > 0
        ),
        "confirmation_median_player_nonnegative": bool(
            np.isfinite(conf["median_player_gain"])
            and conf["median_player_gain"] >= 0
        ),
        "confirmation_improved_share_gt_worsened": bool(
            np.isfinite(conf["improved_player_share"])
            and conf["improved_player_share"] > conf["worsened_player_share"]
        ),
        "confirmation_player_bootstrap_ci_low_positive": bool(
            conf.get("player_bootstrap",{}).get("reps",0) == N_BOOT
            and np.isfinite(conf["player_bootstrap"]["ci_low"])
            and conf["player_bootstrap"]["ci_low"] > 0
        ),
    }

    qualified = bool(all(support_gates.values()) and all(qualification_gates.values()))
    disposition = (
        "PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY"
        if qualified else
        "PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_CLOSED"
    )

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "return_source_run": 39162437467,
        "return_source_artifact": 12149417234,
        "return_source_digest": "sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e",
        "health_source_run": 39154145666,
        "health_source_artifact": 12146625382,
        "health_source_digest": "sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "disposition": disposition,
        "qualified_state_only": qualified,
        "audit": audit,
        "development_2024": dev,
        "confirmation_2025": conf,
        "support_gates": support_gates,
        "qualification_gates": qualification_gates,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "performance_fields_used": 0,
        "targets_used": 0,
        "carries_used": 0,
        "receptions_used": 0,
        "yards_used": 0,
        "touchdowns_used": 0,
        "routes_inferred_from_snaps": False,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }

    rows.to_csv(
        a.out_dir / "return_state_next_role_persistence_rows.csv",
        index=False,
        float_format="%.12f",
    )
    (a.out_dir / "return_state_next_role_persistence_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
