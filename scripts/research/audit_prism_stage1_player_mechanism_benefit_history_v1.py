#!/usr/bin/env python3
"""Source-only audit for strictly-prior same-player Stage-1 mechanism benefit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "d8df3acf233103628e455a45358ff9f697657504"
PARENT_RUN = 38907459996
PARENT_ARTIFACT = 12049120604
PARENT_DIGEST = "sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b"
FAMILIES = ("QB", "RB", "WR", "TE")
TOL = 1e-12

GATES = {
    "QB": {2024: {"rows": 550, "players": 65}, 2025: {"rows": 700, "players": 70}},
    "RB": {2024: {"rows": 900, "players": 105}, 2025: {"rows": 1100, "players": 120}},
    "WR": {2024: {"rows": 1300, "players": 175}, 2025: {"rows": 1700, "players": 190}},
    "TE": {2024: {"rows": 850, "players": 105}, 2025: {"rows": 1100, "players": 110}},
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clip_prob(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=float), 1e-12, 1.0 - 1e-12)


def binary_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pp = clip_prob(p)
    yy = np.asarray(y, dtype=float)
    return -(yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp))


def tier(n: int) -> str:
    if n <= 0:
        return "TIER0_0"
    if n <= 2:
        return "TIER1_1_2"
    if n <= 5:
        return "TIER2_3_5"
    if n <= 9:
        return "TIER3_6_9"
    return "TIER4_10_PLUS"


def load_rows(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing Stage-1 parent rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "opponent", "game_key", "family", "player_id",
        "player_name_key", "y", "n_pos_prior", "k_pos_prior", "p_baseline",
        "n_player_prior", "k_player_prior", "p_candidate",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"missing parent columns: {sorted(missing)}")
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns present: {forbidden}")

    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    for c in ("y", "n_pos_prior", "k_pos_prior", "p_baseline", "n_player_prior", "k_player_prior", "p_candidate"):
        x[c] = num(x[c])
    for c in ("team", "opponent", "game_key", "family", "player_id", "player_name_key"):
        x[c] = x[c].astype("string").fillna("").str.strip()

    x = x.loc[x["season"].isin([2024, 2025]) & x["family"].isin(FAMILIES)].copy()
    if x.empty:
        raise RuntimeError("zero Stage-1 rows")
    if x["player_id"].eq("").any() or x["team"].eq("").any() or x["game_key"].eq("").any():
        raise RuntimeError("blank identity")
    if x.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate family/player/week/team identity")
    if not x["y"].isin([0, 1]).all():
        raise RuntimeError("nonbinary participation target")
    for c in ("p_baseline", "p_candidate"):
        if not ((x[c] > 0) & (x[c] < 1) & np.isfinite(x[c])).all():
            raise RuntimeError(f"{c} outside open interval")
    return x


def add_benefit(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    y = num(z["y"]).to_numpy(float)
    pb = num(z["p_baseline"]).to_numpy(float)
    pc = num(z["p_candidate"]).to_numpy(float)
    z["baseline_brier"] = np.square(y - pb)
    z["candidate_brier"] = np.square(y - pc)
    z["delta_brier"] = z["baseline_brier"] - z["candidate_brier"]
    z["baseline_logloss"] = binary_logloss(y, pb)
    z["candidate_logloss"] = binary_logloss(y, pc)
    z["delta_logloss"] = z["baseline_logloss"] - z["candidate_logloss"]
    return z


def build_prior_state(x: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    z = add_benefit(x)
    z = z.sort_values(["family", "player_id", "season", "week", "game_key", "team"]).reset_index(drop=True)
    n_prior = np.zeros(len(z), dtype=int)
    mean_brier = np.full(len(z), np.nan)
    med_brier = np.full(len(z), np.nan)
    pos_share = np.full(len(z), np.nan)
    mean_logloss = np.full(len(z), np.nan)
    violations = 0

    for _, idxs in z.groupby(["family", "player_id"], sort=False).groups.items():
        hist_b: list[float] = []
        hist_l: list[float] = []
        prev: tuple[int, int] | None = None
        for i in idxs:
            key = (int(z.at[i, "season"]), int(z.at[i, "week"]))
            if prev is not None and key <= prev:
                violations += 1
            n_prior[i] = len(hist_b)
            if hist_b:
                b = np.asarray(hist_b, dtype=float)
                l = np.asarray(hist_l, dtype=float)
                mean_brier[i] = float(b.mean())
                med_brier[i] = float(np.median(b))
                pos_share[i] = float((b > 0).mean())
                mean_logloss[i] = float(l.mean())
            hist_b.append(float(z.at[i, "delta_brier"]))
            hist_l.append(float(z.at[i, "delta_logloss"]))
            prev = key

    z["prior_mechanism_rows"] = n_prior
    z["prior_mean_delta_brier"] = mean_brier
    z["prior_median_delta_brier"] = med_brier
    z["prior_positive_share_delta_brier"] = pos_share
    z["prior_mean_delta_logloss"] = mean_logloss
    z["support_tier"] = [tier(int(n)) for n in n_prior]
    return z, violations


def summarize(z: pd.DataFrame, family: str, season: int) -> dict:
    g = z.loc[z["family"].eq(family) & z["season"].eq(season)].copy()
    ge3 = g.loc[g["prior_mechanism_rows"].ge(3)]
    req = GATES[family][season]
    gates = {
        "rows_with_ge3_prior": int(len(ge3)) >= req["rows"],
        "players_with_ge3_prior": int(ge3["player_id"].nunique()) >= req["players"],
        "teams": int(g["team"].nunique()) >= 28,
        "stable_identity_ge_99pct": float(g["player_id"].ne("").mean()) >= 0.99,
    }
    counts = g["support_tier"].value_counts().to_dict()
    return {
        "family": family,
        "season": season,
        "target_rows": int(len(g)),
        "target_players": int(g["player_id"].nunique()),
        "rows_with_ge3_prior": int(len(ge3)),
        "players_with_ge3_prior": int(ge3["player_id"].nunique()),
        "players_with_ge6_prior": int(g.loc[g["prior_mechanism_rows"].ge(6), "player_id"].nunique()),
        "players_with_ge10_prior": int(g.loc[g["prior_mechanism_rows"].ge(10), "player_id"].nunique()),
        "teams": int(g["team"].nunique()),
        "prior_history_median": float(g["prior_mechanism_rows"].median()),
        "prior_history_p25": float(g["prior_mechanism_rows"].quantile(0.25)),
        "prior_history_p75": float(g["prior_mechanism_rows"].quantile(0.75)),
        "tier_counts": {str(k): int(v) for k, v in counts.items()},
        "source_gates": gates,
        "source_ready": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        x = load_rows(args.parent_rows)
        state, violations = build_prior_state(x)
        if violations:
            raise RuntimeError(f"chronology violations={violations}")
        dupes = int(state.duplicated(["season", "week", "team", "family", "player_id"]).sum())
        if dupes:
            raise RuntimeError(f"duplicates={dupes}")

        summaries: dict[str, dict] = {}
        dispositions: dict[str, str] = {}
        for family in FAMILIES:
            summaries[family] = {}
            ready = True
            for season in (2024, 2025):
                rec = summarize(state, family, season)
                summaries[family][str(season)] = rec
                ready = ready and rec["source_ready"]
            dispositions[family] = (
                f"{family}_STAGE1_PLAYER_BENEFIT_HISTORY_SOURCE_READY"
                if ready else
                f"{family}_STAGE1_PLAYER_BENEFIT_HISTORY_SOURCE_BLOCKED"
            )

        nready = sum(v.endswith("_READY") for v in dispositions.values())
        overall = (
            "PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL"
            if nready == len(FAMILIES)
            else "PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_PARTIAL"
            if nready
            else "PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_BLOCKED"
        )

        state[
            [
                "season", "week", "team", "opponent", "game_key", "family",
                "player_id", "player_name_key", "y", "p_baseline", "p_candidate",
                "n_player_prior", "prior_mechanism_rows", "support_tier",
                "prior_mean_delta_brier", "prior_median_delta_brier",
                "prior_positive_share_delta_brier", "prior_mean_delta_logloss",
            ]
        ].to_csv(args.out_dir / "stage1_player_mechanism_benefit_strict_prior_state.csv", index=False)

        flat = []
        for family in FAMILIES:
            for season in (2024, 2025):
                rec = summaries[family][str(season)]
                row = {k: v for k, v in rec.items() if k not in {"tier_counts", "source_gates"}}
                for k, v in rec["tier_counts"].items():
                    row[f"tier_{k}"] = v
                for k, v in rec["source_gates"].items():
                    row[f"gate_{k}"] = v
                flat.append(row)
        pd.DataFrame(flat).to_csv(args.out_dir / "stage1_player_mechanism_benefit_source_support.csv", index=False)

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "families": list(FAMILIES),
            "summaries": summaries,
            "family_dispositions": dispositions,
            "disposition": overall,
            "chronology_violations": 0,
            "duplicate_identities": 0,
            "personalization_rule_scored": False,
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
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "disposition": "PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_INTEGRITY_FAILURE",
            "error": str(exc),
            "personalization_rule_scored": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "stage1_player_mechanism_benefit_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "stage1_player_mechanism_benefit_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
