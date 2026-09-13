#!/usr/bin/env python3
"""Frozen RB-PD2 yard-difficulty Monte Carlo width experiment.

Research only. This evaluator changes no production code and consumes no sportsbook
inputs. It reuses the exact PR #556 current-route mean lineage, pairs it with
checksum-verified historical Monte Carlo arrays, and grades the preregistered
mean-neutral yard-width candidate from RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.evaluate_rb_pd2_multiseason_current_route_v1 import (
    MIN_PRIOR,
    RB_POS,
    TARGET_SEASONS,
    build_panel,
    build_wf,
    key,
    num,
    read,
    team,
    verify_m95q_parity,
)

ITERATIONS = 2000
REF_MIN = 100
WIDTH_CAP = 0.30
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 42027
TAIL_THRESHOLDS = (50.0, 75.0, 100.0)
EXPECTED_PARENT_ROWS = 5607
EXPECTED_PARENT_SCOREABLE = 4652
IDENTITY = ["season", "week", "team", "opponent", "player_key", "market"]


def strict_prior_difficulty_scores(wf: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen WR-R3-style strictly-prior empirical difficulty score."""
    out = wf.copy().sort_values(["season", "week", "player_key"], kind="stable").reset_index(drop=True)
    out["difficulty_score"] = np.nan
    out["difficulty_reference_n"] = 0
    out["difficulty_reference_max_ord"] = np.nan
    ref: list[float] = []
    last_ref_ord = np.nan
    for (season, week), idx in out.groupby(["season", "week"], sort=True).groups.items():
        ids = list(idx)
        ref_n = len(ref)
        for i in ids:
            out.at[i, "difficulty_reference_n"] = ref_n
            out.at[i, "difficulty_reference_max_ord"] = last_ref_ord
            if int(out.at[i, "prior_games"]) < MIN_PRIOR or ref_n < REF_MIN:
                continue
            value = float(pd.to_numeric(pd.Series([out.at[i, "prior8_yard_mae"]]), errors="coerce").iloc[0])
            if np.isfinite(value):
                out.at[i, "difficulty_score"] = bisect.bisect_right(ref, value) / ref_n
        # Same-week rows enter only after every row in the week has been scored.
        for i in ids:
            if int(out.at[i, "prior_games"]) < MIN_PRIOR:
                continue
            value = float(pd.to_numeric(pd.Series([out.at[i, "prior8_yard_mae"]]), errors="coerce").iloc[0])
            if np.isfinite(value):
                bisect.insort(ref, value)
        last_ref_ord = int(season) * 100 + int(week)
    return out


def width_multiplier(score: float) -> float:
    if not np.isfinite(score):
        return 1.0
    return float(1.0 + WIDTH_CAP * np.clip((float(score) - 0.50) / 0.50, 0.0, 1.0))


def widen_mean_neutral(draws: np.ndarray, multiplier: float) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError("invalid baseline draw array")
    mu = float(np.mean(x))
    raw = np.maximum(0.0, mu + float(multiplier) * (x - mu))
    raw_mean = float(np.mean(raw))
    if not np.isfinite(raw_mean) or raw_mean <= 0.0:
        raise RuntimeError("candidate width transform produced invalid mean")
    out = raw * (mu / raw_mean)
    if not np.isfinite(out).all() or (out < 0).any():
        raise RuntimeError("candidate width transform produced invalid draws")
    return out


def empirical_crps(draws: np.ndarray, actual: float) -> float:
    """CRPS for an equally weighted empirical distribution."""
    x = np.sort(np.asarray(draws, dtype=float))
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(actual):
        return np.nan
    n = len(x)
    coeff = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    return float(np.mean(np.abs(x - float(actual))) - np.dot(coeff, x) / (n * n))


def row_distribution_metrics(draws: np.ndarray, actual: float, prefix: str) -> dict[str, float]:
    x = np.asarray(draws, dtype=float)
    q05, q10, q90, q95 = np.quantile(x, [0.05, 0.10, 0.90, 0.95], method="linear")
    out: dict[str, float] = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_crps": empirical_crps(x, actual),
        f"{prefix}_cover80": float(q10 <= actual <= q90),
        f"{prefix}_cover90": float(q05 <= actual <= q95),
    }
    for threshold in TAIL_THRESHOLDS:
        tag = int(threshold)
        p = float(np.mean(x >= threshold))
        y = float(actual >= threshold)
        out[f"{prefix}_p_ge_{tag}"] = p
        out[f"{prefix}_brier_ge_{tag}"] = float((p - y) ** 2)
    return out


def pooled_metrics(rows: pd.DataFrame, prefix: str) -> dict[str, float]:
    if rows.empty:
        return {"n": 0, "crps": np.nan, "coverage80": np.nan, "coverage80_gap": np.nan,
                "coverage90": np.nan, "coverage90_gap": np.nan,
                "brier_ge_50": np.nan, "brier_ge_75": np.nan, "brier_ge_100": np.nan}
    c80 = float(pd.to_numeric(rows[f"{prefix}_cover80"], errors="coerce").mean())
    c90 = float(pd.to_numeric(rows[f"{prefix}_cover90"], errors="coerce").mean())
    return {
        "n": int(len(rows)),
        "crps": float(pd.to_numeric(rows[f"{prefix}_crps"], errors="coerce").mean()),
        "coverage80": c80,
        "coverage80_gap": abs(c80 - 0.80),
        "coverage90": c90,
        "coverage90_gap": abs(c90 - 0.90),
        "brier_ge_50": float(pd.to_numeric(rows[f"{prefix}_brier_ge_50"], errors="coerce").mean()),
        "brier_ge_75": float(pd.to_numeric(rows[f"{prefix}_brier_ge_75"], errors="coerce").mean()),
        "brier_ge_100": float(pd.to_numeric(rows[f"{prefix}_brier_ge_100"], errors="coerce").mean()),
    }


def improvement_pct(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or baseline <= 0:
        return np.nan
    return float(100.0 * (baseline - candidate) / baseline)


def player_cluster_bootstrap_probability(rows: pd.DataFrame) -> float:
    """Paired player-cluster bootstrap P(candidate pooled CRPS < baseline)."""
    z = rows[["player_key", "baseline_crps", "candidate_crps"]].copy()
    z["delta"] = pd.to_numeric(z.candidate_crps, errors="coerce") - pd.to_numeric(z.baseline_crps, errors="coerce")
    z = z.dropna(subset=["player_key", "delta"])
    grouped = z.groupby("player_key", sort=True).delta.agg(["sum", "count"])
    if grouped.empty:
        return np.nan
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    wins = 0
    n_players = len(grouped)
    for _ in range(BOOTSTRAP_REPS):
        pick = rng.integers(0, n_players, size=n_players)
        denom = float(counts[pick].sum())
        mean_delta = float(sums[pick].sum() / denom) if denom > 0 else np.nan
        wins += int(np.isfinite(mean_delta) and mean_delta < 0.0)
    return float(wins / BOOTSTRAP_REPS)


def build_yard_lineage(component_root: Path, parent_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in TARGET_SEASONS:
        x = read(component_root / str(season) / "component_predictions.csv")
        x["position"] = x.position.fillna("").astype(str).str.upper().str.strip()
        x["market"] = x.market.fillna("").astype(str).str.lower().str.strip()
        x = x.loc[x.position.isin(RB_POS) & x.market.eq("rush_yards") & num(x.week).between(1, 18)].copy()
        x["season"] = num(x.season).astype(int)
        x["week"] = num(x.week).astype(int)
        x["team"] = x.team.map(team)
        x["opponent"] = x.opponent.map(team)
        x["player_key"] = x.get("player_clean_key", x.player).map(key)
        keep = ["season", "week", "team", "opponent", "player_key", "mc_proj"]
        if x.duplicated(["season", "week", "team", "player_key"]).any():
            raise RuntimeError(f"duplicate rush-yard component identity target={season}")
        rows.append(x[keep])
    raw = pd.concat(rows, ignore_index=True)
    panel = parent_panel[["season", "week", "team", "player_key", "pred_yard", "actual_yard"]].copy()
    out = panel.merge(raw, on=["season", "week", "team", "player_key"], how="left", validate="one_to_one")
    if out[["opponent", "mc_proj"]].isna().any().any():
        raise RuntimeError("parent panel missing rush-yard component lineage")
    out.rename(columns={"pred_yard": "ensemble_proj", "actual_yard": "actual"}, inplace=True)
    out["market"] = "rush_yards"
    return out


def load_distribution_index(root: Path) -> pd.DataFrame:
    parts = []
    for meta_path in sorted(root.rglob("*_metadata.csv")):
        m = pd.read_csv(meta_path, low_memory=False)
        m.columns = [str(c).strip().lower() for c in m.columns]
        required = {"season", "week", "team", "opponent", "player_clean_key", "market", "array_key", "npz_file", "mc_mean", "draws"}
        if not required.issubset(m.columns):
            raise RuntimeError(f"distribution metadata missing columns {sorted(required-set(m.columns))}: {meta_path}")
        m["season"] = pd.to_numeric(m.season, errors="raise").astype(int)
        m["week"] = pd.to_numeric(m.week, errors="raise").astype(int)
        m["team"] = m.team.map(team)
        m["opponent"] = m.opponent.map(team)
        m["player_key"] = m.player_clean_key.map(key)
        m["market"] = m.market.astype(str).str.lower().str.strip()
        m["npz_path"] = m.npz_file.map(lambda name: str(meta_path.parent / str(name)))
        parts.append(m[["season", "week", "team", "opponent", "player_key", "market", "array_key", "npz_path", "mc_mean", "draws"]])
    if not parts:
        raise RuntimeError(f"no distribution metadata found under {root}")
    out = pd.concat(parts, ignore_index=True)
    out = out.loc[out.season.isin(TARGET_SEASONS) & out.market.eq("rush_yards")].copy()
    if out.duplicated(IDENTITY).any():
        raise RuntimeError("duplicate distribution identity")
    return out


def attach_features_and_distributions(
    lineage: pd.DataFrame,
    scored_wf: pd.DataFrame,
    dist_index: pd.DataFrame,
) -> pd.DataFrame:
    features = scored_wf[["season", "week", "team", "player_key", "prior_games", "prior8_yard_mae",
                          "last_prior_ord", "difficulty_score", "difficulty_reference_n",
                          "difficulty_reference_max_ord"]].copy()
    out = lineage.merge(features, on=["season", "week", "team", "player_key"], how="left", validate="one_to_one")
    out = out.merge(dist_index, on=IDENTITY, how="left", validate="one_to_one")
    return out


def grade_distributions(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    eligible = rows.loc[
        (pd.to_numeric(rows.prior_games, errors="coerce") >= MIN_PRIOR)
        & pd.to_numeric(rows.difficulty_score, errors="coerce").notna()
    ].copy()
    if eligible.empty:
        raise RuntimeError("no primary eligible rows after strict-prior reference floor")
    required = ["array_key", "npz_path", "mc_mean", "draws", "mc_proj", "ensemble_proj", "actual"]
    if eligible[required].isna().any().any():
        bad = eligible.loc[eligible[required].isna().any(axis=1), ["season", "week", "team", "player_key"]].head(10).to_dict("records")
        raise RuntimeError(f"eligible rows missing distribution lineage: {bad}")

    caches: dict[str, object] = {}
    records = []
    max_raw_delta = 0.0
    max_meta_delta = 0.0
    max_baseline_mean_delta = 0.0
    max_candidate_mean_delta = 0.0
    for r in eligible.itertuples(index=False):
        path = str(r.npz_path)
        if path not in caches:
            caches[path] = np.load(path, allow_pickle=False)
        archive = caches[path]
        arr = np.asarray(archive[str(r.array_key)], dtype=float)
        if len(arr) != ITERATIONS:
            raise RuntimeError(f"draw count mismatch {r.season} W{r.week} {r.player_key}: {len(arr)}")
        raw_mean = float(np.mean(arr))
        source_mc = float(r.mc_proj)
        meta_mc = float(r.mc_mean)
        max_raw_delta = max(max_raw_delta, abs(raw_mean - source_mc))
        max_meta_delta = max(max_meta_delta, abs(raw_mean - meta_mc))
        if not np.isfinite(source_mc) or source_mc <= 0.0:
            raise RuntimeError(f"eligible row has non-positive mc_proj: {r.season} W{r.week} {r.player_key}")
        target_mean = float(r.ensemble_proj)
        baseline = arr * max(0.0, target_mean / source_mc)
        baseline_mean = float(np.mean(baseline))
        max_baseline_mean_delta = max(max_baseline_mean_delta, abs(baseline_mean - target_mean))
        mult = width_multiplier(float(r.difficulty_score))
        candidate = widen_mean_neutral(baseline, mult)
        candidate_mean = float(np.mean(candidate))
        max_candidate_mean_delta = max(max_candidate_mean_delta, abs(candidate_mean - baseline_mean))
        rec = {c: getattr(r, c) for c in ["season", "week", "team", "opponent", "player_key", "market"]}
        rec.update({
            "actual": float(r.actual),
            "mc_proj": source_mc,
            "ensemble_proj": target_mean,
            "prior_games": int(r.prior_games),
            "prior8_yard_mae": float(r.prior8_yard_mae),
            "last_prior_ord": float(r.last_prior_ord),
            "difficulty_score": float(r.difficulty_score),
            "difficulty_reference_n": int(r.difficulty_reference_n),
            "difficulty_reference_max_ord": float(r.difficulty_reference_max_ord),
            "width_mult": mult,
            "raw_mc_mean": raw_mean,
        })
        rec.update(row_distribution_metrics(baseline, float(r.actual), "baseline"))
        rec.update(row_distribution_metrics(candidate, float(r.actual), "candidate"))
        records.append(rec)
    for archive in caches.values():
        archive.close()
    casebook = pd.DataFrame(records)
    checks = {
        "max_abs_raw_mc_vs_source_mc_proj": float(max_raw_delta),
        "max_abs_raw_mc_vs_metadata_mc_mean": float(max_meta_delta),
        "max_abs_baseline_mean_vs_ensemble_proj": float(max_baseline_mean_delta),
        "max_abs_candidate_vs_baseline_mean": float(max_candidate_mean_delta),
    }
    return casebook, checks


def evaluate_gates(casebook: pd.DataFrame, checks: dict[str, float], parent_source_rows: int,
                   parent_scoreable_rows: int, parity: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pooled_base = pooled_metrics(casebook, "baseline")
    pooled_cand = pooled_metrics(casebook, "candidate")
    pooled_crps_improvement = improvement_pct(pooled_base["crps"], pooled_cand["crps"])
    bootstrap_p = player_cluster_bootstrap_probability(casebook)

    # difficulty_score is itself a strictly-prior percentile. The global Q75 of
    # those scores implements the preregistered "top quartile among primary rows" slice.
    high_threshold = float(pd.to_numeric(casebook.difficulty_score, errors="coerce").quantile(0.75))
    high = casebook.loc[pd.to_numeric(casebook.difficulty_score, errors="coerce") >= high_threshold].copy()
    high_base = pooled_metrics(high, "baseline")
    high_cand = pooled_metrics(high, "candidate")
    high_crps_improvement = improvement_pct(high_base["crps"], high_cand["crps"])

    by_season_rows = []
    pooled_better_seasons = 0
    high_better_adequate_seasons = 0
    for season in TARGET_SEASONS:
        q = casebook.loc[casebook.season.eq(season)]
        qb, qc = pooled_metrics(q, "baseline"), pooled_metrics(q, "candidate")
        qh = high.loc[high.season.eq(season)]
        qhb, qhc = pooled_metrics(qh, "baseline"), pooled_metrics(qh, "candidate")
        pooled_improved = bool(np.isfinite(qb["crps"]) and np.isfinite(qc["crps"]) and qc["crps"] < qb["crps"])
        high_adequate = int(len(qh)) >= 100
        high_improved = bool(high_adequate and np.isfinite(qhb["crps"]) and np.isfinite(qhc["crps"]) and qhc["crps"] < qhb["crps"])
        pooled_better_seasons += int(pooled_improved)
        high_better_adequate_seasons += int(high_improved)
        by_season_rows.append({
            "season": season,
            "rows": int(len(q)),
            "baseline_crps": qb["crps"],
            "candidate_crps": qc["crps"],
            "crps_improvement_pct": improvement_pct(qb["crps"], qc["crps"]),
            "pooled_crps_improved": pooled_improved,
            "high_rows": int(len(qh)),
            "high_baseline_crps": qhb["crps"],
            "high_candidate_crps": qhc["crps"],
            "high_crps_improvement_pct": improvement_pct(qhb["crps"], qhc["crps"]),
            "high_adequate": high_adequate,
            "high_crps_improved": high_improved,
        })
    by_season = pd.DataFrame(by_season_rows)

    baseline_point_mae = float(np.mean(np.abs(pd.to_numeric(casebook.baseline_mean) - pd.to_numeric(casebook.actual))))
    candidate_point_mae = float(np.mean(np.abs(pd.to_numeric(casebook.candidate_mean) - pd.to_numeric(casebook.actual))))
    point_mae_delta = candidate_point_mae - baseline_point_mae

    current_ord = pd.to_numeric(casebook.season) * 100 + pd.to_numeric(casebook.week)
    last_prior = pd.to_numeric(casebook.last_prior_ord, errors="coerce")
    ref_prior = pd.to_numeric(casebook.difficulty_reference_max_ord, errors="coerce")
    recent = by_season.set_index("season")
    gates = {
        "A_m95q_source_parity": bool(parity.get("m91_universe_2024_pass") and parity.get("downstream_2024_parity_pass") and parity.get("m95q_disposition") == "M95Q_EXPANDED_PANEL_READY"),
        "A_parent_panel_matches_556": bool(parent_source_rows == EXPECTED_PARENT_ROWS and parent_scoreable_rows == EXPECTED_PARENT_SCOREABLE),
        "A_prior_season_only_ensemble_weights": bool(checks.get("prior_season_weight_lineage_pass", False)),
        "A_target_seasons_exact_no_2025": bool(set(casebook.season.astype(int).unique()) == set(TARGET_SEASONS) and not casebook.season.eq(2025).any()),
        "A_raw_mc_reproduces_source": bool(checks["max_abs_raw_mc_vs_source_mc_proj"] <= 1e-8),
        "A_metadata_mc_reproduces_array": bool(checks["max_abs_raw_mc_vs_metadata_mc_mean"] <= 1e-8),
        "A_baseline_mean_matches_ensemble": bool(checks["max_abs_baseline_mean_vs_ensemble_proj"] <= 1e-8),
        "A_strict_prior_feature_history": bool(last_prior.notna().all() and (last_prior < current_ord).all()),
        "A_strict_prior_percentile_reference": bool(ref_prior.notna().all() and (ref_prior < current_ord).all() and (pd.to_numeric(casebook.difficulty_reference_n) >= REF_MIN).all()),
        "A_zero_pre2021_history_panel": bool(checks.get("history_panel_min_season") == 2021),
        "A_zero_sportsbook_inputs": True,
        "A_production_changed_false": True,
        "B_candidate_mean_neutral": bool(checks["max_abs_candidate_vs_baseline_mean"] <= 1e-8),
        "B_point_mae_identical": bool(abs(point_mae_delta) <= 1e-8),
        "B_no_carry_mean_ypc_allocation_change": True,
        "C_pooled_crps_improvement_ge_0_5pct": bool(np.isfinite(pooled_crps_improvement) and pooled_crps_improvement >= 0.5),
        "C_player_cluster_bootstrap_p_ge_0_95": bool(np.isfinite(bootstrap_p) and bootstrap_p >= 0.95),
        "D_high_crps_improvement_ge_1pct": bool(np.isfinite(high_crps_improvement) and high_crps_improvement >= 1.0),
        "D_high_coverage80_gap_strictly_better": bool(high_cand["coverage80_gap"] < high_base["coverage80_gap"]),
        "D_high_coverage90_gap_strictly_better": bool(high_cand["coverage90_gap"] < high_base["coverage90_gap"]),
        "E_pooled_coverage80_nonworse": bool(pooled_cand["coverage80_gap"] <= pooled_base["coverage80_gap"]),
        "E_pooled_coverage90_nonworse": bool(pooled_cand["coverage90_gap"] <= pooled_base["coverage90_gap"]),
        "E_at_least_one_pooled_coverage_gap_strictly_better": bool((pooled_cand["coverage80_gap"] < pooled_base["coverage80_gap"]) or (pooled_cand["coverage90_gap"] < pooled_base["coverage90_gap"])),
        "F_brier100_strictly_better": bool(pooled_cand["brier_ge_100"] < pooled_base["brier_ge_100"]),
        "F_brier50_nonworse": bool(pooled_cand["brier_ge_50"] <= pooled_base["brier_ge_50"]),
        "F_brier75_nonworse": bool(pooled_cand["brier_ge_75"] <= pooled_base["brier_ge_75"]),
        "G_pooled_crps_improves_3_of_4": bool(pooled_better_seasons >= 3),
        "G_2023_2024_pooled_crps_nonworse": bool(recent.loc[2023, "candidate_crps"] <= recent.loc[2023, "baseline_crps"] and recent.loc[2024, "candidate_crps"] <= recent.loc[2024, "baseline_crps"]),
        "G_high_crps_improves_3_of_4_adequate": bool(high_better_adequate_seasons >= 3),
    }
    gate_frame = pd.DataFrame([{"gate": name, "pass": bool(value)} for name, value in gates.items()])
    integrity_names = [name for name in gates if name.startswith("A_")]
    integrity_pass = all(gates[name] for name in integrity_names)
    all_pass = all(gates.values())
    if not integrity_pass:
        disposition = "RB_YARD_DIFFICULTY_WIDTH_INTEGRITY_FAILURE"
    elif all_pass:
        disposition = "RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED"
    else:
        disposition = "NO_ACTIONABLE_RB_YARD_DIFFICULTY_MC_WIDTH"

    result = {
        "migration": "RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1",
        "target_seasons": TARGET_SEASONS,
        "iterations": ITERATIONS,
        "history_window": 8,
        "minimum_prior_games": MIN_PRIOR,
        "minimum_prior_reference_rows": REF_MIN,
        "width_cap": WIDTH_CAP,
        "primary_rows": int(len(casebook)),
        "primary_players": int(casebook.player_key.nunique()),
        "high_difficulty_threshold": high_threshold,
        "high_difficulty_rows": int(len(high)),
        "parent_source_rows": int(parent_source_rows),
        "parent_scoreable_rows": int(parent_scoreable_rows),
        "baseline_point_mae": baseline_point_mae,
        "candidate_point_mae": candidate_point_mae,
        "point_mae_delta": point_mae_delta,
        "pooled_baseline": pooled_base,
        "pooled_candidate": pooled_cand,
        "pooled_crps_improvement_pct": pooled_crps_improvement,
        "cluster_bootstrap_p_candidate_crps_lower": bootstrap_p,
        "high_baseline": high_base,
        "high_candidate": high_cand,
        "high_crps_improvement_pct": high_crps_improvement,
        "pooled_crps_better_seasons": int(pooled_better_seasons),
        "high_crps_better_adequate_seasons": int(high_better_adequate_seasons),
        "integrity_checks": checks,
        "source_parity": parity,
        "sportsbook_inputs_used": False,
        "production_changed": False,
        "all_gates_pass": bool(all_pass),
        "gates": gates,
        "disposition": disposition,
    }
    return gate_frame, by_season, result


def run(component_root: Path, parity_root: Path, distribution_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    parity = verify_m95q_parity(parity_root)
    parent_panel, weights = build_panel(component_root)
    wf = build_wf(parent_panel)
    parent_scoreable = int((pd.to_numeric(wf.prior_games, errors="coerce") >= MIN_PRIOR).sum())
    scored = strict_prior_difficulty_scores(wf)
    lineage = build_yard_lineage(component_root, parent_panel)
    dist_index = load_distribution_index(distribution_root)
    joined = attach_features_and_distributions(lineage, scored, dist_index)
    casebook, checks = grade_distributions(joined)

    fit_season = pd.to_numeric(weights.get("fit_season"), errors="coerce")
    target_season = pd.to_numeric(weights.get("target_season"), errors="coerce")
    checks["prior_season_weight_lineage_pass"] = bool(
        len(weights) > 0
        and fit_season.notna().all()
        and target_season.notna().all()
        and ((target_season - fit_season) == 1).all()
        and set(target_season.astype(int).unique()) == set(TARGET_SEASONS)
    )
    checks["history_panel_min_season"] = int(pd.to_numeric(wf.season, errors="raise").min())
    checks["history_panel_max_season"] = int(pd.to_numeric(wf.season, errors="raise").max())

    gates, by_season, result = evaluate_gates(casebook, checks, len(parent_panel), parent_scoreable, parity)

    parent_panel.to_csv(out_dir / "rb_pd2_yard_width_parent_panel.csv", index=False)
    weights.to_csv(out_dir / "rb_pd2_yard_width_weights.csv", index=False)
    scored.to_csv(out_dir / "rb_pd2_yard_width_strict_prior_features.csv", index=False)
    casebook.to_csv(out_dir / "rb_pd2_yard_width_casebook.csv", index=False)
    gates.to_csv(out_dir / "rb_pd2_yard_width_gates.csv", index=False)
    by_season.to_csv(out_dir / "rb_pd2_yard_width_by_season.csv", index=False)
    (out_dir / "rb_pd2_yard_width_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component-root", type=Path, required=True)
    ap.add_argument("--parity-root", type=Path, required=True)
    ap.add_argument("--distribution-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    try:
        result = run(a.component_root, a.parity_root, a.distribution_root, a.out_dir)
    except Exception as exc:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "migration": "RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1",
            "disposition": "RB_YARD_DIFFICULTY_WIDTH_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "sportsbook_inputs_used": False,
            "production_changed": False,
        }
        (a.out_dir / "rb_pd2_yard_width_result.json").write_text(json.dumps(failure, indent=2, sort_keys=True))
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
