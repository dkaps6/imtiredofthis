"""RB Lane A -- gate scoring (transition-gated allocation V1).

Implements the frozen "Protected cohorts and gates" section of
``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V1_PLAN.md``
(Amendments 1-3). This is the only module in this candidate chain that
touches actual (postgame) outcomes -- by design, per the plan's own
required execution order, everything upstream (Gate 0, both comparator
reconstructions, the candidate mechanism itself) is built and verified
outcome-blind first.

Adequacy floor: ``n>=30`` (Amendment 1). Protected-cohort bootstrap:
paired player-clustered (`BOOT_N=10000`, `BOOTSTRAP_GATE=0.90`, reusing the
repo's own M89/M90 convention) plus a dependence-aware crossed player x game
bootstrap, adapted from
``scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py::
crossed_player_game_bootstrap_probability`` (same mechanics, MAE delta in
place of CRPS delta, since Lane A's decisive endpoint is rushing-yard MAE,
not a full predictive distribution). Both bootstraps required >=0.90;
neither can rescue the other's failure (Amendment 1 point #7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADEQUACY_FLOOR = 30
BOOT_N = 10_000
BOOTSTRAP_GATE = 0.90
LANE_A_BOOTSTRAP_SEED = 616  # disclosed, study-specific (PR #616), not silently reused from PD2/M89


def adequacy_check(scored_population: pd.DataFrame, cohort_masks: dict[str, pd.Series]) -> dict:
    """Amendment 1: `n>=30` adequacy floor on the overall scored V1 transition
    subpopulation and every required protected cohort. Fail-closed --
    `INSUFFICIENT_EVIDENCE` on any cohort below the floor prevents
    `QUALIFIED` regardless of how any scoreable gate performs.
    """
    overall_n = int(len(scored_population))
    cohorts = {}
    passed = overall_n >= ADEQUACY_FLOOR
    for name, mask in cohort_masks.items():
        n = int(mask.sum())
        cohorts[name] = {"n": n, "adequate": n >= ADEQUACY_FLOOR}
        passed = passed and cohorts[name]["adequate"]
    return {
        "disposition": "ADEQUATE" if passed else "INSUFFICIENT_EVIDENCE",
        "overall_n": overall_n,
        "overall_adequate": overall_n >= ADEQUACY_FLOOR,
        "cohorts": cohorts,
    }


def compute_mae_delta_rows(
    scored_rows: pd.DataFrame,
    *,
    candidate_col: str,
    comparator_col: str,
    actual_col: str,
    player_key_col: str = "player_clean_key",
    game_key_col: str = "game_key",
) -> pd.DataFrame:
    """`delta_i = |candidate_i - actual_i| - |comparator_i - actual_i|` per
    scored row. Negative delta means the candidate's absolute error is
    smaller than the comparator's on that row -- the bootstrap's "win"
    criterion is a negative mean delta.
    """
    required = {candidate_col, comparator_col, actual_col, player_key_col, game_key_col}
    missing = required - set(scored_rows.columns)
    if missing:
        raise RuntimeError(f"compute_mae_delta_rows: missing columns {sorted(missing)}")
    out = scored_rows[[player_key_col, game_key_col, candidate_col, comparator_col, actual_col]].copy()
    candidate_err = (
        pd.to_numeric(out[candidate_col], errors="coerce") - pd.to_numeric(out[actual_col], errors="coerce")
    ).abs()
    comparator_err = (
        pd.to_numeric(out[comparator_col], errors="coerce") - pd.to_numeric(out[actual_col], errors="coerce")
    ).abs()
    out["delta"] = candidate_err - comparator_err
    out = out.rename(columns={player_key_col: "player_key", game_key_col: "game_key"})
    return out.dropna(subset=["player_key", "game_key", "delta"])


def player_cluster_bootstrap_probability(
    delta_rows: pd.DataFrame, *, boot_n: int = BOOT_N, seed: int = LANE_A_BOOTSTRAP_SEED
) -> float:
    """Paired player-clustered bootstrap (M89/M90 `BOOT_N=10000` convention):
    resample players with replacement, compute the pooled mean delta under
    each resample, report the fraction of resamples with a negative mean
    delta (candidate beats comparator).
    """
    z = delta_rows.dropna(subset=["player_key", "delta"])
    if z.empty:
        return float("nan")
    grouped = z.groupby("player_key", sort=True)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    n_players = len(grouped)
    wins = 0
    for _ in range(boot_n):
        pick = rng.integers(0, n_players, size=n_players)
        denom = float(counts[pick].sum())
        mean_delta = float(sums[pick].sum() / denom) if denom > 0 else np.nan
        wins += int(np.isfinite(mean_delta) and mean_delta < 0.0)
    return float(wins / boot_n)


def crossed_player_game_bootstrap_probability(
    delta_rows: pd.DataFrame, *, boot_n: int = BOOT_N, seed: int = LANE_A_BOOTSTRAP_SEED
) -> float:
    """Dependence-aware crossed player x game bootstrap, adapted unchanged
    (mechanics only) from `evaluate_rb_pd2_yard_difficulty_mc_width_v1.py::
    crossed_player_game_bootstrap_probability` -- multiple RB rows on the
    same team-game are mechanically coupled by the conservation/reallocation
    identity, so player-only clustering under-accounts for that dependence.
    Additive to `player_cluster_bootstrap_probability` and cannot rescue its
    failure (Amendment 1 point #7): both required `>=0.90`.
    """
    z = delta_rows.dropna(subset=["player_key", "game_key", "delta"])
    if z.empty:
        return float("nan")
    players = np.sort(z["player_key"].unique())
    games = np.sort(z["game_key"].unique())
    n_players, n_games = len(players), len(games)
    player_idx = {p: i for i, p in enumerate(players)}
    game_idx = {g: i for i, g in enumerate(games)}
    row_player = z["player_key"].map(player_idx).to_numpy()
    row_game = z["game_key"].map(game_idx).to_numpy()
    deltas = z["delta"].to_numpy(float)
    rng = np.random.default_rng(seed)
    wins = 0
    valid = 0
    while valid < boot_n:
        player_mult = np.bincount(rng.integers(0, n_players, size=n_players), minlength=n_players)
        game_mult = np.bincount(rng.integers(0, n_games, size=n_games), minlength=n_games)
        weight = player_mult[row_player] * game_mult[row_game]
        denom = float(weight.sum())
        if denom <= 0.0:
            continue
        mean_delta = float(np.dot(weight, deltas) / denom)
        wins += int(np.isfinite(mean_delta) and mean_delta < 0.0)
        valid += 1
    return float(wins / boot_n)


def bootstrap_gate_report(delta_rows: pd.DataFrame, *, boot_n: int = BOOT_N, seed: int = LANE_A_BOOTSTRAP_SEED) -> dict:
    """Both bootstraps, both required `>=0.90` (Amendment 1 point #7)."""
    player_p = player_cluster_bootstrap_probability(delta_rows, boot_n=boot_n, seed=seed)
    crossed_p = crossed_player_game_bootstrap_probability(delta_rows, boot_n=boot_n, seed=seed)
    passed = (
        np.isfinite(player_p) and player_p >= BOOTSTRAP_GATE
        and np.isfinite(crossed_p) and crossed_p >= BOOTSTRAP_GATE
    )
    return {
        "disposition": "BOOTSTRAP_GATE_PASS" if passed else "BOOTSTRAP_GATE_FAILURE",
        "player_cluster_bootstrap_probability": player_p,
        "crossed_player_game_bootstrap_probability": crossed_p,
        "gate_threshold": BOOTSTRAP_GATE,
        "boot_n": boot_n,
    }


def per_season_nonregression_check(delta_rows_by_rotation: dict[int, pd.DataFrame]) -> dict:
    """Amendment 1: the candidate must not regress the promotion comparator
    in either rotation individually -- pooled mean delta per rotation must
    be strictly negative (candidate MAE < comparator MAE), reported
    per-rotation, not just pooled.
    """
    out = {}
    all_nonregressed = True
    for rotation, rows in delta_rows_by_rotation.items():
        mean_delta = float(pd.to_numeric(rows["delta"], errors="coerce").mean()) if len(rows) else float("nan")
        nonregressed = np.isfinite(mean_delta) and mean_delta < 0.0
        all_nonregressed = all_nonregressed and nonregressed
        out[str(rotation)] = {"mean_delta": mean_delta, "nonregressed": nonregressed, "n": int(len(rows))}
    return {
        "disposition": "PER_SEASON_NONREGRESSION_PASS" if all_nonregressed else "PER_SEASON_NONREGRESSION_FAILURE",
        "rotations": out,
    }


def whole_season_deployable_safety_check(
    deployable_rows: pd.DataFrame,
    *,
    deployable_col: str = "deployable_candidate_rush_yards",
    comparator_col: str = "promotion_rush_yards",
    actual_col: str = "actual_rush_yards",
) -> dict:
    """Amendment 4: the deployable arm (scored + stable rows together) must
    be non-worse than the promotion comparator across the WHOLE season, not
    just on scored-transition rows -- guards against the reallocation
    mechanism quietly degrading overall accuracy even if it wins narrowly on
    the transition subpopulation.
    """
    required = {deployable_col, comparator_col, actual_col}
    missing = required - set(deployable_rows.columns)
    if missing:
        raise RuntimeError(f"whole_season_deployable_safety_check: missing columns {sorted(missing)}")
    deployable_mae = (
        pd.to_numeric(deployable_rows[deployable_col], errors="coerce")
        - pd.to_numeric(deployable_rows[actual_col], errors="coerce")
    ).abs().mean()
    comparator_mae = (
        pd.to_numeric(deployable_rows[comparator_col], errors="coerce")
        - pd.to_numeric(deployable_rows[actual_col], errors="coerce")
    ).abs().mean()
    non_worse = bool(np.isfinite(deployable_mae) and np.isfinite(comparator_mae) and deployable_mae <= comparator_mae)
    return {
        "disposition": "WHOLE_SEASON_SAFETY_PASS" if non_worse else "WHOLE_SEASON_SAFETY_FAILURE",
        "deployable_mae": float(deployable_mae) if np.isfinite(deployable_mae) else None,
        "comparator_mae": float(comparator_mae) if np.isfinite(comparator_mae) else None,
        "n": int(len(deployable_rows)),
    }
