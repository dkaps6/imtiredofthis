#!/usr/bin/env python3
"""Evaluate frozen TE named-player allocation uncertainty on marginal rec-yard draws.

Research only. This consumes:
- the confirmed allocation-only TE difficulty rows;
- rebuilt historical TE-R5P production-order rec-yard arrays.

It never changes means, target shares, team mass, teammates, or production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION = "TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1"
TARGET_SEASONS = (2024, 2025)
DRAW_COUNT = 2000
BOOT_REPS = 5000
PLAYER_SEED = 2026101501
GAME_SEED = 2026101502
MEAN_TOL = 1e-10
REC_MARKETS = {
    "player_reception_yds",
    "player_rec_yds",
    "player_receiving_yards",
    "rec_yards",
}
TE_ROUTE = "TE_R5P_OOS_PRODUCTION_ORDER"


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _stable_seed(row: pd.Series) -> int:
    text = "|".join([
        VERSION,
        str(int(row["season"])),
        str(int(row["week"])),
        str(row["team"]),
        str(row["player_clean_key"]),
    ])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**32 - 1)


def _standard_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    z = rng.normal(0.0, 1.0, int(n)).astype(float)
    z -= float(z.mean())
    sd = float(z.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("invalid deterministic standard-normal vector")
    z /= sd
    if abs(float(z.mean())) > 1e-12 or abs(float(z.std(ddof=0)) - 1.0) > 1e-12:
        raise RuntimeError("standard-normal normalization drift")
    return z


def _overlay(base: np.ndarray, sigma: float, z: np.ndarray) -> np.ndarray:
    x = np.asarray(base, dtype=float)
    if x.ndim != 1 or len(x) != DRAW_COUNT:
        raise RuntimeError(f"baseline draw shape drift: {x.shape}")
    if not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError("baseline rec-yard draws must be finite/nonnegative")
    mu = float(x.mean())
    if not np.isfinite(mu) or mu <= 0:
        raise RuntimeError(f"invalid baseline mean {mu}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise RuntimeError(f"invalid overlay sigma {sigma}")
    if sigma == 0:
        out = x.copy()
    else:
        out = np.maximum(0.0, x + sigma * z)
        om = float(out.mean())
        if not np.isfinite(om) or om <= 0:
            raise RuntimeError("nonpositive overlay mean before reanchor")
        out *= mu / om
    gap = abs(float(out.mean()) - mu)
    if gap > MEAN_TOL:
        raise RuntimeError(f"mean reanchor failed gap={gap}")
    if not np.isfinite(out).all() or (out < 0).any():
        raise RuntimeError("overlay produced invalid draws")
    return out


def _empirical_crps(draws: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(draws, dtype=float))
    n = len(x)
    if n <= 0:
        raise RuntimeError("zero draws")
    term1 = float(np.mean(np.abs(x - float(y))))
    idx = np.arange(1, n + 1, dtype=float)
    term2 = float(np.sum((2.0 * idx - n - 1.0) * x) / (n * n))
    out = term1 - term2
    if not np.isfinite(out) or out < -1e-10:
        raise RuntimeError(f"invalid empirical CRPS {out}")
    return max(0.0, out)


def _interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(
        np.asarray(draws, dtype=float),
        [tail, 1.0 - tail],
        method="linear",
    )
    return float(lo), float(hi)


def _coverage_gap(hit: pd.Series, nominal: float) -> float:
    return abs(float(hit.mean()) - float(nominal))


def _load_parent(rows_path: Path, result_path: Path) -> pd.DataFrame:
    parent = json.loads(result_path.read_text(encoding="utf-8"))
    if parent.get("te", {}).get("disposition") != "TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED":
        raise RuntimeError(f"parent TE disposition drift: {parent.get('te', {}).get('disposition')}")
    x = _read_csv(rows_path, "allocation-only rows")
    need = {
        "season", "week", "team", "player_clean_key", "player_identity_key",
        "position", "game_id", "promoted_targets", "mc_rec_yards",
        "rec_yards", "candidate_targets_r5p", "candidate_rec_yards_r5p",
        "candidate_sigma", "pool_sigma", "player_mae", "difficulty_q4",
    }
    miss = need - set(x.columns)
    if miss:
        raise RuntimeError(f"allocation parent missing columns: {sorted(miss)}")
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str).str.strip()
    x = x.loc[
        x["position"].astype(str).str.upper().eq("TE")
        & x["season"].isin(TARGET_SEASONS)
    ].copy()
    for col in [
        "promoted_targets", "mc_rec_yards", "rec_yards",
        "candidate_targets_r5p", "candidate_rec_yards_r5p",
        "candidate_sigma", "pool_sigma", "player_mae",
    ]:
        x[col] = _num(x[col])

    # The exact allocation-only parent retained the recovered TE-R5P casebook
    # columns. Use its rec_yards outcome directly and prove its promoted
    # target/yard means still match the frozen R5P columns before scoring.
    x["actual_rec_yards"] = x["rec_yards"]
    target_gap = (x["promoted_targets"] - x["candidate_targets_r5p"]).abs()
    rec_mean_gap = (x["mc_rec_yards"] - x["candidate_rec_yards_r5p"]).abs()
    if float(target_gap.max()) > 1e-10 or float(rec_mean_gap.max()) > 1e-10:
        raise RuntimeError(
            "allocation parent / retained TE-R5P numerical lineage drift "
            f"targets={float(target_gap.max())} rec_yards={float(rec_mean_gap.max())}"
        )

    x["pregame_ypt"] = x["mc_rec_yards"] / x["promoted_targets"]
    good = (
        x["season"].notna()
        & x["week"].notna()
        & x["team"].ne("")
        & x["player_clean_key"].ne("")
        & x["player_identity_key"].astype(str).ne("")
        & x["promoted_targets"].gt(0)
        & x["pregame_ypt"].gt(0)
        & np.isfinite(x["pregame_ypt"])
        & x["actual_rec_yards"].ge(0)
        & x["candidate_sigma"].ge(0)
        & x["pool_sigma"].ge(0)
    )
    x = x.loc[good].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        sample = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate parent identities: {sample}")
    x["player_sigma_yards"] = x["candidate_sigma"] * x["pregame_ypt"]
    x["pool_sigma_yards"] = x["pool_sigma"] * x["pregame_ypt"]
    return x.reset_index(drop=True)

def _metadata_rows(dist_dir: Path) -> pd.DataFrame:
    parts = []
    for season in TARGET_SEASONS:
        for p in sorted(dist_dir.glob(f"{season}_week_*_metadata.csv")):
            x = _read_csv(p, f"distribution metadata {p.name}")
            need = {
                "season", "week", "team", "player_clean_key", "position", "market",
                "array_key", "npz_file", "wrte_authorized_treatment", "wrte_route",
            }
            miss = need - set(x.columns)
            if miss:
                raise RuntimeError(f"metadata missing columns {sorted(miss)} file={p}")
            x = x.loc[
                x["position"].astype(str).str.upper().eq("TE")
                & x["market"].astype(str).str.lower().isin(REC_MARKETS)
                & x["wrte_authorized_treatment"].astype(str).str.lower().isin({"true", "1"})
                & x["wrte_route"].astype(str).eq(TE_ROUTE)
            ].copy()
            if len(x):
                x["_metadata_path"] = str(p)
                parts.append(x)
    if not parts:
        raise RuntimeError("zero TE-R5P receiving-yard metadata rows")
    m = pd.concat(parts, ignore_index=True)
    m["season"] = _num(m["season"]).astype(int)
    m["week"] = _num(m["week"]).astype(int)
    m["team"] = m["team"].map(canon_team)
    m["player_clean_key"] = m["player_clean_key"].astype(str).str.strip()
    keys = ["season", "week", "team", "player_clean_key"]
    if m.duplicated(keys).any():
        sample = m.loc[m.duplicated(keys, keep=False), keys + ["market"]].head(20).to_dict("records")
        raise RuntimeError(f"duplicate TE rec-yard metadata identity: {sample}")
    return m


def _score_rows(parent: pd.DataFrame, dist_dir: Path) -> pd.DataFrame:
    meta = _metadata_rows(dist_dir)
    keys = ["season", "week", "team", "player_clean_key"]
    z = parent.merge(
        meta[keys + ["array_key", "npz_file", "mc_mean"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if z.empty:
        raise RuntimeError("zero parent-to-distribution matches")
    missing = len(parent) - len(z)
    if missing:
        raise RuntimeError(f"eligible parent rows missing TE-R5P distribution arrays: {missing}")

    cache: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for _, r in z.sort_values(["season", "week", "team", "player_clean_key"]).iterrows():
        npz_name = str(r["npz_file"])
        if npz_name not in cache:
            p = dist_dir / npz_name
            if not p.exists():
                raise RuntimeError(f"missing NPZ shard {p}")
            with np.load(p) as d:
                cache[npz_name] = {k: np.asarray(d[k], dtype=float) for k in d.files}
        arr = cache[npz_name].get(str(r["array_key"]))
        if arr is None:
            raise RuntimeError(f"array key missing {npz_name}:{r['array_key']}")
        if len(arr) != DRAW_COUNT:
            raise RuntimeError(f"draw-count drift {len(arr)} for {npz_name}:{r['array_key']}")
        meta_mean = float(r["mc_mean"])
        if abs(float(arr.mean()) - meta_mean) > 1e-10:
            raise RuntimeError("metadata/NPZ mean drift")

        seed = _stable_seed(r)
        stdz = _standard_noise(len(arr), seed)
        pooled = _overlay(arr, float(r["pool_sigma_yards"]), stdz)
        named = _overlay(arr, float(r["player_sigma_yards"]), stdz)

        y = float(r["actual_rec_yards"])
        b80 = _interval(arr, .80)
        p80 = _interval(pooled, .80)
        n80 = _interval(named, .80)
        b90 = _interval(arr, .90)
        p90 = _interval(pooled, .90)
        n90 = _interval(named, .90)

        bcrps = _empirical_crps(arr, y)
        pcrps = _empirical_crps(pooled, y)
        ncrps = _empirical_crps(named, y)

        d = r.to_dict()
        d.update({
            "baseline_mean": float(arr.mean()),
            "pooled_mean": float(pooled.mean()),
            "named_mean": float(named.mean()),
            "baseline_sd": float(arr.std(ddof=1)),
            "pooled_sd": float(pooled.std(ddof=1)),
            "named_sd": float(named.std(ddof=1)),
            "baseline_crps": bcrps,
            "pooled_crps": pcrps,
            "named_crps": ncrps,
            "gain_named_vs_baseline": bcrps - ncrps,
            "gain_named_vs_pooled": pcrps - ncrps,
            "baseline_80_hit": bool(b80[0] <= y <= b80[1]),
            "pooled_80_hit": bool(p80[0] <= y <= p80[1]),
            "named_80_hit": bool(n80[0] <= y <= n80[1]),
            "baseline_90_hit": bool(b90[0] <= y <= b90[1]),
            "pooled_90_hit": bool(p90[0] <= y <= p90[1]),
            "named_90_hit": bool(n90[0] <= y <= n90[1]),
            "baseline_80_low": b80[0], "baseline_80_high": b80[1],
            "pooled_80_low": p80[0], "pooled_80_high": p80[1],
            "named_80_low": n80[0], "named_80_high": n80[1],
            "baseline_90_low": b90[0], "baseline_90_high": b90[1],
            "pooled_90_low": p90[0], "pooled_90_high": p90[1],
            "named_90_low": n90[0], "named_90_high": n90[1],
            "max_mean_drift": max(
                abs(float(pooled.mean()) - float(arr.mean())),
                abs(float(named.mean()) - float(arr.mean())),
            ),
            "row_seed": int(seed),
        })
        rows.append(d)
    return pd.DataFrame(rows)


def _metric(q: pd.DataFrame) -> dict:
    if q.empty:
        return {"rows": 0, "players": 0, "games": 0}
    return {
        "rows": int(len(q)),
        "players": int(q["player_identity_key"].nunique()),
        "games": int(q["game_id"].nunique()),
        "baseline_crps": float(q["baseline_crps"].mean()),
        "pooled_crps": float(q["pooled_crps"].mean()),
        "named_crps": float(q["named_crps"].mean()),
        "gain_named_vs_baseline": float(q["gain_named_vs_baseline"].mean()),
        "gain_named_vs_pooled": float(q["gain_named_vs_pooled"].mean()),
        "baseline_80_coverage": float(q["baseline_80_hit"].mean()),
        "pooled_80_coverage": float(q["pooled_80_hit"].mean()),
        "named_80_coverage": float(q["named_80_hit"].mean()),
        "baseline_80_gap": _coverage_gap(q["baseline_80_hit"], .80),
        "pooled_80_gap": _coverage_gap(q["pooled_80_hit"], .80),
        "named_80_gap": _coverage_gap(q["named_80_hit"], .80),
        "baseline_90_coverage": float(q["baseline_90_hit"].mean()),
        "pooled_90_coverage": float(q["pooled_90_hit"].mean()),
        "named_90_coverage": float(q["named_90_hit"].mean()),
        "baseline_90_gap": _coverage_gap(q["baseline_90_hit"], .90),
        "pooled_90_gap": _coverage_gap(q["pooled_90_hit"], .90),
        "named_90_gap": _coverage_gap(q["named_90_hit"], .90),
        "baseline_mean_sd": float(q["baseline_sd"].mean()),
        "pooled_mean_sd": float(q["pooled_sd"].mean()),
        "named_mean_sd": float(q["named_sd"].mean()),
    }


def _cluster_boot(q: pd.DataFrame, gain_col: str, cluster_col: str, seed: int) -> dict:
    z = q[[gain_col, cluster_col]].dropna().copy()
    groups = {str(k): g[gain_col].to_numpy(float) for k, g in z.groupby(cluster_col, sort=True)}
    keys = np.asarray(sorted(groups), dtype=object)
    if len(keys) < 2:
        return {
            "reps": 0,
            "observed": float(z[gain_col].mean()) if len(z) else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_positive": np.nan,
        }
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        sample = rng.choice(keys, size=len(keys), replace=True)
        vals[i] = float(np.concatenate([groups[str(k)] for k in sample]).mean())
    return {
        "reps": BOOT_REPS,
        "observed": float(z[gain_col].mean()),
        "ci_low": float(np.quantile(vals, .025, method="linear")),
        "ci_high": float(np.quantile(vals, .975, method="linear")),
        "p_positive": float(np.mean(vals > 0)),
    }


def _player_diag(q: pd.DataFrame) -> dict:
    vals = []
    for _, g in q.loc[q["season"].eq(2025)].groupby("player_identity_key"):
        if len(g) < 6:
            continue
        vals.append(float(g["gain_named_vs_baseline"].mean()))
    a = np.asarray(vals, dtype=float)
    return {
        "qualifying_players_ge6": int(len(a)),
        "median_per_player_gain": float(np.median(a)) if len(a) else np.nan,
        "fraction_players_positive": float(np.mean(a > 0)) if len(a) else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation-rows", type=Path, required=True)
    ap.add_argument("--parent-result", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    parent = _load_parent(a.allocation_rows, a.parent_result)
    scored = _score_rows(parent, a.distribution_dir)
    max_drift = float(scored["max_mean_drift"].max()) if len(scored) else np.inf

    s2024 = _metric(scored.loc[scored["season"].eq(2024)])
    s2025 = _metric(scored.loc[scored["season"].eq(2025)])
    pooled = _metric(scored)
    q4 = _metric(scored.loc[scored["difficulty_q4"].astype(bool)])

    base_player = _cluster_boot(scored, "gain_named_vs_baseline", "player_identity_key", PLAYER_SEED)
    base_game = _cluster_boot(scored, "gain_named_vs_baseline", "game_id", GAME_SEED)
    pool_2025 = scored.loc[scored["season"].eq(2025)].copy()
    pool_player = _cluster_boot(pool_2025, "gain_named_vs_pooled", "player_identity_key", PLAYER_SEED + 100)
    pool_game = _cluster_boot(pool_2025, "gain_named_vs_pooled", "game_id", GAME_SEED + 100)
    player_diag = _player_diag(scored)

    support = (
        s2024["rows"] >= 600 and s2024["players"] >= 60
        and s2025["rows"] >= 600 and s2025["players"] >= 60
        and pooled["rows"] >= 1300 and pooled["players"] >= 90
    )
    coverage_strict = (
        pooled["named_80_gap"] < pooled["baseline_80_gap"]
        or pooled["named_90_gap"] < pooled["baseline_90_gap"]
    )
    gates = {
        "support": bool(support),
        "mean_neutrality": bool(max_drift <= MEAN_TOL),
        "2024_named_beats_canonical_crps": bool(s2024["named_crps"] < s2024["baseline_crps"]),
        "2025_named_beats_canonical_crps": bool(s2025["named_crps"] < s2025["baseline_crps"]),
        "2024_named_beats_pooled_crps": bool(s2024["named_crps"] < s2024["pooled_crps"]),
        "2025_named_beats_pooled_crps": bool(s2025["named_crps"] < s2025["pooled_crps"]),
        "pooled_named_vs_canonical_player_ci_low_positive": bool(base_player["ci_low"] > 0),
        "pooled_named_vs_canonical_game_ci_low_positive": bool(base_game["ci_low"] > 0),
        "2025_named_vs_pooled_player_ci_low_positive": bool(pool_player["ci_low"] > 0),
        "2025_named_vs_pooled_game_ci_low_positive": bool(pool_game["ci_low"] > 0),
        "pooled_80_gap_nonworse": bool(pooled["named_80_gap"] <= pooled["baseline_80_gap"]),
        "pooled_90_gap_nonworse": bool(pooled["named_90_gap"] <= pooled["baseline_90_gap"]),
        "pooled_at_least_one_coverage_gap_strictly_better": bool(coverage_strict),
        "q4_named_beats_canonical_crps": bool(q4["named_crps"] < q4["baseline_crps"]),
        "2025_player_median_gain_positive": bool(player_diag["median_per_player_gain"] > 0),
        "2025_player_fraction_positive_gt_half": bool(player_diag["fraction_players_positive"] > .50),
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "outcomes_2026_read_zero": True,
        "production_unchanged": True,
    }
    qualified = all(gates.values())
    disposition = (
        "TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_QUALIFIED"
        if qualified
        else "NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS"
    )

    result = {
        "version": VERSION,
        "disposition": disposition,
        "season_metrics": {"2024": s2024, "2025": s2025},
        "pooled": pooled,
        "high_difficulty_q4": q4,
        "named_vs_canonical_player_cluster": base_player,
        "named_vs_canonical_game_cluster": base_game,
        "2025_named_vs_pooled_player_cluster": pool_player,
        "2025_named_vs_pooled_game_cluster": pool_game,
        "player_diagnostics_2025": player_diag,
        "max_mean_drift": max_drift,
        "eligible_parent_rows": int(len(parent)),
        "scored_rows": int(len(scored)),
        "scored_players": int(scored["player_identity_key"].nunique()),
        "draw_count": DRAW_COUNT,
        "candidate_models_fit": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "projection_mean_changes": 0,
        "joint_distribution_claimed": False,
        "production_changed": False,
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
    }
    scored.to_csv(a.out_dir / "te_player_allocation_marginal_rec_yards_rows.csv", index=False)
    (a.out_dir / "te_player_allocation_marginal_rec_yards_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
