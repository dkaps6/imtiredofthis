"""Empirical-Bayesian player baseline for the canonical projection stack.

This replaces the legacy ``scripts/models/bayes_hier.py`` placeholder with a
real shrinkage model that combines:

1. position-level population priors,
2. prior-season player evidence, and
3. current-season pregame evidence only.

The module is deliberately market-independent and leakage-safe. It consumes the
already-cutoff PlayerForm consensus artifact; it never reads future game results
or sportsbook lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
CONSENSUS = DATA / "player_form_consensus.csv"

RATE_METRICS = ("tgt_share", "rush_share", "route_rate", "receptions_per_target")
EFF_METRICS = ("yprr", "ypt", "ypc", "ypa")
ALL_METRICS = RATE_METRICS + EFF_METRICS

# Position-pool prior equivalent sample. These are intentionally modest: player
# history should matter, but a tiny/current sample must not erase the population
# prior. 2025 walk-forward evaluation will be allowed to recalibrate these.
GROUP_STRENGTH = {
    "tgt_share": 3.0,
    "rush_share": 3.0,
    "route_rate": 3.0,
    "receptions_per_target": 4.0,
    "yprr": 4.0,
    "ypt": 5.0,
    "ypc": 5.0,
    "ypa": 6.0,
}
PRIOR_PLAYER_CAP = {
    "tgt_share": 6.0,
    "rush_share": 6.0,
    "route_rate": 6.0,
    "receptions_per_target": 8.0,
    "yprr": 8.0,
    "ypt": 8.0,
    "ypc": 8.0,
    "ypa": 10.0,
}

DEFAULTS = {
    "tgt_share": {"QB": 0.0, "RB": 0.08, "WR": 0.16, "TE": 0.12, "OTHER": 0.06},
    "rush_share": {"QB": 0.14, "RB": 0.32, "WR": 0.02, "TE": 0.00, "OTHER": 0.02},
    "route_rate": {"QB": np.nan, "RB": 0.42, "WR": 0.70, "TE": 0.62, "OTHER": 0.35},
    "receptions_per_target": {"QB": 0.0, "RB": 0.76, "WR": 0.64, "TE": 0.68, "OTHER": 0.64},
    "yprr": {"QB": np.nan, "RB": 1.25, "WR": 1.65, "TE": 1.45, "OTHER": 1.20},
    "ypt": {"QB": np.nan, "RB": 6.0, "WR": 8.0, "TE": 7.4, "OTHER": 7.0},
    "ypc": {"QB": 5.0, "RB": 4.2, "WR": 6.0, "TE": 4.0, "OTHER": 4.2},
    "ypa": {"QB": 7.0, "RB": np.nan, "WR": np.nan, "TE": np.nan, "OTHER": np.nan},
}


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _position_group(value) -> str:
    p = str(value or "").upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR", "WIDE RECEIVER", "SLOT WR"}:
        return "WR"
    if p in {"RB", "FB", "HB"}:
        return "RB"
    if p in {"TE"}:
        return "TE"
    if p in {"QB"}:
        return "QB"
    return "OTHER"


def _num_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _weighted_group_mean(df: pd.DataFrame, metric: str, pos: str) -> float:
    pv = _num_series(df, f"{metric}_prior")
    pg = _num_series(df, "prior_games").fillna(0.0).clip(lower=0.0)
    mask = df["bayes_position_group"].eq(pos) & pv.notna() & pg.gt(0)
    if mask.any() and float(pg.loc[mask].sum()) > 0:
        return float(np.average(pv.loc[mask], weights=pg.loc[mask]))
    mask = pv.notna()
    if mask.any():
        return float(pv.loc[mask].median())
    return float(DEFAULTS[metric][pos])


def _group_scale(df: pd.DataFrame, metric: str, pos: str, mean: float) -> float:
    pv = _num_series(df, f"{metric}_prior")
    mask = df["bayes_position_group"].eq(pos) & pv.notna()
    if int(mask.sum()) >= 5:
        sd = float(pv.loc[mask].std(ddof=1))
        if np.isfinite(sd) and sd > 0:
            return sd
    # Stable fallback scale used only for posterior uncertainty diagnostics.
    if metric in RATE_METRICS:
        return max(0.03, min(0.20, abs(mean) * 0.35 + 0.02))
    return max(0.35, abs(mean) * 0.20)


def _posterior_mean(group_mean: float, prior_value: float, prior_games: float,
                    current_value: float, current_games: float, metric: str) -> tuple[float, float]:
    weights = [GROUP_STRENGTH[metric]]
    values = [group_mean]
    if np.isfinite(prior_value) and prior_games > 0:
        weights.append(min(float(prior_games), PRIOR_PLAYER_CAP[metric]))
        values.append(float(prior_value))
    if np.isfinite(current_value) and current_games > 0:
        weights.append(float(current_games))
        values.append(float(current_value))
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    return float(np.average(v, weights=w)), float(w.sum())


def build_bayesian_baseline(consensus: pd.DataFrame) -> pd.DataFrame:
    """Create one leakage-safe empirical-Bayes posterior row per active player."""
    if consensus is None or consensus.empty:
        raise RuntimeError("Bayesian baseline requires non-empty player_form_consensus")
    df = consensus.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"player", "team", "season", "position"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"player_form_consensus missing columns: {sorted(missing)}")

    df["team"] = df["team"].map(canon_team)
    if df["team"].eq("").any():
        raise RuntimeError("Bayesian baseline found unresolved team identity")
    df["player_clean_key"] = df.get("player_clean_key", df["player"]).map(_key)
    df["bayes_position_group"] = df["position"].map(_position_group)
    df["prior_games"] = _num_series(df, "prior_games").fillna(0.0).clip(lower=0.0)
    df["current_games"] = _num_series(df, "current_games").fillna(0.0).clip(lower=0.0)

    # If the consensus artifact predates explicit prior/current columns, the
    # already-blended metric may seed the player-prior slot, but provenance makes
    # that compatibility path visible. Current migrations should carry the split.
    compatibility_used = False
    for metric in ALL_METRICS:
        pcol, ccol = f"{metric}_prior", f"{metric}_current"
        if pcol not in df.columns:
            df[pcol] = _num_series(df, metric)
            compatibility_used = True
        if ccol not in df.columns:
            df[ccol] = np.nan

    group_cache: dict[tuple[str, str], tuple[float, float]] = {}
    for metric in ALL_METRICS:
        for pos in ("QB", "RB", "WR", "TE", "OTHER"):
            gm = _weighted_group_mean(df, metric, pos)
            gs = _group_scale(df, metric, pos, gm)
            group_cache[(metric, pos)] = (gm, gs)

    for metric in ALL_METRICS:
        means, sds, eff_n = [], [], []
        for _, row in df.iterrows():
            pos = str(row["bayes_position_group"])
            group_mean, group_sd = group_cache[(metric, pos)]
            pv = pd.to_numeric(pd.Series([row.get(f"{metric}_prior")]), errors="coerce").iloc[0]
            cv = pd.to_numeric(pd.Series([row.get(f"{metric}_current")]), errors="coerce").iloc[0]
            mean, n_eff = _posterior_mean(
                group_mean,
                float(pv) if pd.notna(pv) else np.nan,
                float(row["prior_games"]),
                float(cv) if pd.notna(cv) else np.nan,
                float(row["current_games"]),
                metric,
            )
            if metric in RATE_METRICS and np.isfinite(mean):
                mean = float(np.clip(mean, 0.0, 1.0))
            means.append(mean)
            eff_n.append(n_eff)
            sds.append(float(group_sd / np.sqrt(max(1.0, n_eff))))
        df[f"bayes_{metric}"] = means
        df[f"bayes_{metric}_sd"] = sds
        df[f"bayes_{metric}_effective_n"] = eff_n

    df["bayes_available"] = 1
    df["bayes_method"] = "empirical_bayes_position+player_prior+current"
    df["bayes_compatibility_prior_used"] = int(compatibility_used)
    df["bayes_evidence_state"] = np.select(
        [df["current_games"].gt(0), df["prior_games"].gt(0)],
        ["prior+current", "prior_only"],
        default="position_prior_only",
    )

    cols = [
        "player", "player_clean_key", "team", "season", "position", "bayes_position_group",
        "prior_games", "current_games", "bayes_available", "bayes_method",
        "bayes_compatibility_prior_used", "bayes_evidence_state",
    ]
    for metric in ALL_METRICS:
        cols += [f"bayes_{metric}", f"bayes_{metric}_sd", f"bayes_{metric}_effective_n"]
    return df[cols].drop_duplicates(["team", "player_clean_key"]).reset_index(drop=True)


def load_bayesian_baseline(path: Path = CONSENSUS) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Bayesian baseline source missing: {path}")
    return build_bayesian_baseline(pd.read_csv(path))


def apply_bayesian_to_metrics(metrics: pd.DataFrame, baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach posterior football baselines to every sportsbook row for a player."""
    if metrics is None or metrics.empty:
        return metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame()
    post = load_bayesian_baseline() if baseline is None else baseline.copy()
    out = metrics.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out["team"] = out["team"].map(canon_team)
    source = out["player_clean_key"] if "player_clean_key" in out.columns else out["player"]
    out["_bayes_key"] = source.map(_key)
    keep = [c for c in post.columns if c.startswith("bayes_")] + ["team", "player_clean_key"]
    joined = post[keep].rename(columns={"player_clean_key": "_bayes_key"})
    out = out.merge(joined, on=["team", "_bayes_key"], how="left", validate="many_to_one")
    out["bayes_applied"] = pd.to_numeric(out.get("bayes_available", 0), errors="coerce").fillna(0).astype(int)
    out.drop(columns=["_bayes_key"], inplace=True)
    return out
