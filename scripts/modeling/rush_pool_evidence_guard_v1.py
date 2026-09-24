"""Production-integration seam for Rush Pool Evidence Guard V1.

Scientific authority:
- RUSH_POOL_EVIDENCE_GUARD_V1_QUALIFIED
- run 36045359696 / artifact 10828600510
- docs/research/RUSH_POOL_EVIDENCE_GUARD_V1_RESULT.md
- docs/production/RUSH_POOL_EVIDENCE_GUARD_V1_INTEGRATION_PLAN.md

This module does not fit or transform rushing shares. It changes only finite
top-five pool membership from Week 2 onward when explicitly enabled.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

VERSION = "RUSH_POOL_EVIDENCE_GUARD_V1"
ENV_VAR = "RUSH_POOL_EVIDENCE_GUARD_V1"
FALLBACK_STATE = "position_prior_only"
POOL_SIZE = 5


def enabled() -> bool:
    return str(os.environ.get(ENV_VAR, "")).strip().lower() in {"1", "true", "yes", "on"}


def stable_rush_seed(*, simulation_seed: int, game: object, team: object) -> int:
    payload = f"{VERSION}|{int(simulation_seed)}|{str(game)}|{str(team)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _week(team_df: pd.DataFrame) -> int:
    if "week" not in team_df.columns:
        raise RuntimeError(f"{VERSION} requires week in simulation metrics")
    values = pd.to_numeric(team_df["week"], errors="coerce").dropna().astype(int).unique()
    if len(values) != 1:
        raise RuntimeError(f"{VERSION} requires exactly one team-week; got {values.tolist()}")
    return int(values[0])


def select_shares(
    team_df: pd.DataFrame,
    raw_rush_shares: np.ndarray,
    baseline_top5_shares: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return the exact frozen V1 share selector and audit metadata.

    Week 1 is an exact no-op. From Week 2 onward, positive-share players with
    player-specific evidence fill the existing five finite slots before
    position-prior-only fallbacks. Share magnitudes are never changed here.
    """
    raw = np.clip(
        np.nan_to_num(np.asarray(raw_rush_shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        0.95,
    )
    baseline = np.clip(
        np.nan_to_num(np.asarray(baseline_top5_shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        0.95,
    )
    if len(raw) != len(team_df) or len(baseline) != len(team_df):
        raise RuntimeError(f"{VERSION} share/team length mismatch")

    week = _week(team_df)
    if week <= 1:
        return baseline.copy(), {
            "version": VERSION,
            "week": week,
            "applied": False,
            "reason": "week1_noop",
            "baseline_selected": int(np.count_nonzero(baseline > 0.0)),
            "candidate_selected": int(np.count_nonzero(baseline > 0.0)),
            "fallback_selected_baseline": 0,
            "fallback_selected_candidate": 0,
        }

    if "bayes_evidence_state" not in team_df.columns:
        raise RuntimeError(f"{VERSION} requires bayes_evidence_state from canonical Bayesian inputs")

    states = team_df["bayes_evidence_state"].fillna("").astype(str).to_numpy(object)
    positive = np.flatnonzero(raw > 0.0)
    missing = positive[np.asarray([not str(states[i]).strip() for i in positive], dtype=bool)]
    if len(missing):
        sample = team_df.iloc[missing][["team", "player"]].head(10).to_dict("records")
        raise RuntimeError(f"{VERSION} positive-share players missing evidence state: {sample}")

    evidenced = positive[np.asarray([states[i] != FALLBACK_STATE for i in positive], dtype=bool)]
    fallback = positive[np.asarray([states[i] == FALLBACK_STATE for i in positive], dtype=bool)]
    if len(evidenced):
        evidenced = evidenced[np.argsort(-raw[evidenced], kind="stable")]
    if len(fallback):
        fallback = fallback[np.argsort(-raw[fallback], kind="stable")]

    keep = list(evidenced[:POOL_SIZE])
    if len(keep) < POOL_SIZE:
        keep.extend(list(fallback[: POOL_SIZE - len(keep)]))

    candidate = np.zeros_like(raw)
    if keep:
        idx = np.asarray(keep, dtype=int)
        candidate[idx] = raw[idx]

    baseline_mask = baseline > 0.0
    candidate_mask = candidate > 0.0
    changed = not np.array_equal(baseline_mask, candidate_mask)

    return candidate, {
        "version": VERSION,
        "week": week,
        "applied": bool(changed),
        "reason": "evidence_first_top5" if changed else "same_pool_membership",
        "baseline_selected": int(baseline_mask.sum()),
        "candidate_selected": int(candidate_mask.sum()),
        "fallback_selected_baseline": int(((states == FALLBACK_STATE) & baseline_mask).sum()),
        "fallback_selected_candidate": int(((states == FALLBACK_STATE) & candidate_mask).sum()),
        "evidenced_omitted_baseline": int(((states != FALLBACK_STATE) & (raw > 0.0) & ~baseline_mask).sum()),
        "evidenced_omitted_candidate": int(((states != FALLBACK_STATE) & (raw > 0.0) & ~candidate_mask).sum()),
    }
