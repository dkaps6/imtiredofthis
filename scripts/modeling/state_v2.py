"""Leakage-safe first-order Markov state model for player prop outcomes.

The legacy ``scripts/models/markov.py`` was not a Markov model; it multiplied
attempts by efficiency and applied a Normal CDF, returning 0.5 when inputs were
missing. This module implements an actual discrete state-transition model.

For each supported target and position group, historical pregame data is split
into LOW/MID/HIGH outcome states. We estimate smoothed P(next_state | state)
from consecutive same-season player games and predict the next-game outcome as
the transition-weighted destination-state mean.

Production fitting is cut off strictly before the target slate week. Sportsbook
lines are never used as state boundaries, features, or targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
LOGS = DATA / "player_game_logs.csv"
CONSENSUS = DATA / "player_form_consensus.csv"
MODEL_PATH = DATA / "model_state_v2.joblib"
STATES = ("LOW", "MID", "HIGH")

TARGET_COLUMNS = {
    "pass_yards": "pass_yards",
    "rush_yards": "rush_yards",
    "rec_yards": "rec_yards",
    "receptions": "receptions",
    "rush_att": "rushes",
    "rush_rec_yards": "rush_rec_yards",
}

MARKET_MAP = {
    "player_pass_yds": "pass_yards", "player_passing_yards": "pass_yards", "pass_yards": "pass_yards",
    "player_rush_yds": "rush_yards", "player_rushing_yards": "rush_yards", "rush_yards": "rush_yards",
    "player_reception_yds": "rec_yards", "player_rec_yds": "rec_yards", "player_receiving_yards": "rec_yards", "rec_yards": "rec_yards",
    "player_receptions": "receptions", "receptions": "receptions",
    "player_rush_att": "rush_att", "rush_att": "rush_att",
    "player_rush_reception_yds": "rush_rec_yards", "player_rush_rec_yds": "rush_rec_yards", "rush_rec_yards": "rush_rec_yards",
}

MIN_TRANSITIONS = {
    "pass_yards": 40,
    "rush_yards": 100,
    "rec_yards": 100,
    "receptions": 100,
    "rush_att": 100,
    "rush_rec_yards": 100,
}


@dataclass
class StateSpec:
    target: str
    position_group: str
    low_cut: float
    high_cut: float
    state_means: Dict[str, float]
    transition_probs: Dict[str, Dict[str, float]]
    transitions: int


@dataclass
class StateBundle:
    specs: Dict[tuple[str, str], StateSpec]
    target_season: int
    target_week: int
    method: str = "first_order_markov_outcome_regime_v2"


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
    if p == "TE":
        return "TE"
    if p == "QB":
        return "QB"
    return "OTHER"


def _prepare_logs(logs: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    if logs is None or logs.empty:
        raise RuntimeError("State v2 requires non-empty player_game_logs")
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "player", "team", "position"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"player_game_logs missing state-model columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[
        x["season"].lt(int(target_season))
        | (x["season"].eq(int(target_season)) & x["week"].lt(int(target_week)))
    ].copy()
    if x.empty:
        raise RuntimeError(f"State v2 found no pregame logs before {target_season} week {target_week}")
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x.get("player_clean_key", x["player"]).map(_key)
    x["position_group"] = x["position"].map(_position_group)
    for col in ("pass_yards", "rush_yards", "rec_yards", "receptions", "rushes", "pass_att", "targets"):
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["rush_rec_yards"] = x["rush_yards"].fillna(0.0) + x["rec_yards"].fillna(0.0)
    return x.sort_values(["player_clean_key", "season", "week"]).reset_index(drop=True)


def _eligible(frame: pd.DataFrame, target: str, position_group: str) -> pd.Series:
    y = pd.to_numeric(frame[TARGET_COLUMNS[target]], errors="coerce")
    mask = frame["position_group"].eq(position_group) & y.notna()
    if target == "pass_yards":
        mask &= frame["position_group"].eq("QB") & pd.to_numeric(frame["pass_att"], errors="coerce").gt(0)
    elif target in {"rec_yards", "receptions"}:
        mask &= frame["position_group"].isin(["WR", "TE", "RB"]) & pd.to_numeric(frame["targets"], errors="coerce").gt(0)
    elif target in {"rush_yards", "rush_att"}:
        mask &= frame["position_group"].isin(["QB", "RB", "WR", "TE"]) & pd.to_numeric(frame["rushes"], errors="coerce").gt(0)
    elif target == "rush_rec_yards":
        opportunities = pd.to_numeric(frame["rushes"], errors="coerce").fillna(0) + pd.to_numeric(frame["targets"], errors="coerce").fillna(0)
        mask &= frame["position_group"].isin(["QB", "RB", "WR", "TE"]) & opportunities.gt(0)
    return mask


def _state(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "LOW"
    if value >= high_cut:
        return "HIGH"
    return "MID"


def _fit_spec(frame: pd.DataFrame, target: str, position_group: str) -> StateSpec | None:
    mask = _eligible(frame, target, position_group)
    part = frame.loc[mask, ["player_clean_key", "season", "week", TARGET_COLUMNS[target]]].copy()
    if len(part) < 30:
        return None
    ycol = TARGET_COLUMNS[target]
    vals = pd.to_numeric(part[ycol], errors="coerce").dropna()
    if vals.nunique() < 3:
        return None
    low_cut = float(vals.quantile(1.0 / 3.0))
    high_cut = float(vals.quantile(2.0 / 3.0))
    if not np.isfinite(low_cut) or not np.isfinite(high_cut) or high_cut <= low_cut:
        return None
    part["state"] = pd.to_numeric(part[ycol], errors="coerce").map(lambda v: _state(float(v), low_cut, high_cut))
    part = part.sort_values(["player_clean_key", "season", "week"])
    # Do not treat Week 1 as an immediate transition from the prior season's finale.
    # The current-state predictor may still use the most recent prior-season game
    # for a Week 1 forecast, but transition probabilities are learned in-season.
    part["prev_state"] = part.groupby(["player_clean_key", "season"])["state"].shift(1)
    trans = part.loc[part["prev_state"].notna()].copy()
    n_trans = len(trans)
    if n_trans < MIN_TRANSITIONS[target]:
        return None

    state_means = {}
    global_mean = float(vals.mean())
    for s in STATES:
        sv = pd.to_numeric(part.loc[part["state"].eq(s), ycol], errors="coerce").dropna()
        state_means[s] = float(sv.mean()) if len(sv) else global_mean

    probs: Dict[str, Dict[str, float]] = {}
    for src in STATES:
        counts = trans.loc[trans["prev_state"].eq(src), "state"].value_counts().to_dict()
        denom = float(sum(counts.get(dst, 0) for dst in STATES) + len(STATES))
        probs[src] = {dst: float((counts.get(dst, 0) + 1.0) / denom) for dst in STATES}

    return StateSpec(target, position_group, low_cut, high_cut, state_means, probs, n_trans)


def train_state_model(logs: pd.DataFrame, target_season: int, target_week: int) -> StateBundle:
    x = _prepare_logs(logs, target_season, target_week)
    specs: Dict[tuple[str, str], StateSpec] = {}
    for target in TARGET_COLUMNS:
        for pos in ("QB", "RB", "WR", "TE"):
            spec = _fit_spec(x, target, pos)
            if spec is not None:
                specs[(target, pos)] = spec
    if not specs:
        raise RuntimeError("State v2 trained zero transition specifications")
    return StateBundle(specs=specs, target_season=int(target_season), target_week=int(target_week))


def predict_current(bundle: StateBundle, logs: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    history = _prepare_logs(logs, bundle.target_season, bundle.target_week)
    c = consensus.copy()
    c.columns = [str(v).strip().lower() for v in c.columns]
    required = {"player", "team", "position"}
    missing = required - set(c.columns)
    if missing:
        raise RuntimeError(f"player_form_consensus missing state-model columns: {sorted(missing)}")
    c["team"] = c["team"].map(canon_team)
    c["player_clean_key"] = c.get("player_clean_key", c["player"]).map(_key)
    c["position_group"] = c["position"].map(_position_group)
    by_player = {k: part.sort_values(["season", "week"]) for k, part in history.groupby("player_clean_key", sort=False)}

    rows = []
    for _, player in c.drop_duplicates(["team", "player_clean_key"]).iterrows():
        hist = by_player.get(player["player_clean_key"], pd.DataFrame())
        rec = {
            "player": player["player"], "player_clean_key": player["player_clean_key"], "team": player["team"],
            "season": int(bundle.target_season), "week": int(bundle.target_week), "position": player["position"],
            "state_position_group": player["position_group"], "state_method": bundle.method,
            "state_training_cutoff": f"{bundle.target_season}-W{bundle.target_week:02d} pregame",
        }
        available = []
        for target in TARGET_COLUMNS:
            spec = bundle.specs.get((target, player["position_group"]))
            ycol = TARGET_COLUMNS[target]
            last = np.nan
            if spec is not None and not hist.empty and ycol in hist.columns:
                eligible_hist = hist.loc[_eligible(hist, target, player["position_group"])]
                s = pd.to_numeric(eligible_hist.get(ycol, pd.Series(dtype=float)), errors="coerce").dropna()
                if len(s):
                    last = float(s.iloc[-1])
            if spec is None or not np.isfinite(last):
                rec[f"state_{target}"] = np.nan
                rec[f"state_{target}_current_state"] = ""
                rec[f"state_{target}_p_low"] = np.nan
                rec[f"state_{target}_p_mid"] = np.nan
                rec[f"state_{target}_p_high"] = np.nan
                continue
            current_state = _state(last, spec.low_cut, spec.high_cut)
            p = spec.transition_probs[current_state]
            rec[f"state_{target}"] = float(max(0.0, sum(p[s] * spec.state_means[s] for s in STATES)))
            rec[f"state_{target}_current_state"] = current_state
            rec[f"state_{target}_p_low"] = p["LOW"]
            rec[f"state_{target}_p_mid"] = p["MID"]
            rec[f"state_{target}_p_high"] = p["HIGH"]
            available.append(target)
        rec["state_available"] = int(bool(available))
        rec["state_targets_available"] = ",".join(available)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_state_predictions(logs: pd.DataFrame, consensus: pd.DataFrame, target_season: int, target_week: int) -> tuple[StateBundle, pd.DataFrame]:
    bundle = train_state_model(logs, target_season, target_week)
    return bundle, predict_current(bundle, logs, consensus)


def save_bundle(bundle: StateBundle, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def apply_state_to_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach market-specific state projections without blending them."""
    if metrics is None or metrics.empty:
        return metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame()
    out = metrics.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out["team"] = out["team"].map(canon_team)
    source = out["player_clean_key"] if "player_clean_key" in out.columns else out["player"]
    out["_state_key"] = source.map(_key)
    keep = ["team", "player_clean_key", "state_available", "state_method", "state_training_cutoff"] + [f"state_{t}" for t in TARGET_COLUMNS]
    joined = predictions[keep].rename(columns={"player_clean_key": "_state_key"})
    out = out.merge(joined, on=["team", "_state_key"], how="left", validate="many_to_one")
    canonical_market = out["market"].astype(str).str.lower().map(lambda x: MARKET_MAP.get(x, x))
    out["state_proj"] = [row.get(f"state_{market}", np.nan) for (_, row), market in zip(out.iterrows(), canonical_market)]
    out["state_applied"] = pd.to_numeric(out["state_proj"], errors="coerce").notna().astype(int)
    out.drop(columns=["_state_key"], inplace=True)
    return out
