"""Leakage-safe supervised ML projections for the canonical model stack.

The legacy ``scripts/models/ml_ensemble.py`` did not train or load a model; it
merely consumed an externally supplied ``p_ml`` and otherwise returned 0.5.
This module replaces that placeholder with real, deterministic scikit-learn
regressors trained from historical player-game observations.

Design principles:
- sportsbook lines are never model features;
- every training row uses only games that occurred before that row;
- production training is cut off strictly before the target slate week;
- the ML signal remains independent of Bayesian/rules/Monte Carlo so the future
  ensemble can measure whether it contributes incremental predictive value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
LOGS = DATA / "player_game_logs.csv"
CONSENSUS = DATA / "player_form_consensus.csv"
MODEL_PATH = DATA / "model_ml_v2.joblib"

BASE_STATS = (
    "targets", "receptions", "rec_yards", "rushes", "rush_yards", "pass_att", "pass_yards",
    "tgt_share_game", "rush_share_game", "ypt_game", "ypc_game", "ypa_game", "catch_rate_game",
)
POSITION_COLUMNS = ("pos_qb", "pos_rb", "pos_wr", "pos_te", "pos_other")
FEATURE_COLUMNS = (
    "hist_games",
    *tuple(f"prev_{s}" for s in BASE_STATS),
    *tuple(f"mean3_{s}" for s in BASE_STATS),
    *tuple(f"mean5_{s}" for s in BASE_STATS),
    *POSITION_COLUMNS,
)

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

MIN_TRAIN_ROWS = {
    "pass_yards": 80,
    "rush_yards": 250,
    "rec_yards": 250,
    "receptions": 250,
    "rush_att": 250,
    "rush_rec_yards": 250,
}


@dataclass
class MLBundle:
    models: Dict[str, HistGradientBoostingRegressor]
    train_rows: Dict[str, int]
    feature_columns: tuple[str, ...]
    target_season: int
    target_week: int
    method: str = "hist_gradient_boosting_lagged_player_history_v2"


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
        raise RuntimeError("ML v2 requires non-empty player_game_logs")
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "player", "team", "position"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"player_game_logs missing ML columns: {sorted(missing)}")

    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[
        x["season"].lt(int(target_season))
        | (x["season"].eq(int(target_season)) & x["week"].lt(int(target_week)))
    ].copy()
    if x.empty:
        raise RuntimeError(f"ML v2 found no pregame logs before {target_season} week {target_week}")

    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x.get("player_clean_key", x["player"]).map(_key)
    x["position_group"] = x["position"].map(_position_group)
    for col in BASE_STATS:
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["rush_rec_yards"] = pd.to_numeric(x.get("rush_yards"), errors="coerce").fillna(0.0) + pd.to_numeric(x.get("rec_yards"), errors="coerce").fillna(0.0)
    x = x.sort_values(["season", "week", "player_clean_key", "team"]).reset_index(drop=True)
    return x


def _add_position_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pos = out["position_group"].astype(str)
    for group, col in (("QB", "pos_qb"), ("RB", "pos_rb"), ("WR", "pos_wr"), ("TE", "pos_te"), ("OTHER", "pos_other")):
        out[col] = pos.eq(group).astype(float)
    return out


def build_training_frame(logs: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    """Build historical examples with all player-history features shifted one game."""
    x = _prepare_logs(logs, target_season, target_week)
    g = x.groupby("player_clean_key", sort=False, group_keys=False)
    x["hist_games"] = g.cumcount().astype(float)
    for stat in BASE_STATS:
        x[f"prev_{stat}"] = g[stat].shift(1)
        x[f"mean3_{stat}"] = g[stat].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        x[f"mean5_{stat}"] = g[stat].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    x = _add_position_features(x)
    return x


def build_current_features(logs: pd.DataFrame, consensus: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    """Build one pregame ML feature row for each active slate player."""
    history = _prepare_logs(logs, target_season, target_week)
    c = consensus.copy()
    c.columns = [str(v).strip().lower() for v in c.columns]
    required = {"player", "team", "position"}
    missing = required - set(c.columns)
    if missing:
        raise RuntimeError(f"player_form_consensus missing ML columns: {sorted(missing)}")
    c["team"] = c["team"].map(canon_team)
    c["player_clean_key"] = c.get("player_clean_key", c["player"]).map(_key)
    c["position_group"] = c["position"].map(_position_group)

    by_player = {k: part.sort_values(["season", "week"]) for k, part in history.groupby("player_clean_key", sort=False)}
    rows = []
    for _, player in c.drop_duplicates(["team", "player_clean_key"]).iterrows():
        hist = by_player.get(player["player_clean_key"], pd.DataFrame())
        row = {
            "player": player["player"],
            "player_clean_key": player["player_clean_key"],
            "team": player["team"],
            "season": int(target_season),
            "week": int(target_week),
            "position": player["position"],
            "position_group": player["position_group"],
            "hist_games": float(len(hist)),
        }
        for stat in BASE_STATS:
            s = pd.to_numeric(hist.get(stat, pd.Series(dtype=float)), errors="coerce") if not hist.empty else pd.Series(dtype=float)
            row[f"prev_{stat}"] = float(s.iloc[-1]) if len(s) and pd.notna(s.iloc[-1]) else np.nan
            row[f"mean3_{stat}"] = float(s.tail(3).mean()) if s.notna().any() else np.nan
            row[f"mean5_{stat}"] = float(s.tail(5).mean()) if s.notna().any() else np.nan
        rows.append(row)
    out = _add_position_features(pd.DataFrame(rows))
    return out


def _eligible_training_rows(frame: pd.DataFrame, target: str) -> pd.Series:
    ycol = TARGET_COLUMNS[target]
    y = pd.to_numeric(frame[ycol], errors="coerce")
    mask = y.notna() & frame["hist_games"].ge(1)
    if target == "pass_yards":
        mask &= frame["position_group"].eq("QB") & pd.to_numeric(frame["pass_att"], errors="coerce").gt(0)
    elif target in {"rec_yards", "receptions"}:
        mask &= pd.to_numeric(frame["targets"], errors="coerce").gt(0)
    return mask


def train_models(training: pd.DataFrame, target_season: int, target_week: int) -> MLBundle:
    models: Dict[str, HistGradientBoostingRegressor] = {}
    train_rows: Dict[str, int] = {}
    X_all = training[list(FEATURE_COLUMNS)].astype(float)
    for target, ycol in TARGET_COLUMNS.items():
        mask = _eligible_training_rows(training, target)
        n = int(mask.sum())
        train_rows[target] = n
        if n < MIN_TRAIN_ROWS[target]:
            continue
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(X_all.loc[mask], pd.to_numeric(training.loc[mask, ycol], errors="coerce").astype(float))
        models[target] = model
    if not models:
        raise RuntimeError(f"ML v2 trained zero target models; rows={train_rows}")
    return MLBundle(models=models, train_rows=train_rows, feature_columns=FEATURE_COLUMNS, target_season=int(target_season), target_week=int(target_week))


def predict_current(bundle: MLBundle, features: pd.DataFrame) -> pd.DataFrame:
    out = features[["player", "player_clean_key", "team", "season", "week", "position", "position_group", "hist_games"]].copy()
    X = features[list(bundle.feature_columns)].astype(float)
    available = []
    for target in TARGET_COLUMNS:
        col = f"ml_{target}"
        if target in bundle.models:
            pred = bundle.models[target].predict(X)
            out[col] = np.clip(pred, 0.0, None)
            available.append(target)
        else:
            out[col] = np.nan
    out["ml_available"] = int(bool(available))
    out["ml_targets_available"] = ",".join(available)
    out["ml_method"] = bundle.method
    out["ml_training_cutoff"] = f"{bundle.target_season}-W{bundle.target_week:02d} pregame"
    return out


def build_and_train(logs: pd.DataFrame, consensus: pd.DataFrame, target_season: int, target_week: int) -> tuple[MLBundle, pd.DataFrame]:
    training = build_training_frame(logs, target_season, target_week)
    bundle = train_models(training, target_season, target_week)
    current = build_current_features(logs, consensus, target_season, target_week)
    predictions = predict_current(bundle, current)
    return bundle, predictions


def save_bundle(bundle: MLBundle, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> MLBundle:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"ML v2 model artifact missing: {path}")
    return joblib.load(path)


def apply_ml_to_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach target-specific ML projections to sportsbook rows without blending them."""
    if metrics is None or metrics.empty:
        return metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame()
    out = metrics.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out["team"] = out["team"].map(canon_team)
    source = out["player_clean_key"] if "player_clean_key" in out.columns else out["player"]
    out["_ml_key"] = source.map(_key)
    keep = ["team", "player_clean_key", "ml_available", "ml_method", "ml_training_cutoff"] + [f"ml_{t}" for t in TARGET_COLUMNS]
    joined = predictions[keep].rename(columns={"player_clean_key": "_ml_key"})
    out = out.merge(joined, on=["team", "_ml_key"], how="left", validate="many_to_one")
    canonical_market = out["market"].astype(str).str.lower().map(lambda x: MARKET_MAP.get(x, x))
    out["ml_proj"] = [row.get(f"ml_{market}", np.nan) for (_, row), market in zip(out.iterrows(), canonical_market)]
    out["ml_applied"] = pd.to_numeric(out["ml_proj"], errors="coerce").notna().astype(int)
    out.drop(columns=["_ml_key"], inplace=True)
    return out
