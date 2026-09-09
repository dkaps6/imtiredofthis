from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.player_form_v2 import _normalize_weekly, _to_pandas

EPS = 0.02
FEATURES = [
    "prior_targets_pg", "prior_receptions_pg", "prior_target_share", "prior_rb_room_share",
    "prior_5plus_target_rate", "prior_7plus_target_rate",
    "last8_targets_pg", "last8_receptions_pg", "last8_target_share", "last8_rb_room_share",
    "prev_season_targets_pg", "prev_season_receptions_pg", "prev_season_target_share", "prev_season_rb_room_share",
    "same_team_prior_targets_pg", "same_team_prior_rb_room_share",
    "log1p_prior_games", "log1p_same_team_prior_games", "prev_season_available",
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _load_historical_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    out = _to_pandas(raw)
    if out.empty:
        raise RuntimeError(f"nflreadpy returned zero weekly player rows for {season}")
    return out


def _load_logs(seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        x = _normalize_weekly(_load_historical_weekly(int(season)), int(season))
        x["week"] = _num(x["week"]).astype("Int64")
        max_reg_week = 17 if int(season) <= 2020 else 18
        x = x.loc[x["week"].between(1, max_reg_week)].copy()
        frames.append(x)
        print(f"[rb_receiving_identity_runtime] loaded season={season} rows={len(x)}")
    if not frames:
        raise RuntimeError("no historical player logs loaded")
    logs = pd.concat(frames, ignore_index=True, sort=False)
    logs["position_family"] = logs.get("position", "").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
    logs["time_key"] = _num(logs["season"]).fillna(0).astype(int) * 100 + _num(logs["week"]).fillna(0).astype(int)
    for c in ("targets", "receptions", "rec_yards", "rushes", "tgt_share_game", "route_rate_game"):
        if c not in logs.columns:
            logs[c] = np.nan
        logs[c] = _num(logs[c])
    return logs.sort_values(["time_key", "team", "player_clean_key"]).reset_index(drop=True)


def _add_rb_room_share(logs: pd.DataFrame) -> pd.DataFrame:
    x = logs.copy()
    is_rb = x["position_family"].isin({"RB", "FB"})
    rb_tot = (
        x.loc[is_rb]
        .groupby(["season", "week", "team"], dropna=False)["targets"]
        .sum(min_count=1)
        .rename("rb_room_targets")
        .reset_index()
    )
    x = x.merge(rb_tot, on=["season", "week", "team"], how="left")
    x["rb_room_share_game"] = np.where(
        _num(x["rb_room_targets"]).fillna(0.0) > 0,
        _num(x["targets"]).fillna(0.0) / _num(x["rb_room_targets"]).replace(0, np.nan),
        0.0,
    )
    return x


def _series_state(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("time_key").copy()
    g["prior_games"] = np.arange(len(g), dtype=float)
    base_cols = {
        "targets": "targets_pg",
        "receptions": "receptions_pg",
        "rec_yards": "rec_yards_pg",
        "rushes": "carries_pg",
        "tgt_share_game": "target_share",
        "rb_room_share_game": "rb_room_share",
        "route_rate_game": "route_rate",
    }
    for raw, name in base_cols.items():
        s = _num(g[raw])
        g[f"prior_{name}"] = s.shift(1).expanding(min_periods=1).mean()
        g[f"last4_{name}"] = s.shift(1).rolling(4, min_periods=1).mean()
        g[f"last8_{name}"] = s.shift(1).rolling(8, min_periods=1).mean()
        g[f"after_{name}"] = s.expanding(min_periods=1).mean()
        g[f"after_last4_{name}"] = s.rolling(4, min_periods=1).mean()
        g[f"after_last8_{name}"] = s.rolling(8, min_periods=1).mean()
    flags = {
        "3plus_target_rate": _num(g["targets"]).fillna(0).ge(3).astype(float),
        "5plus_target_rate": _num(g["targets"]).fillna(0).ge(5).astype(float),
        "7plus_target_rate": _num(g["targets"]).fillna(0).ge(7).astype(float),
        "3plus_reception_rate": _num(g["receptions"]).fillna(0).ge(3).astype(float),
        "5plus_reception_rate": _num(g["receptions"]).fillna(0).ge(5).astype(float),
        "40plus_rec_yard_rate": _num(g["rec_yards"]).fillna(0).ge(40).astype(float),
        "60plus_rec_yard_rate": _num(g["rec_yards"]).fillna(0).ge(60).astype(float),
    }
    for name, s in flags.items():
        g[f"prior_{name}"] = s.shift(1).expanding(min_periods=1).mean()
        g[f"last8_{name}"] = s.shift(1).rolling(8, min_periods=1).mean()
        g[f"after_{name}"] = s.expanding(min_periods=1).mean()
        g[f"after_last8_{name}"] = s.rolling(8, min_periods=1).mean()
    g["after_games"] = np.arange(1, len(g) + 1, dtype=float)
    return g


def _build_states(rb: pd.DataFrame) -> pd.DataFrame:
    pieces = [_series_state(g) for _, g in rb.groupby("player_clean_key", sort=False, dropna=False)]
    out = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    out = out.sort_values(["player_clean_key", "team", "time_key"]).copy()
    for raw, name in (("targets", "targets_pg"), ("rb_room_share_game", "rb_room_share")):
        out[f"same_team_prior_{name}"] = out.groupby(["player_clean_key", "team"], dropna=False)[raw].transform(
            lambda s: _num(s).shift(1).expanding(min_periods=1).mean()
        )
        out[f"same_team_after_{name}"] = out.groupby(["player_clean_key", "team"], dropna=False)[raw].transform(
            lambda s: _num(s).expanding(min_periods=1).mean()
        )
    out["same_team_prior_games"] = out.groupby(["player_clean_key", "team"], dropna=False).cumcount().astype(float)
    out["same_team_after_games"] = out["same_team_prior_games"] + 1.0
    return out.sort_values(["player_clean_key", "time_key"]).reset_index(drop=True)


def _previous_season_features(rb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, season), g in rb.groupby(["player_clean_key", "season"], dropna=False):
        rows.append({
            "player_clean_key": key,
            "season": int(season) + 1,
            "prev_season_games": int(len(g)),
            "prev_season_targets_pg": float(_num(g["targets"]).mean()),
            "prev_season_receptions_pg": float(_num(g["receptions"]).mean()),
            "prev_season_rec_yards_pg": float(_num(g["rec_yards"]).mean()),
            "prev_season_carries_pg": float(_num(g["rushes"]).mean()),
            "prev_season_target_share": float(_num(g["tgt_share_game"]).mean()),
            "prev_season_rb_room_share": float(_num(g["rb_room_share_game"]).mean()),
            "prev_season_5plus_target_rate": float(_num(g["targets"]).fillna(0).ge(5).mean()),
            "prev_season_7plus_target_rate": float(_num(g["targets"]).fillna(0).ge(7).mean()),
            "prev_season_route_rate": float(_num(g["route_rate_game"]).mean()) if _num(g["route_rate_game"]).notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _snapshot_queries(queries: pd.DataFrame, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    q = queries.copy()
    q["time_key"] = _num(q["season"]).astype(int) * 100 + _num(q["week"]).astype(int)
    q["_qrow"] = np.arange(len(q))
    keep_map = {
        "after_games": "prior_games",
        "after_targets_pg": "prior_targets_pg",
        "after_receptions_pg": "prior_receptions_pg",
        "after_rec_yards_pg": "prior_rec_yards_pg",
        "after_carries_pg": "prior_carries_pg",
        "after_target_share": "prior_target_share",
        "after_rb_room_share": "prior_rb_room_share",
        "after_route_rate": "prior_route_rate",
        "after_last4_targets_pg": "last4_targets_pg",
        "after_last8_targets_pg": "last8_targets_pg",
        "after_last8_receptions_pg": "last8_receptions_pg",
        "after_last8_target_share": "last8_target_share",
        "after_last8_rb_room_share": "last8_rb_room_share",
        "after_last8_route_rate": "last8_route_rate",
        "after_3plus_target_rate": "prior_3plus_target_rate",
        "after_5plus_target_rate": "prior_5plus_target_rate",
        "after_7plus_target_rate": "prior_7plus_target_rate",
        "after_3plus_reception_rate": "prior_3plus_reception_rate",
        "after_5plus_reception_rate": "prior_5plus_reception_rate",
        "after_40plus_rec_yard_rate": "prior_40plus_rec_yard_rate",
        "after_60plus_rec_yard_rate": "prior_60plus_rec_yard_rate",
        "after_last8_5plus_target_rate": "last8_5plus_target_rate",
        "after_last8_7plus_target_rate": "last8_7plus_target_rate",
    }
    available = {k: v for k, v in keep_map.items() if k in states.columns}
    s = states[["player_clean_key", "time_key", *available.keys()]].copy().rename(columns=available)
    s = s.sort_values(["time_key", "player_clean_key"])
    qsort = q.sort_values(["time_key", "player_clean_key"])
    merged = pd.merge_asof(qsort, s, on="time_key", by="player_clean_key", direction="backward", allow_exact_matches=False)
    team_state = states[[
        "player_clean_key", "team", "time_key", "same_team_after_games",
        "same_team_after_targets_pg", "same_team_after_rb_room_share",
    ]].copy().rename(columns={
        "same_team_after_games": "same_team_prior_games",
        "same_team_after_targets_pg": "same_team_prior_targets_pg",
        "same_team_after_rb_room_share": "same_team_prior_rb_room_share",
    })
    team_state = team_state.sort_values(["time_key", "player_clean_key", "team"])
    merged = pd.merge_asof(
        merged.sort_values(["time_key", "player_clean_key", "team"]),
        team_state,
        on="time_key", by=["player_clean_key", "team"], direction="backward", allow_exact_matches=False,
        suffixes=("", "_same_team"),
    )
    if not prev.empty:
        merged = merged.merge(prev, on=["player_clean_key", "season"], how="left")
    return merged.sort_values("_qrow").drop(columns="_qrow").reset_index(drop=True)


def identity_atlas(history_start: int, through_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = _add_rb_room_share(_load_logs(list(range(history_start, through_season + 1))))
    rb = logs.loc[logs.position_family.isin({"RB", "FB"})].copy()
    return _build_states(rb), _previous_season_features(rb)


def attach_identity(rb: pd.DataFrame, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    q = rb[["player_clean_key", "team"]].copy()
    q["season"] = int(season)
    q["week"] = int(week)
    feat = _snapshot_queries(q, states, prev)
    keep = ["player_clean_key", "team", "season", "week"] + [c for c in feat.columns if c in set(FEATURES) | {
        "prior_games", "same_team_prior_games", "prev_season_games", "prior_rb_room_share"
    }]
    feat = feat[keep].copy()
    feat["log1p_prior_games"] = np.log1p(pd.to_numeric(feat.get("prior_games", 0), errors="coerce").fillna(0).clip(lower=0))
    feat["log1p_same_team_prior_games"] = np.log1p(pd.to_numeric(feat.get("same_team_prior_games", 0), errors="coerce").fillna(0).clip(lower=0))
    feat["prev_season_available"] = pd.to_numeric(feat.get("prev_season_games", np.nan), errors="coerce").notna().astype(float)
    for c in FEATURES:
        if c not in feat.columns:
            feat[c] = 0.0
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    out = rb.merge(feat[["player_clean_key", "team", *FEATURES]], on=["player_clean_key", "team"], how="left", validate="one_to_one")
    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out
