#!/usr/bin/env python3
"""Diagnostic-only RB receiving identity and catastrophic-miss atlas.

Purpose
-------
Test the football hypothesis that RB receiving opportunity is concentrated among
persistent player/team receiving roles rather than being a universal RB trait.
All candidate signals are built strictly from games completed before the game
being evaluated. Current-game outcomes are used only as diagnostic labels.
Sportsbook data is neither loaded nor used.

This script does NOT fit or promote a production model. It answers:
1) Do pregame receiving-history signals concentrate future 5+/7+ target games?
2) Do those same signals concentrate 40+/60+ receiving-yard games?
3) In a supplied current-model decomposition, are 30+/50+ receiving-yard misses
   concentrated among historically receiving-heavy backs?
4) How does ordinary-game MAE differ from the catastrophic tail by receiving
   identity quantile?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.historical_player_logs import _load_historical_weekly
from scripts.player_form_v2 import _normalize_weekly

RB_POS = {"RB", "FB", "HB", "TB"}

HISTORY_FEATURES = [
    "prior_targets_pg",
    "prior_receptions_pg",
    "prior_target_share",
    "prior_rb_room_share",
    "prior_5plus_target_rate",
    "prior_7plus_target_rate",
    "last8_targets_pg",
    "last8_receptions_pg",
    "last8_target_share",
    "last8_rb_room_share",
    "prev_season_targets_pg",
    "prev_season_receptions_pg",
    "prev_season_target_share",
    "prev_season_rb_room_share",
    "same_team_prior_targets_pg",
    "same_team_prior_rb_room_share",
    "prior_route_rate",
    "last8_route_rate",
]


def _parse_seasons(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _load_logs(seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        x = _normalize_weekly(_load_historical_weekly(int(season)), int(season))
        x["week"] = _num(x["week"]).astype("Int64")
        # The 17-game regular season began in 2021. nflverse weekly data may
        # expose postseason rows with numeric week 18 for older seasons.
        max_reg_week = 17 if int(season) <= 2020 else 18
        x = x.loc[x["week"].between(1, max_reg_week)].copy()
        frames.append(x)
        print(f"[rb-recv-identity] loaded season={season} rows={len(x)}")
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
    """Return pregame features plus after-game state for one player."""
    g = g.sort_values("time_key").copy()
    n = np.arange(len(g), dtype=float)
    g["prior_games"] = n

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
    pieces = []
    for _, g in rb.groupby("player_clean_key", sort=False, dropna=False):
        pieces.append(_series_state(g))
    out = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()

    # Same-team role history: important after trades / backfield changes.
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
    """Attach state from the last completed game strictly before each query."""
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
    merged = pd.merge_asof(
        qsort,
        s,
        on="time_key",
        by="player_clean_key",
        direction="backward",
        allow_exact_matches=False,
    )

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
        on="time_key",
        by=["player_clean_key", "team"],
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_same_team"),
    )

    if not prev.empty:
        merged = merged.merge(prev, on=["player_clean_key", "season"], how="left")
    return merged.sort_values("_qrow").drop(columns="_qrow").reset_index(drop=True)


def _pct_rank(frame: pd.DataFrame, feature: str) -> pd.Series:
    x = _num(frame[feature])
    if {"season", "week"}.issubset(frame.columns):
        return frame.assign(_x=x).groupby(["season", "week"], dropna=False)["_x"].rank(pct=True, method="average")
    return x.rank(pct=True, method="average")


def _safe_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def _capture(mask: pd.Series, top: pd.Series) -> float:
    denom = int(mask.fillna(False).sum())
    return float((mask.fillna(False) & top.fillna(False)).sum() / denom) if denom else np.nan


def _lift(mask: pd.Series, top: pd.Series) -> tuple[float, float, float]:
    m = mask.fillna(False)
    t = top.fillna(False)
    a = _safe_rate(m.loc[t]) if t.any() else np.nan
    b = _safe_rate(m.loc[~t]) if (~t).any() else np.nan
    lift = float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan
    return a, b, lift


def _feature_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in HISTORY_FEATURES:
        if feature not in atlas.columns:
            continue
        valid = _num(atlas[feature]).notna() & _num(atlas.get("prior_games", pd.Series(np.nan, index=atlas.index))).ge(1)
        g = atlas.loc[valid].copy()
        if len(g) < 100:
            continue
        pct = _pct_rank(g, feature)
        top20 = pct.gt(0.80)
        top10 = pct.gt(0.90)
        events = {
            "targets_5plus": _num(g["actual_targets"]).ge(5),
            "targets_7plus": _num(g["actual_targets"]).ge(7),
            "rec_yards_40plus": _num(g["actual_rec_yards"]).ge(40),
            "rec_yards_60plus": _num(g["actual_rec_yards"]).ge(60),
        }
        row = {
            "feature": feature,
            "n": int(len(g)),
            "coverage": float(len(g) / max(1, len(atlas))),
            "spearman_targets": float(_num(g[feature]).corr(_num(g["actual_targets"]), method="spearman")),
            "spearman_rec_yards": float(_num(g[feature]).corr(_num(g["actual_rec_yards"]), method="spearman")),
        }
        for label, mask in events.items():
            a, b, lift = _lift(mask, top20)
            row[f"top20_capture_{label}"] = _capture(mask, top20)
            row[f"top20_rate_{label}"] = a
            row[f"rest_rate_{label}"] = b
            row[f"top20_lift_{label}"] = lift
            row[f"top10_capture_{label}"] = _capture(mask, top10)
        if "cat30" in g.columns:
            for label in ("cat30", "cat50"):
                mask = g[label].fillna(False).astype(bool)
                a, b, lift = _lift(mask, top20)
                row[f"top20_capture_{label}"] = _capture(mask, top20)
                row[f"top20_rate_{label}"] = a
                row[f"rest_rate_{label}"] = b
                row[f"top20_lift_{label}"] = lift
                row[f"top10_capture_{label}"] = _capture(mask, top10)
        rows.append(row)
    return pd.DataFrame(rows)


def _decile_atlas(atlas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in HISTORY_FEATURES:
        if feature not in atlas.columns:
            continue
        g = atlas.loc[_num(atlas[feature]).notna()].copy()
        if len(g) < 100:
            continue
        pct = _pct_rank(g, feature)
        g["quantile"] = np.minimum(5, np.maximum(1, np.ceil(pct * 5))).astype(int)
        for q, h in g.groupby("quantile"):
            row = {
                "feature": feature,
                "quintile": int(q),
                "n": int(len(h)),
                "feature_mean": float(_num(h[feature]).mean()),
                "actual_targets_mean": float(_num(h["actual_targets"]).mean()),
                "actual_rec_yards_mean": float(_num(h["actual_rec_yards"]).mean()),
                "target_5plus_rate": _safe_rate(_num(h["actual_targets"]).ge(5)),
                "target_7plus_rate": _safe_rate(_num(h["actual_targets"]).ge(7)),
                "rec_yards_40plus_rate": _safe_rate(_num(h["actual_rec_yards"]).ge(40)),
                "rec_yards_60plus_rate": _safe_rate(_num(h["actual_rec_yards"]).ge(60)),
            }
            if "abs_error" in h.columns:
                row.update({
                    "baseline_mae": float(_num(h["abs_error"]).mean()),
                    "baseline_bias": float(_num(h["signed_error"]).mean()),
                    "cat30_rate": _safe_rate(h["cat30"].astype(bool)),
                    "cat50_rate": _safe_rate(h["cat50"].astype(bool)),
                    "absolute_error_share": float(_num(h["abs_error"]).sum() / max(1e-12, _num(g["abs_error"]).sum())),
                    "cat30_share": float(h["cat30"].sum() / max(1, g["cat30"].sum())),
                    "cat50_share": float(h["cat50"].sum() / max(1, g["cat50"].sum())),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _overall_error_summary(casebook: pd.DataFrame) -> dict:
    if casebook.empty:
        return {}
    e = _num(casebook["abs_error"]).dropna()
    return {
        "n": int(len(e)),
        "mae": float(e.mean()),
        "p90_abs_error": float(e.quantile(0.90)),
        "cat30_count": int((e >= 30).sum()),
        "cat30_rate": float((e >= 30).mean()),
        "cat50_count": int((e >= 50).sum()),
        "cat50_rate": float((e >= 50).mean()),
        "mae_excluding_cat30": float(e.loc[e < 30].mean()),
        "mae_excluding_cat50": float(e.loc[e < 50].mean()),
        "mae_excluding_worst_10pct": float(e.loc[e <= e.quantile(0.90)].mean()),
        "cat30_absolute_error_share": float(e.loc[e >= 30].sum() / e.sum()) if e.sum() > 0 else np.nan,
        "cat50_absolute_error_share": float(e.loc[e >= 50].sum() / e.sum()) if e.sum() > 0 else np.nan,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history-seasons", default="2020-2025")
    p.add_argument("--eval-seasons", default="2022-2025")
    p.add_argument("--decomposition", type=Path, default=Path("data/backtests/rb_receiving_decomposition_current_v1/rb_receiving_player_decomposition.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_receiving_identity_diagnostic_v1"))
    a = p.parse_args()

    history_seasons = _parse_seasons(a.history_seasons)
    eval_seasons = _parse_seasons(a.eval_seasons)
    logs = _add_rb_room_share(_load_logs(history_seasons))
    rb = logs.loc[logs["position_family"].isin({"RB", "FB"})].copy()
    if rb.empty:
        raise RuntimeError("no RB/FB rows in historical logs")

    states = _build_states(rb)
    prev = _previous_season_features(rb)

    atlas = states.loc[states["season"].isin(eval_seasons)].copy()
    atlas = atlas.rename(columns={
        "targets": "actual_targets",
        "receptions": "actual_receptions",
        "rec_yards": "actual_rec_yards",
    })
    atlas = atlas.merge(prev, on=["player_clean_key", "season"], how="left")

    casebook = pd.DataFrame()
    if a.decomposition.exists() and a.decomposition.stat().st_size > 0:
        d = pd.read_csv(a.decomposition)
        required = {"season", "week", "team", "player_clean_key", "actual_targets", "actual_rec_yards", "baseline_targets", "baseline_rec_yards"}
        if not required.issubset(d.columns):
            raise RuntimeError(f"decomposition missing columns: {sorted(required - set(d.columns))}")
        casebook = _snapshot_queries(d, states, prev)
        casebook["signed_error"] = _num(casebook["baseline_rec_yards"]) - _num(casebook["actual_rec_yards"])
        casebook["abs_error"] = casebook["signed_error"].abs()
        casebook["cat20"] = casebook["abs_error"].ge(20)
        casebook["cat30"] = casebook["abs_error"].ge(30)
        casebook["cat40"] = casebook["abs_error"].ge(40)
        casebook["cat50"] = casebook["abs_error"].ge(50)
        casebook["target_error"] = _num(casebook["baseline_targets"]) - _num(casebook["actual_targets"])

        labels = casebook[["season", "week", "team", "player_clean_key", "cat30", "cat50"]].drop_duplicates()
        atlas = atlas.merge(labels, on=["season", "week", "team", "player_clean_key"], how="left")

    usage_feature_summary = _feature_summary(atlas)
    case_feature_summary = _feature_summary(casebook) if not casebook.empty else pd.DataFrame()
    quintiles = _decile_atlas(casebook if not casebook.empty else atlas)

    player_cat = pd.DataFrame()
    if not casebook.empty:
        player_cat = (
            casebook.groupby(["player_clean_key", "player"], dropna=False)
            .agg(
                games=("week", "size"),
                cat20=("cat20", "sum"),
                cat30=("cat30", "sum"),
                cat40=("cat40", "sum"),
                cat50=("cat50", "sum"),
                absolute_error=("abs_error", "sum"),
                mean_abs_error=("abs_error", "mean"),
                mean_actual_targets=("actual_targets", "mean"),
                mean_baseline_targets=("baseline_targets", "mean"),
                prior_targets_pg=("prior_targets_pg", "mean"),
                prior_rb_room_share=("prior_rb_room_share", "mean"),
            )
            .reset_index()
            .sort_values(["cat30", "absolute_error"], ascending=[False, False])
        )

    route_coverage = float(_num(atlas.get("prior_route_rate", pd.Series(dtype=float))).notna().mean()) if len(atlas) else 0.0
    result = {
        "diagnostic": "RB_RECEIVING_IDENTITY_DIAGNOSTIC_V1",
        "disposition": "DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE",
        "history_seasons": history_seasons,
        "eval_seasons": eval_seasons,
        "historical_rb_player_games": int(len(rb)),
        "eval_rb_player_games": int(len(atlas)),
        "sportsbook_inputs_used": 0,
        "current_future_outcomes_used_in_features": 0,
        "feature_contract": "strictly games with season/week earlier than evaluated game; previous-season completed aggregates only",
        "route_history_coverage": route_coverage,
        "baseline_error_summary": _overall_error_summary(casebook),
    }
    if not usage_feature_summary.empty:
        fs = usage_feature_summary.copy()
        if not case_feature_summary.empty:
            cat_cols = [c for c in ("feature", "top20_capture_cat30", "top20_lift_cat30", "top10_capture_cat30", "top20_capture_cat50", "top20_lift_cat50") if c in case_feature_summary.columns]
            fs = fs.merge(case_feature_summary[cat_cols], on="feature", how="left", suffixes=("", "_case"))
        for c in ("top20_capture_targets_7plus", "top20_capture_cat30"):
            if c not in fs.columns:
                fs[c] = np.nan
        fs["diagnostic_rank_score"] = fs[["top20_capture_targets_7plus", "top20_capture_cat30"]].mean(axis=1, skipna=True)
        top = fs.sort_values("diagnostic_rank_score", ascending=False).head(8)
        result["top_descriptive_signals"] = top[[
            "feature", "n", "top20_capture_targets_5plus", "top20_capture_targets_7plus",
            "top20_lift_targets_7plus", "top20_capture_cat30", "top20_lift_cat30",
        ]].replace({np.nan: None}).to_dict(orient="records")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(a.out_dir / "rb_receiving_actual_usage_atlas.csv", index=False)
    usage_feature_summary.to_csv(a.out_dir / "rb_receiving_usage_history_signal_summary.csv", index=False)
    if not case_feature_summary.empty:
        case_feature_summary.to_csv(a.out_dir / "rb_receiving_catastrophe_history_signal_summary.csv", index=False)
    quintiles.to_csv(a.out_dir / "rb_receiving_history_quintile_atlas.csv", index=False)
    if not casebook.empty:
        casebook.sort_values("abs_error", ascending=False).to_csv(a.out_dir / "rb_receiving_catastrophic_casebook.csv", index=False)
        player_cat.to_csv(a.out_dir / "rb_receiving_catastrophic_player_concentration.csv", index=False)
    (a.out_dir / "rb_receiving_identity_diagnostic_result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    print("\n=== RB receiving identity diagnostic ===")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not usage_feature_summary.empty:
        show = [c for c in [
            "feature", "n", "spearman_targets", "top20_capture_targets_5plus",
            "top20_capture_targets_7plus", "top20_lift_targets_7plus",
        ] if c in usage_feature_summary.columns]
        print("\n=== usage history signal summary ===")
        print(usage_feature_summary.sort_values("top20_capture_targets_7plus", ascending=False)[show].to_string(index=False))
        if not case_feature_summary.empty:
            cshow = [c for c in ["feature", "n", "top20_capture_cat30", "top20_lift_cat30", "top20_capture_cat50", "top20_lift_cat50"] if c in case_feature_summary.columns]
            print("\n=== catastrophic history signal summary ===")
            print(case_feature_summary.sort_values("top20_capture_cat30", ascending=False)[cshow].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
