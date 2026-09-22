#!/usr/bin/env python3
"""Current-Season State Persistence V1.

Diagnostic only. Compares prior-season, current-season-to-date, and the exact
PlayerForm four-game pseudo-prior blend as predictors of the next game's
production-aligned player metric.

2026 outcomes are never loaded. A separate 2026 snap-count source audit checks
live source readiness without using snaps as outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scripts.modeling.te_r5p_entitlement_adapter_v1 import SOURCE_SEASONS as TE_R5P_SNAP_SOURCE_SEASONS
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _team as canon_snap_team
from scripts.player_form_v2 import _normalize_weekly
from scripts.player_stats_loader_v2 import load_weekly_player_stats


EVAL_SEASONS = [2022, 2023, 2024, 2025]
LOAD_SEASONS = [2021, 2022, 2023, 2024, 2025]
PSEUDO_PRIOR_GAMES = 4.0

METRICS: dict[str, list[tuple[str, str]]] = {
    "QB": [("ypa", "ypa_game")],
    "RB": [
        ("rush_share", "rush_share_game"),
        ("tgt_share", "tgt_share_game"),
        ("ypc", "ypc_game"),
    ],
    "WR": [
        ("tgt_share", "tgt_share_game"),
        ("ypt", "ypt_game"),
        ("receptions_per_target", "catch_rate_game"),
    ],
    "TE": [
        ("tgt_share", "tgt_share_game"),
        ("ypt", "ypt_game"),
        ("receptions_per_target", "catch_rate_game"),
    ],
}

SUM_COLUMNS = [
    "targets", "receptions", "rec_yards", "rushes", "rush_yards",
    "pass_att", "pass_yards", "team_targets", "team_rushes",
]


def _load_logs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in LOAD_SEASONS:
        raw = load_weekly_player_stats(season)
        normalized = _normalize_weekly(raw, season)
        normalized = normalized.loc[
            pd.to_numeric(normalized["week"], errors="coerce").between(1, 18)
        ].copy()
        frames.append(normalized)
    logs = pd.concat(frames, ignore_index=True, sort=False)
    if logs.empty:
        raise RuntimeError("current-season persistence input has zero normalized rows")
    if "player_identity_key" not in logs.columns:
        raise RuntimeError("normalized weekly logs missing player_identity_key")
    if logs["player_identity_key"].astype(str).eq("").any():
        raise RuntimeError("normalized weekly logs contain blank stable identity")
    return logs


def _aggregate_history(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=[
            "player_identity_key", "history_games", "tgt_share", "rush_share",
            "ypc", "ypa", "ypt", "receptions_per_target",
        ])
    missing = set(SUM_COLUMNS + ["player_identity_key", "week"]) - set(logs.columns)
    if missing:
        raise RuntimeError(f"history aggregation missing columns: {sorted(missing)}")
    g = logs.groupby("player_identity_key", dropna=False)
    agg = g.agg(
        history_games=("week", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rec_yards=("rec_yards", "sum"),
        rushes=("rushes", "sum"),
        rush_yards=("rush_yards", "sum"),
        pass_att=("pass_att", "sum"),
        pass_yards=("pass_yards", "sum"),
        team_targets=("team_targets", "sum"),
        team_rushes=("team_rushes", "sum"),
    ).reset_index()
    agg["tgt_share"] = np.where(
        agg["team_targets"] > 0, agg["targets"] / agg["team_targets"], np.nan
    )
    agg["rush_share"] = np.where(
        agg["team_rushes"] > 0, agg["rushes"] / agg["team_rushes"], np.nan
    )
    agg["ypc"] = np.where(agg["rushes"] > 0, agg["rush_yards"] / agg["rushes"], np.nan)
    agg["ypa"] = np.where(agg["pass_att"] > 0, agg["pass_yards"] / agg["pass_att"], np.nan)
    agg["ypt"] = np.where(agg["targets"] > 0, agg["rec_yards"] / agg["targets"], np.nan)
    agg["receptions_per_target"] = np.where(
        agg["targets"] > 0, agg["receptions"] / agg["targets"], np.nan
    )
    return agg


def _target_rows(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    q = logs.loc[
        pd.to_numeric(logs["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(logs["week"], errors="coerce").eq(int(week))
    ].copy()
    if q.empty:
        return q
    # Weekly stats are expected one row per stable identity/week. Fail closed on
    # conflicting duplicates rather than silently averaging target-game truth.
    dup = q.duplicated(["player_identity_key"], keep=False)
    if dup.any():
        sample = q.loc[dup, ["player_identity_key", "player", "team", "week"]].head(20).to_dict("records")
        raise RuntimeError(f"duplicate target-week player identity rows: {sample}")
    return q


def build_persistence_panel(logs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    seasons = pd.to_numeric(logs["season"], errors="coerce")
    weeks = pd.to_numeric(logs["week"], errors="coerce")

    for season in EVAL_SEASONS:
        prior_logs = logs.loc[seasons.eq(season - 1)].copy()
        prior = _aggregate_history(prior_logs).rename(
            columns={
                "history_games": "prior_games",
                **{m: f"{m}_prior" for pairs in METRICS.values() for m, _ in pairs},
            }
        )
        for week in range(2, 19):
            target = _target_rows(logs, season, week)
            if target.empty:
                continue
            current_logs = logs.loc[seasons.eq(season) & weeks.lt(week)].copy()
            current = _aggregate_history(current_logs).rename(
                columns={
                    "history_games": "current_games",
                    **{m: f"{m}_current" for pairs in METRICS.values() for m, _ in pairs},
                }
            )
            joined = target.merge(prior, on="player_identity_key", how="left", validate="one_to_one")
            joined = joined.merge(current, on="player_identity_key", how="left", validate="one_to_one")

            for _, rec in joined.iterrows():
                pos_raw = rec.get("position")
                pos = "" if pd.isna(pos_raw) else str(pos_raw).upper().strip()
                if pos not in METRICS:
                    continue
                for metric, actual_col in METRICS[pos]:
                    prior_v = pd.to_numeric(pd.Series([rec.get(f"{metric}_prior")]), errors="coerce").iloc[0]
                    current_v = pd.to_numeric(pd.Series([rec.get(f"{metric}_current")]), errors="coerce").iloc[0]
                    actual_v = pd.to_numeric(pd.Series([rec.get(actual_col)]), errors="coerce").iloc[0]
                    current_games = pd.to_numeric(pd.Series([rec.get("current_games")]), errors="coerce").iloc[0]
                    prior_games = pd.to_numeric(pd.Series([rec.get("prior_games")]), errors="coerce").iloc[0]
                    if any(pd.isna(v) for v in (prior_v, current_v, actual_v, current_games, prior_games)):
                        continue
                    if float(current_games) < 1 or float(prior_games) < 1:
                        continue
                    w_current = float(current_games) / (float(current_games) + PSEUDO_PRIOR_GAMES)
                    blend = (1.0 - w_current) * float(prior_v) + w_current * float(current_v)
                    rows.append({
                        "season": int(season),
                        "target_week": int(week),
                        "player_identity_key": str(rec["player_identity_key"]),
                        "player": "" if pd.isna(rec.get("player")) else str(rec.get("player")),
                        "team": "" if pd.isna(rec.get("team")) else str(rec.get("team")),
                        "position": pos,
                        "metric": metric,
                        "prior_games": int(prior_games),
                        "current_games": int(current_games),
                        "prior_value": float(prior_v),
                        "current_value": float(current_v),
                        "blend4_value": float(blend),
                        "actual_value": float(actual_v),
                        "w_current_blend4": float(w_current),
                        "state_delta_current_vs_prior": float(current_v - prior_v),
                        "next_delta_vs_prior": float(actual_v - prior_v),
                    })
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("current-season persistence panel has zero eligible rows")
    if panel.duplicated(["season", "target_week", "player_identity_key", "metric"]).any():
        raise RuntimeError("persistence panel contains duplicate player-week-metric rows")
    return panel


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    q = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(q) < 3 or q["x"].nunique() < 2 or q["y"].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(pearsonr(q["x"], q["y"]).statistic)
    return float(spearmanr(q["x"], q["y"]).statistic)


def _game_bucket(n: int) -> str:
    if n <= 4:
        return str(int(n))
    if n <= 8:
        return "5-8"
    return "9+"


def _summary_row(q: pd.DataFrame, label: str, position: str, metric: str, bucket: str = "ALL") -> dict:
    out: dict[str, object] = {
        "sample": label,
        "position": position,
        "metric": metric,
        "current_games_bucket": bucket,
        "n": int(len(q)),
    }
    actual = q["actual_value"].astype(float)
    for name, col in [
        ("prior", "prior_value"),
        ("current", "current_value"),
        ("blend4", "blend4_value"),
    ]:
        pred = q[col].astype(float)
        err = pred - actual
        out[f"{name}_mae"] = float(np.abs(err).mean())
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
        out[f"{name}_bias"] = float(err.mean())
        out[f"{name}_pearson"] = _corr(pred, actual, "pearson")
        out[f"{name}_spearman"] = _corr(pred, actual, "spearman")
    d1 = q["state_delta_current_vs_prior"].astype(float)
    d2 = q["next_delta_vs_prior"].astype(float)
    out["delta_pearson"] = _corr(d1, d2, "pearson")
    out["delta_spearman"] = _corr(d1, d2, "spearman")
    nonzero = d1.ne(0) & d2.ne(0)
    out["delta_sign_n"] = int(nonzero.sum())
    out["delta_sign_agreement"] = (
        float(np.sign(d1.loc[nonzero]).eq(np.sign(d2.loc[nonzero])).mean())
        if nonzero.any()
        else np.nan
    )
    out["blend4_mae_gain_vs_prior"] = float(out["prior_mae"] - out["blend4_mae"])
    out["current_mae_gain_vs_prior"] = float(out["prior_mae"] - out["current_mae"])
    out["blend4_not_best"] = bool(out["current_mae"] < out["blend4_mae"])
    return out


def summarize_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    buckets: list[dict] = []

    samples: list[tuple[str, pd.DataFrame]] = []
    for season in EVAL_SEASONS:
        samples.append((str(season), panel.loc[panel["season"].eq(season)].copy()))
    samples.extend([
        ("DEV_2022_2024", panel.loc[panel["season"].between(2022, 2024)].copy()),
        ("REPLICATION_2025", panel.loc[panel["season"].eq(2025)].copy()),
        ("POOLED_ALL", panel.copy()),
    ])

    for label, frame in samples:
        for (position, metric), q in frame.groupby(["position", "metric"], sort=True):
            summaries.append(_summary_row(q, label, position, metric))

    with_bucket = panel.copy()
    with_bucket["current_games_bucket"] = with_bucket["current_games"].map(_game_bucket)
    for (season, position, metric, bucket), q in with_bucket.groupby(
        ["season", "position", "metric", "current_games_bucket"], sort=True
    ):
        buckets.append(_summary_row(q, str(int(season)), position, metric, bucket))

    return pd.DataFrame(summaries), pd.DataFrame(buckets)


def audit_2026_snap_source() -> dict:
    result: dict[str, object] = {
        "source": "nflreadpy.load_snap_counts",
        "production_te_r5p_source_seasons": [int(v) for v in TE_R5P_SNAP_SOURCE_SEASONS],
        "production_wr_r15_uses_shared_te_r5p_loader": True,
        "production_source_includes_2026": 2026 in set(int(v) for v in TE_R5P_SNAP_SOURCE_SEASONS),
    }
    try:
        import nflreadpy as nfl

        raw = nfl.load_snap_counts(seasons=[2026])
        snaps = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        snaps.columns = [str(c).strip().lower() for c in snaps.columns]
        if snaps.empty:
            raise RuntimeError("2026 snap source returned zero rows")
        if "week" not in snaps.columns:
            raise RuntimeError("2026 snap source missing week")
        snaps["week"] = pd.to_numeric(snaps["week"], errors="coerce")
        snaps = snaps.loc[snaps["week"].between(1, 18)].copy()
        team_col = next((c for c in ("team", "team_abbr", "club") if c in snaps.columns), None)
        if team_col is None:
            raise RuntimeError("2026 snap source missing team field")
        snaps["_team"] = snaps[team_col].map(canon_snap_team)
        offense_snaps_col = next((c for c in ("offense_snaps",) if c in snaps.columns), None)
        offense_pct_col = next((c for c in ("offense_pct", "offense_percentage") if c in snaps.columns), None)
        weeks = sorted(int(v) for v in snaps["week"].dropna().unique())
        per_week = []
        for week, q in snaps.groupby("week", sort=True):
            per_week.append({
                "week": int(week),
                "rows": int(len(q)),
                "teams": int(q["_team"].replace("", np.nan).nunique()),
                "offense_snaps_nonnull": int(pd.to_numeric(q[offense_snaps_col], errors="coerce").notna().sum()) if offense_snaps_col else 0,
                "offense_pct_nonnull": int(pd.to_numeric(q[offense_pct_col], errors="coerce").notna().sum()) if offense_pct_col else 0,
            })
        result.update({
            "available": True,
            "weeks_available": weeks,
            "rows": int(len(snaps)),
            "teams": int(snaps["_team"].replace("", np.nan).nunique()),
            "per_week": per_week,
        })
        if 1 in weeks and 2 in weeks and int(result["teams"]) == 32 and not result["production_source_includes_2026"]:
            result["disposition"] = "CURRENT_2026_SNAP_SOURCE_AVAILABLE_NOT_CONSUMED"
        elif not result["production_source_includes_2026"]:
            result["disposition"] = "CURRENT_2026_SNAP_SOURCE_PARTIAL_NOT_CONSUMED"
        else:
            result["disposition"] = "CURRENT_2026_SNAP_SOURCE_CONSUMED"
    except Exception as exc:
        result.update({
            "available": False,
            "error": str(exc),
            "disposition": "CURRENT_2026_SNAP_SOURCE_UNAVAILABLE_OR_UNVERIFIED",
        })
    return result


def build_result(summary: pd.DataFrame, snap_audit: dict) -> dict:
    repl = summary.loc[summary["sample"].eq("REPLICATION_2025")].copy()
    rows = []
    for _, r in repl.iterrows():
        replicates = bool(
            float(r["blend4_mae_gain_vs_prior"]) > 0
            and pd.notna(r["delta_spearman"])
            and float(r["delta_spearman"]) > 0
        )
        rows.append({
            "position": str(r["position"]),
            "metric": str(r["metric"]),
            "n": int(r["n"]),
            "prior_mae": float(r["prior_mae"]),
            "current_mae": float(r["current_mae"]),
            "blend4_mae": float(r["blend4_mae"]),
            "blend4_mae_gain_vs_prior": float(r["blend4_mae_gain_vs_prior"]),
            "delta_spearman": None if pd.isna(r["delta_spearman"]) else float(r["delta_spearman"]),
            "blend4_not_best": bool(r["blend4_not_best"]),
            "disposition": "CURRENT_SEASON_SIGNAL_REPLICATES" if replicates else "NO_REPLICATED_CURRENT_SIGNAL",
        })
    return {
        "version": "CURRENT_SEASON_STATE_PERSISTENCE_V1",
        "plan_disposition": "CURRENT_SEASON_STATE_PERSISTENCE_V1_PLAN_FROZEN",
        "evaluation_seasons": EVAL_SEASONS,
        "development_seasons": [2022, 2023, 2024],
        "replication_season": 2025,
        "playerform_pseudo_prior_games": PSEUDO_PRIOR_GAMES,
        "sportsbook_inputs_used": 0,
        "outcomes_2026_used": 0,
        "production_changed": 0,
        "replication": rows,
        "snap_source_audit_2026": snap_audit,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/research/current_season_state_persistence_v1"))
    a = p.parse_args()

    logs = _load_logs()
    panel = build_persistence_panel(logs)
    summary, buckets = summarize_panel(panel)
    snap_audit = audit_2026_snap_source()
    result = build_result(summary, snap_audit)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(a.out_dir / "current_season_state_persistence_panel.csv", index=False)
    summary.to_csv(a.out_dir / "current_season_state_persistence_summary.csv", index=False)
    buckets.to_csv(a.out_dir / "current_season_state_persistence_game_count_summary.csv", index=False)
    (a.out_dir / "current_season_snap_source_audit_2026.json").write_text(
        json.dumps(snap_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (a.out_dir / "current_season_state_persistence_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps({
        "panel_rows": int(len(panel)),
        "summary_rows": int(len(summary)),
        "game_count_summary_rows": int(len(buckets)),
        "replication_rows": len(result["replication"]),
        "snap_disposition": snap_audit.get("disposition"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
