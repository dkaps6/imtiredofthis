#!/usr/bin/env python3
"""WR Phase 4C Stage-1 predicted realized pass-volume increment V1.

Research-only implementation of the frozen Stage-1 A/B/C design.

The script supports two explicit modes:
  * --preflight-only: build and validate source/temporal mechanics, generate
    forward-chained script predictions, fit the 2023 A/B/C residual models,
    and emit sealed 2024 predictions WITHOUT reading/scoring 2024 team-target
    outcomes or realized-pass-volume outcomes.
  * --run-outcomes: expose the prospectively frozen blind-2024 metrics and
    disposition. This mode is not authorized until implementation review.

No production/model/threshold changes are made by this script.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team
from scripts.backtest.historical_inputs import build_team_weekly_from_pbp

TG = ["season", "week", "team"]
WINDOW = 8
MIN_PRIOR = 3
RIDGE_ALPHA = 20.0
BOOT_REPS = 10_000
BOOT_SEED = 20260915
MATERIALITY_TARGETS = 0.10
BIAS_TOLERANCE = 0.25
EXPECTED_LAYER23 = {2023: 510, 2024: 515}
RAW_LAYER1_TAILS_2024 = {
    "UNDERPROJECT_30_PLUS_OPP_DOM": 222,
    "ACTUAL_100_PLUS_OPP_DOM": 87,
    "UNDERPROJECT_30_PLUS": 312,
}
EXPECTED_TAILS_2024 = {
    "UNDERPROJECT_30_PLUS_OPP_DOM": 208,
    "ACTUAL_100_PLUS_OPP_DOM": 85,
    "UNDERPROJECT_30_PLUS": 294,
}
SCRIPT_FEATURES = [
    "team_plays_prior8",
    "team_dropback_rate_prior8",
    "opp_plays_allowed_prior8",
    "opp_dropback_rate_allowed_prior8",
    "home_flag",
    "rest_diff",
]


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _numeric(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")


def load_schedule_team_rows(seasons: Sequence[int]) -> pd.DataFrame:
    """Load pregame schedule geometry at team-game grain."""
    import nflreadpy as nfl

    rows: list[dict] = []
    for season in sorted(set(int(s) for s in seasons)):
        raw = _to_pandas(nfl.load_schedules(int(season)))
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        if "game_type" in raw.columns:
            raw = raw.loc[raw["game_type"].astype(str).str.upper().eq("REG")].copy()
        elif "season_type" in raw.columns:
            raw = raw.loc[raw["season_type"].astype(str).str.upper().eq("REG")].copy()
        required = {
            "season", "week", "home_team", "away_team", "total_line",
            "home_rest", "away_rest",
        }
        missing = required - set(raw.columns)
        if missing:
            raise RuntimeError(f"schedule {season} missing required Stage-1 columns: {sorted(missing)}")
        for _, r in raw.iterrows():
            home = canon_team(r["home_team"])
            away = canon_team(r["away_team"])
            game_id = str(r.get("game_id", "") or "").strip()
            if not game_id:
                game_id = f"{int(r['season'])}_{int(r['week']):02d}_{away}_{home}"
            total = pd.to_numeric(pd.Series([r["total_line"]]), errors="coerce").iloc[0]
            hrest = pd.to_numeric(pd.Series([r["home_rest"]]), errors="coerce").iloc[0]
            arest = pd.to_numeric(pd.Series([r["away_rest"]]), errors="coerce").iloc[0]
            rows.extend([
                {
                    "season": int(r["season"]), "week": int(r["week"]),
                    "team": home, "opponent": away, "game_id": game_id,
                    "home_flag": 1.0, "team_rest": hrest, "opp_rest": arest,
                    "market_total": total,
                },
                {
                    "season": int(r["season"]), "week": int(r["week"]),
                    "team": away, "opponent": home, "game_id": game_id,
                    "home_flag": 0.0, "team_rest": arest, "opp_rest": hrest,
                    "market_total": total,
                },
            ])
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Stage-1 schedule loader produced zero rows")
    if out.duplicated(TG).any():
        raise RuntimeError("Stage-1 schedule contains duplicate team-games")
    _numeric(out, ["season", "week", "home_flag", "team_rest", "opp_rest", "market_total"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["rest_diff"] = out["team_rest"] - out["opp_rest"]
    if out[["team_rest", "opp_rest"]].isna().any().any():
        bad = out.loc[out[["team_rest", "opp_rest"]].isna().any(axis=1), TG].head(20)
        raise RuntimeError(f"missing pregame rest fields: {bad.to_dict('records')}")
    return out.sort_values(TG).reset_index(drop=True)


def _add_prior_roll(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_cols: Sequence[str],
    prefix_map: dict[str, str],
) -> pd.DataFrame:
    """Add strictly-prior rolling means and source-max ordinals."""
    x = frame.sort_values([group_col, "season", "week"]).copy()
    x["_ordinal"] = x["season"].astype(int) * 100 + x["week"].astype(int)
    g = x.groupby(group_col, sort=False)
    for source, dest in prefix_map.items():
        x[dest] = g[source].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_PRIOR).mean()
        )
    x[f"{group_col}_source_max_ordinal"] = g["_ordinal"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_PRIOR).max()
    )
    return x


def build_script_feature_frame(team_weekly: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Construct the six frozen football-only pregame script features."""
    tw = team_weekly.copy()
    tw.columns = [str(c).strip().lower() for c in tw.columns]
    required = {"season", "week", "team", "plays_est", "dropback_rate"}
    missing = required - set(tw.columns)
    if missing:
        raise RuntimeError(f"team weekly source missing: {sorted(missing)}")
    _numeric(tw, ["season", "week", "plays_est", "dropback_rate"])
    tw["season"] = tw["season"].astype(int)
    tw["week"] = tw["week"].astype(int)
    tw["team"] = tw["team"].map(canon_team)
    if tw.duplicated(TG).any():
        raise RuntimeError("team weekly source contains duplicate team-games")

    sched_key = schedule[TG + ["opponent", "game_id", "home_flag", "rest_diff", "market_total"]].copy()
    x = tw.merge(sched_key, on=TG, how="inner", validate="one_to_one")
    if x.empty:
        raise RuntimeError("no overlap between PBP team-week and schedule")
    x["realized_pass_volume"] = x["plays_est"] * x["dropback_rate"]

    own = _add_prior_roll(
        x,
        group_col="team",
        value_cols=["plays_est", "dropback_rate"],
        prefix_map={
            "plays_est": "team_plays_prior8",
            "dropback_rate": "team_dropback_rate_prior8",
        },
    )

    defense_obs = x[["season", "week", "team", "opponent", "plays_est", "dropback_rate"]].copy()
    defense_obs = defense_obs.rename(columns={"opponent": "defense_team"})
    defense_obs = _add_prior_roll(
        defense_obs,
        group_col="defense_team",
        value_cols=["plays_est", "dropback_rate"],
        prefix_map={
            "plays_est": "opp_plays_allowed_prior8",
            "dropback_rate": "opp_dropback_rate_allowed_prior8",
        },
    )
    defense_feat = defense_obs[
        ["season", "week", "defense_team", "opp_plays_allowed_prior8",
         "opp_dropback_rate_allowed_prior8", "defense_team_source_max_ordinal"]
    ].rename(columns={"defense_team": "opponent"})
    own = own.merge(defense_feat, on=["season", "week", "opponent"], how="left", validate="one_to_one")
    own["target_ordinal"] = own["season"] * 100 + own["week"]
    avail_own = own["team_source_max_ordinal"].notna()
    avail_def = own["defense_team_source_max_ordinal"].notna()
    if (own.loc[avail_own, "team_source_max_ordinal"] >= own.loc[avail_own, "target_ordinal"]).any():
        raise RuntimeError("offensive rolling feature used target/future row")
    if (own.loc[avail_def, "defense_team_source_max_ordinal"] >= own.loc[avail_def, "target_ordinal"]).any():
        raise RuntimeError("defensive rolling feature used target/future row")
    own["naive_pass_volume"] = own["team_plays_prior8"] * own["team_dropback_rate_prior8"]
    return own.sort_values(TG).reset_index(drop=True)


def _script_pipeline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def crossfit_script_predictions(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit exactly the two frozen forward script models and emit 2023/2024 predictions."""
    x = feature_frame.copy()
    eligible = x[SCRIPT_FEATURES + ["realized_pass_volume"]].notna().all(axis=1)
    x = x.loc[eligible].copy()
    parts = []
    audit = {"models": []}
    for pred_season, train_seasons in [(2023, [2022]), (2024, [2022, 2023])]:
        train = x.loc[x["season"].isin(train_seasons)].copy()
        test = x.loc[x["season"].eq(pred_season)].copy()
        if len(train) < 100 or len(test) < 100:
            raise RuntimeError(f"insufficient script rows pred={pred_season}: train={len(train)} test={len(test)}")
        if int(train["season"].max()) >= pred_season:
            raise RuntimeError("script fit season is not strictly before prediction season")
        model = _script_pipeline()
        model.fit(train[SCRIPT_FEATURES], train["realized_pass_volume"])
        out = test[TG + ["game_id", "opponent", "market_total", "home_flag", "rest_diff", "naive_pass_volume"]].copy()
        out["pred_realized_pass_volume"] = model.predict(test[SCRIPT_FEATURES])
        out["prediction_season"] = pred_season
        out["script_fit_max_season"] = int(max(train_seasons))
        parts.append(out)
        ridge = model.named_steps["ridge"]
        audit["models"].append({
            "prediction_season": pred_season,
            "train_seasons": list(train_seasons),
            "train_rows": int(len(train)),
            "prediction_rows": int(len(test)),
            "alpha": RIDGE_ALPHA,
            "features": list(SCRIPT_FEATURES),
            "coefficients_scaled": [float(v) for v in ridge.coef_],
            "intercept_scaled": float(ridge.intercept_),
        })
    pred = pd.concat(parts, ignore_index=True)
    if pred.duplicated(TG).any():
        raise RuntimeError("duplicate script predictions")
    if (pred["script_fit_max_season"] >= pred["season"]).any():
        raise RuntimeError("script prediction temporal boundary violated")
    return pred.sort_values(TG).reset_index(drop=True), audit


def load_phase4b(phase4b_dir: Path, *, include_outcomes: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    l23_path = phase4b_dir / "phase4b_layer2_3_team_game_detail.csv"
    l1_path = phase4b_dir / "phase4b_layer1_yard_decomposition.csv"
    if not l23_path.exists() or not l1_path.exists():
        raise RuntimeError(f"Phase4B artifact missing required files under {phase4b_dir}")
    structural = ["season", "week", "team", "implied_team_target_pool", "candidate_wr_room_mass"]
    outcome = ["actual_team_targets", "actual_wr_room_targets"]
    l23 = pd.read_csv(l23_path, usecols=structural + (outcome if include_outcomes else []))
    _numeric(l23, ["season", "week", "implied_team_target_pool", "candidate_wr_room_mass"] + (outcome if include_outcomes else []))
    l23["season"] = l23["season"].astype(int)
    l23["week"] = l23["week"].astype(int)
    l23["team"] = l23["team"].map(canon_team)
    if l23.duplicated(TG).any():
        raise RuntimeError("Phase4B Layer2/3 duplicate team-game")
    counts = l23.groupby("season").size().to_dict()
    if counts != EXPECTED_LAYER23:
        raise RuntimeError(f"Phase4B Layer2/3 count drift: {counts}")
    l1_cols = ["season", "week", "team", "actual_rec_yards", "yard_residual", "opportunity_yards", "efficiency_yards"]
    l1 = pd.read_csv(l1_path, usecols=l1_cols)
    _numeric(l1, ["season", "week", "actual_rec_yards", "yard_residual", "opportunity_yards", "efficiency_yards"])
    l1["season"] = l1["season"].astype(int)
    l1["week"] = l1["week"].astype(int)
    l1["team"] = l1["team"].map(canon_team)
    return l23, l1


def tail_team_games(layer1: pd.DataFrame, season: int = 2024) -> dict[str, pd.DataFrame]:
    x = layer1.loc[layer1["season"].eq(int(season))].copy()
    opp_dom = x["opportunity_yards"].abs() > x["efficiency_yards"].abs()
    masks = {
        "UNDERPROJECT_30_PLUS_OPP_DOM": x["yard_residual"].ge(30.0) & opp_dom,
        "ACTUAL_100_PLUS_OPP_DOM": x["actual_rec_yards"].ge(100.0) & opp_dom,
        "UNDERPROJECT_30_PLUS": x["yard_residual"].ge(30.0),
    }
    out = {}
    for name, mask in masks.items():
        tg = x.loc[mask, TG].drop_duplicates().sort_values(TG).reset_index(drop=True)
        expected = RAW_LAYER1_TAILS_2024[name]
        if len(tg) != expected:
            raise RuntimeError(f"tail count drift {name}: {len(tg)} != {expected}")
        out[name] = tg
    return out


@dataclass
class DownstreamFit:
    train_means: dict[str, float]
    intercept_a: float
    model_b: LinearRegression
    model_c: LinearRegression
    train_rows: int


def _downstream_train_frame(l23_outcomes: pd.DataFrame, script_pred: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    train = l23_outcomes.loc[l23_outcomes["season"].eq(2023)].copy()
    train = train.merge(script_pred.loc[script_pred["season"].eq(2023), TG + ["pred_realized_pass_volume"]], on=TG, how="inner", validate="one_to_one")
    train = train.merge(schedule[TG + ["market_total", "game_id"]], on=TG, how="inner", validate="one_to_one")
    train = train.loc[train[["market_total", "pred_realized_pass_volume", "actual_team_targets"]].notna().all(axis=1)].copy()
    train["low_total_lt38"] = train["market_total"].lt(38.0).astype(float)
    train["residual_target"] = train["actual_team_targets"] - train["implied_team_target_pool"]
    return train


def fit_downstream_models(train: pd.DataFrame) -> DownstreamFit:
    if len(train) < 450:
        raise RuntimeError(f"too few 2023 downstream fit rows: {len(train)}")
    means = {
        "market_total": float(train["market_total"].mean()),
        "pred_realized_pass_volume": float(train["pred_realized_pass_volume"].mean()),
    }
    y = train["residual_target"].to_numpy(dtype=float)
    intercept_a = float(y.mean())
    xb = np.column_stack([
        train["market_total"].to_numpy(dtype=float) - means["market_total"],
        train["low_total_lt38"].to_numpy(dtype=float),
    ])
    xc = np.column_stack([
        xb,
        train["pred_realized_pass_volume"].to_numpy(dtype=float) - means["pred_realized_pass_volume"],
    ])
    b = LinearRegression(fit_intercept=True).fit(xb, y)
    c = LinearRegression(fit_intercept=True).fit(xc, y)
    return DownstreamFit(means, intercept_a, b, c, int(len(train)))


def build_sealed_2024_predictions(l23_struct: pd.DataFrame, script_pred: pd.DataFrame, schedule: pd.DataFrame, fit: DownstreamFit) -> pd.DataFrame:
    test = l23_struct.loc[l23_struct["season"].eq(2024)].copy()
    test = test.merge(script_pred.loc[script_pred["season"].eq(2024), TG + ["pred_realized_pass_volume", "naive_pass_volume"]], on=TG, how="inner", validate="one_to_one")
    test = test.merge(schedule[TG + ["market_total", "game_id"]], on=TG, how="inner", validate="one_to_one")
    test = test.loc[test[["market_total", "pred_realized_pass_volume"]].notna().all(axis=1)].copy()
    test["low_total_lt38"] = test["market_total"].lt(38.0).astype(float)
    xb = np.column_stack([
        test["market_total"].to_numpy(dtype=float) - fit.train_means["market_total"],
        test["low_total_lt38"].to_numpy(dtype=float),
    ])
    xc = np.column_stack([
        xb,
        test["pred_realized_pass_volume"].to_numpy(dtype=float) - fit.train_means["pred_realized_pass_volume"],
    ])
    test["pool_A0"] = test["implied_team_target_pool"]
    test["pool_A"] = test["implied_team_target_pool"] + fit.intercept_a
    test["pool_B"] = test["implied_team_target_pool"] + fit.model_b.predict(xb)
    test["pool_C"] = test["implied_team_target_pool"] + fit.model_c.predict(xc)
    return test.sort_values(TG).reset_index(drop=True)


def paired_cluster_bootstrap_ci(frame: pd.DataFrame, value_col: str, *, cluster_col: str = "game_id", reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """Paired cluster bootstrap of a row-mean difference, resampling NFL games."""
    x = frame[[cluster_col, value_col]].dropna().copy()
    if x.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "clusters": 0, "rows": 0}
    x[value_col] = pd.to_numeric(x[value_col], errors="raise").astype(float)
    grouped = x.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"]).reset_index()
    observed = float(x[value_col].mean())
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(reps), dtype=float)
    for i in range(int(reps)):
        idx = rng.integers(0, len(grouped), size=len(grouped))
        boots[i] = float(sums[idx].sum() / counts[idx].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": observed, "ci_low": float(lo), "ci_high": float(hi),
        "clusters": int(len(grouped)), "rows": int(len(x)), "reps": int(reps),
        "seed": int(seed), "cluster_unit": "actual_nfl_game_id",
    }


def _mae(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.mean(np.abs(pred.to_numpy(dtype=float) - actual.to_numpy(dtype=float))))


def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    d = pred.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _bias(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.mean(pred.to_numpy(dtype=float) - actual.to_numpy(dtype=float)))


def _arm_metrics(test: pd.DataFrame, actual_col: str, prefix: str = "pool") -> dict:
    out = {}
    for arm in ["A0", "A", "B", "C"]:
        p = test[f"{prefix}_{arm}"]
        a = test[actual_col]
        out[arm] = {"mae": _mae(p, a), "rmse": _rmse(p, a), "bias_pred_minus_actual": _bias(p, a)}
    return out


def stage1_disposition(gates: dict[str, bool]) -> str:
    return "QUALIFIED_PREDICTED_PASS_VOLUME_INCREMENT" if gates and all(bool(v) for v in gates.values()) else "NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT"


def mechanics_preflight(phase4b_dir: Path, out_dir: Path) -> dict:
    """Run authorized mechanics while keeping blind-2024 outcomes sealed."""
    schedule = load_schedule_team_rows([2021, 2022, 2023, 2024])
    team_weekly = build_team_weekly_from_pbp([2021, 2022, 2023, 2024])
    feat = build_script_feature_frame(team_weekly, schedule)
    script_pred, script_audit = crossfit_script_predictions(feat)
    l23_struct, l1 = load_phase4b(phase4b_dir, include_outcomes=False)
    tails = tail_team_games(l1, 2024)
    l23_train_outcomes, _ = load_phase4b(phase4b_dir, include_outcomes=True)
    l23_train_outcomes = l23_train_outcomes.loc[l23_train_outcomes["season"].eq(2023)].copy()
    train = _downstream_train_frame(l23_train_outcomes, script_pred, schedule)
    fit = fit_downstream_models(train)
    sealed = build_sealed_2024_predictions(l23_struct, script_pred, schedule, fit)
    full_coverage = len(sealed) / EXPECTED_LAYER23[2024]
    primary = tails["UNDERPROJECT_30_PLUS_OPP_DOM"]
    primary_covered = primary.merge(sealed[TG], on=TG, how="inner").drop_duplicates(TG)
    primary_coverage = len(primary_covered) / EXPECTED_TAILS_2024["UNDERPROJECT_30_PLUS_OPP_DOM"]
    forbidden = {"actual_team_targets", "actual_wr_room_targets", "realized_pass_volume"}
    if forbidden & set(sealed.columns):
        raise RuntimeError(f"preflight blind-output seal violated: {sorted(forbidden & set(sealed.columns))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed.to_csv(out_dir / "stage1_sealed_2024_predictions.csv", index=False)
    source_audit = {
        "plays_est_source": "scripts/backtest/historical_inputs.py::build_team_weekly_from_pbp -> nflverse PBP offensive play count",
        "dropback_rate_source": "scripts/backtest/historical_inputs.py::build_team_weekly_from_pbp -> mean(qb_dropback) over offensive plays",
        "pbp_loader": "scripts/utils/pbp.py::get_pbp -> nflreadpy.load_pbp / nfl_data_py fallback",
        "market_lineage_in_script_target_or_rolling_priors": 0,
        "market_fields_in_script_model": [],
        "downstream_market_fields": ["market_total", "low_total_lt38"],
    }
    result = {
        "specification": "WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_PREFLIGHT",
        "blind_2024_outcomes_scored": False,
        "production_change": False,
        "source_lineage": source_audit,
        "temporal_integrity": {
            "window": WINDOW, "min_prior_games": MIN_PRIOR,
            "script_2023_fit_seasons": [2022], "script_2024_fit_seasons": [2022, 2023],
            "target_or_future_rows_used_in_rolling_features": 0,
        },
        "script_model_audit": script_audit,
        "downstream_fit": {
            "season": 2023, "rows": fit.train_rows, "train_means": fit.train_means,
            "A_intercept": fit.intercept_a,
            "B_intercept": float(fit.model_b.intercept_), "B_coefficients": [float(v) for v in fit.model_b.coef_],
            "C_intercept": float(fit.model_c.intercept_), "C_coefficients": [float(v) for v in fit.model_c.coef_],
            "same_fit_cohort_A_B_C": True,
        },
        "sealed_2024": {
            "canonical_team_games": EXPECTED_LAYER23[2024], "prediction_rows": int(len(sealed)),
            "coverage": float(full_coverage),
            "primary_tail_raw_layer1_team_games": RAW_LAYER1_TAILS_2024["UNDERPROJECT_30_PLUS_OPP_DOM"],
            "primary_tail_team_games": EXPECTED_TAILS_2024["UNDERPROJECT_30_PLUS_OPP_DOM"],
            "primary_tail_prediction_rows": int(len(primary_covered)), "primary_tail_coverage": float(primary_coverage),
            "contains_actual_team_targets": False, "contains_actual_wr_room_targets": False,
            "contains_realized_pass_volume": False,
        },
        "materiality_context": {
            "team_pool_mae_floor_targets": MATERIALITY_TARGETS,
            "phase4b_layer2_reported_baseline_mae_reference": 4.520381,
            "floor_as_pct_of_reference": float(MATERIALITY_TARGETS / 4.520381 * 100.0),
        },
    }
    with (out_dir / "stage1_preflight.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def run_outcomes(phase4b_dir: Path, out_dir: Path) -> dict:
    """Expose the frozen blind-2024 Stage-1 evaluation after review authorization."""
    schedule = load_schedule_team_rows([2021, 2022, 2023, 2024])
    team_weekly = build_team_weekly_from_pbp([2021, 2022, 2023, 2024])
    feat = build_script_feature_frame(team_weekly, schedule)
    script_pred, script_audit = crossfit_script_predictions(feat)
    l23, l1 = load_phase4b(phase4b_dir, include_outcomes=True)
    tails = tail_team_games(l1, 2024)
    train = _downstream_train_frame(l23, script_pred, schedule)
    fit = fit_downstream_models(train)
    struct = l23[TG + ["implied_team_target_pool", "candidate_wr_room_mass"]].copy()
    pred = build_sealed_2024_predictions(struct, script_pred, schedule, fit)
    actual = l23.loc[l23["season"].eq(2024), TG + ["actual_team_targets", "actual_wr_room_targets"]]
    test = pred.merge(actual, on=TG, how="left", validate="one_to_one")
    realized = feat.loc[feat["season"].eq(2024), TG + ["realized_pass_volume"]]
    test = test.merge(realized, on=TG, how="left", validate="one_to_one")
    if test[["actual_team_targets", "actual_wr_room_targets"]].isna().any().any():
        raise RuntimeError("missing Phase4B outcomes on eligible blind test rows")
    full_coverage = len(test) / EXPECTED_LAYER23[2024]
    primary_tg = tails["UNDERPROJECT_30_PLUS_OPP_DOM"]
    primary_covered = primary_tg.merge(test[TG], on=TG, how="inner").drop_duplicates(TG)
    primary_coverage = len(primary_covered) / EXPECTED_TAILS_2024["UNDERPROJECT_30_PLUS_OPP_DOM"]

    script_eval = test.dropna(subset=["realized_pass_volume", "naive_pass_volume", "pred_realized_pass_volume"]).copy()
    script_eval["abs_naive"] = (script_eval["naive_pass_volume"] - script_eval["realized_pass_volume"]).abs()
    script_eval["abs_script"] = (script_eval["pred_realized_pass_volume"] - script_eval["realized_pass_volume"]).abs()
    script_eval["diff_naive_minus_script"] = script_eval["abs_naive"] - script_eval["abs_script"]
    script_ci = paired_cluster_bootstrap_ci(script_eval, "diff_naive_minus_script")
    script_metrics = {
        "rows": int(len(script_eval)), "naive_mae": float(script_eval["abs_naive"].mean()),
        "script_mae": float(script_eval["abs_script"].mean()),
        "mae_improvement_naive_minus_script": float(script_eval["diff_naive_minus_script"].mean()),
        "bootstrap": script_ci,
    }

    pool_metrics = _arm_metrics(test, "actual_team_targets", prefix="pool")
    for left, right, name in [("B", "C", "B_minus_C"), ("A", "C", "A_minus_C"), ("A", "B", "A_minus_B")]:
        test[f"abs_err_{left}"] = (test[f"pool_{left}"] - test["actual_team_targets"]).abs()
        test[f"abs_err_{right}"] = (test[f"pool_{right}"] - test["actual_team_targets"]).abs()
        test[f"diff_{name}"] = test[f"abs_err_{left}"] - test[f"abs_err_{right}"]
    pool_cis = {
        "B_minus_C": paired_cluster_bootstrap_ci(test, "diff_B_minus_C"),
        "A_minus_C": paired_cluster_bootstrap_ci(test, "diff_A_minus_C"),
        "A_minus_B": paired_cluster_bootstrap_ci(test, "diff_A_minus_B"),
    }

    for arm in ["A0", "A", "B", "C"]:
        test[f"wrroom_{arm}"] = test[f"pool_{arm}"] * test["candidate_wr_room_mass"]
    wr_metrics = _arm_metrics(test, "actual_wr_room_targets", prefix="wrroom")
    test["wrroom_abs_B"] = (test["wrroom_B"] - test["actual_wr_room_targets"]).abs()
    test["wrroom_abs_C"] = (test["wrroom_C"] - test["actual_wr_room_targets"]).abs()
    test["wrroom_diff_B_minus_C"] = test["wrroom_abs_B"] - test["wrroom_abs_C"]
    wr_ci = paired_cluster_bootstrap_ci(test, "wrroom_diff_B_minus_C")

    tail_metrics = {}
    for name, tg in tails.items():
        z = test.merge(tg.assign(_tail=1), on=TG, how="inner")
        z["diff_B_minus_C"] = (z["pool_B"] - z["actual_team_targets"]).abs() - (z["pool_C"] - z["actual_team_targets"]).abs()
        tail_metrics[name] = {
            "raw_layer1_team_games": RAW_LAYER1_TAILS_2024[name], "canonical_team_games": EXPECTED_TAILS_2024[name], "eligible_team_games": int(len(z)),
            "coverage": float(len(z) / EXPECTED_TAILS_2024[name]),
            "mae_improvement_B_minus_C": float(z["diff_B_minus_C"].mean()) if len(z) else np.nan,
            "bootstrap": paired_cluster_bootstrap_ci(z, "diff_B_minus_C") if len(z) else {},
        }

    gates = {
        "coverage_integrity": bool(full_coverage >= 0.95 and primary_coverage >= 0.95),
        "script_blind_skill": bool(script_metrics["script_mae"] < script_metrics["naive_mae"] and script_ci["ci_low"] > 0.0),
        "C_gt_B_material": bool(pool_metrics["B"]["mae"] - pool_metrics["C"]["mae"] >= MATERIALITY_TARGETS and pool_cis["B_minus_C"]["ci_low"] > 0.0),
        "C_gt_A_material": bool(pool_metrics["A"]["mae"] - pool_metrics["C"]["mae"] >= MATERIALITY_TARGETS and pool_cis["A_minus_C"]["ci_low"] > 0.0),
        "no_rmse_bias_tradeoff": bool(pool_metrics["C"]["rmse"] <= pool_metrics["B"]["rmse"] and abs(pool_metrics["C"]["bias_pred_minus_actual"]) <= abs(pool_metrics["B"]["bias_pred_minus_actual"]) + BIAS_TOLERANCE),
        "wr_room_translation_positive": bool(wr_metrics["B"]["mae"] - wr_metrics["C"]["mae"] > 0.0 and wr_ci["ci_low"] > 0.0),
        "tail_mechanism_alignment": bool(
            tail_metrics["UNDERPROJECT_30_PLUS_OPP_DOM"]["mae_improvement_B_minus_C"] > 0.0 and
            (tail_metrics["ACTUAL_100_PLUS_OPP_DOM"]["mae_improvement_B_minus_C"] > 0.0 or tail_metrics["UNDERPROJECT_30_PLUS"]["mae_improvement_B_minus_C"] > 0.0)
        ),
    }
    disposition = stage1_disposition(gates)

    out_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_dir / "stage1_2024_team_game_detail.csv", index=False)
    pd.DataFrame([{"comparison": k, **v} for k, v in pool_cis.items()]).to_csv(out_dir / "stage1_pool_bootstrap.csv", index=False)
    pd.DataFrame([
        {"tail": name, "canonical_team_games": values["canonical_team_games"], "eligible_team_games": values["eligible_team_games"],
         "coverage": values["coverage"], "mae_improvement_B_minus_C": values["mae_improvement_B_minus_C"],
         "ci_low": values.get("bootstrap", {}).get("ci_low"), "ci_high": values.get("bootstrap", {}).get("ci_high")}
        for name, values in tail_metrics.items()
    ]).to_csv(out_dir / "stage1_tail_metrics.csv", index=False)

    result = {
        "specification": "WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1",
        "blind_2024_outcomes_scored": True, "production_change": False,
        "source_lineage": {
            "plays_est": "nflverse PBP offensive play count",
            "dropback_rate": "nflverse PBP mean(qb_dropback) over offensive plays",
            "market_lineage_in_script_target_or_priors": 0, "script_model_market_inputs": [],
        },
        "coverage": {
            "canonical_2024_team_games": EXPECTED_LAYER23[2024], "eligible_2024_team_games": int(len(test)),
            "full_coverage": float(full_coverage),
            "primary_tail_raw_layer1": RAW_LAYER1_TAILS_2024["UNDERPROJECT_30_PLUS_OPP_DOM"],
            "primary_tail_canonical": EXPECTED_TAILS_2024["UNDERPROJECT_30_PLUS_OPP_DOM"],
            "primary_tail_eligible": int(len(primary_covered)), "primary_tail_coverage": float(primary_coverage),
        },
        "script_model_audit": script_audit, "script_metrics": script_metrics,
        "team_pool_metrics": pool_metrics, "team_pool_bootstrap": pool_cis,
        "wr_room_metrics": wr_metrics, "wr_room_B_minus_C_bootstrap": wr_ci,
        "tail_metrics": tail_metrics, "gates": gates,
        "materiality_context": {
            "team_target_floor": MATERIALITY_TARGETS,
            "phase4b_layer2_reported_baseline_mae_reference": 4.520381,
            "floor_as_pct_of_reference": float(MATERIALITY_TARGETS / 4.520381 * 100.0),
        },
        "disposition": disposition, "challenger_or_production_authorized": False,
    }
    with (out_dir / "stage1_result.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase4b-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run-outcomes", action="store_true")
    args = p.parse_args()
    result = mechanics_preflight(args.phase4b_dir, args.out_dir) if args.preflight_only else run_outcomes(args.phase4b_dir, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
