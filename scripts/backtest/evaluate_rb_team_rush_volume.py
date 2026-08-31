"""Migration 94: football-only team rushing opportunity / game-script proxy.

M94 isolates the team-volume half of the RB opportunity problem found in M92.
It consumes the frozen M91 temporal artifact and never uses sportsbook fields.
Pregame features are built strictly from team-week observations before the
prediction cutoff plus the already-existing M91 team rush projection.

2024 weeks 1-12 are development training, 2024 weeks 13-18 are the model-family
holdout, and the selected family is then refit on all 2024 and evaluated on 2025.
No 2025 outcome is used to select the model family or coefficients.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM_KEYS = ["season", "week", "team"]
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
ROLL_METRICS = [
    "plays_est", "dropback_rate", "proe", "neutral_pace",
    "success_rate_off", "success_rate_def", "pressure_rate_allowed",
    "pressure_rate_generated", "def_rush_epa", "def_pass_epa",
    "explosive_play_rate_allowed", "pass_attempts_per_dropback",
    "avg_defenders_in_box", "light_box_rate", "heavy_box_rate",
]
ROLL_WINDOWS = [1, 3, 5]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _read_season(root: Path, season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / str(season)
    pred_path = base / "component_predictions.csv"
    team_path = base / "team_weekly_history.csv"
    sched_path = base / "schedule_history.csv"
    for path in (pred_path, team_path, sched_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing frozen M91 file: {path}")
    pred = _lower(pd.read_csv(pred_path, low_memory=False))
    team = _lower(pd.read_csv(team_path, low_memory=False))
    sched = _lower(pd.read_csv(sched_path, low_memory=False))
    return pred, team, sched


def _target_team_table(pred: pd.DataFrame, sched: pd.DataFrame, season: int) -> pd.DataFrame:
    rush = pred.loc[pred["market"].astype(str).str.lower().eq("rush_att")].copy()
    rush["actual"] = pd.to_numeric(rush["actual"], errors="coerce")
    rush["ml_proj"] = pd.to_numeric(rush["ml_proj"], errors="coerce")
    out = rush.groupby(TEAM_KEYS, dropna=False).agg(
        actual_team_rush_att=("actual", lambda s: s.sum(min_count=1)),
        baseline_team_rush_att=("ml_proj", lambda s: s.sum(min_count=1)),
    ).reset_index()
    s = sched.loc[pd.to_numeric(sched["season"], errors="coerce").eq(int(season)),
                  ["season", "week", "team", "opponent", "home_away"]].copy()
    s["home"] = s["home_away"].astype(str).str.lower().eq("home").astype(int)
    s = s.drop(columns=["home_away"])
    out = out.merge(s, on=TEAM_KEYS, how="left", validate="one_to_one")
    if out["opponent"].isna().any():
        raise RuntimeError(f"M94 schedule join left unresolved opponents for {season}")
    return out


def _prior_team_rows(hist: pd.DataFrame, team: str, season: int, week: int) -> pd.DataFrame:
    hs = pd.to_numeric(hist["season"], errors="coerce")
    hw = pd.to_numeric(hist["week"], errors="coerce")
    mask = hist["team"].astype(str).eq(str(team)) & (hs.lt(int(season)) | (hs.eq(int(season)) & hw.lt(int(week))))
    return hist.loc[mask].sort_values(["season", "week"])


def _roll_features(hist: pd.DataFrame, team: str, season: int, week: int, prefix: str) -> dict[str, float]:
    g = _prior_team_rows(hist, team, season, week)
    rec: dict[str, float] = {f"{prefix}history_games": float(len(g))}
    for metric in ROLL_METRICS:
        if metric not in hist.columns:
            continue
        vals = pd.to_numeric(g[metric], errors="coerce")
        for n in ROLL_WINDOWS:
            z = vals.tail(n).dropna()
            rec[f"{prefix}{metric}_avg{n}"] = float(z.mean()) if len(z) else np.nan
    return rec


def build_features(root: Path, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred, hist, sched = _read_season(root, season)
    for c in ["season", "week"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    targets = _target_team_table(pred, sched, season)
    rows: list[dict] = []
    for _, r in targets.iterrows():
        rec = r.to_dict()
        rec.update(_roll_features(hist, str(r["team"]), int(r["season"]), int(r["week"]), "team_"))
        rec.update(_roll_features(hist, str(r["opponent"]), int(r["season"]), int(r["week"]), "opp_"))
        rows.append(rec)
    x = pd.DataFrame(rows)
    # Derived football-only playcalling / possession environment signals.
    for n in ROLL_WINDOWS:
        td = f"team_dropback_rate_avg{n}"; od = f"opp_dropback_rate_avg{n}"
        if td in x.columns:
            x[f"team_rush_rate_avg{n}"] = 1.0 - pd.to_numeric(x[td], errors="coerce")
        if od in x.columns:
            x[f"opp_rush_rate_avg{n}"] = 1.0 - pd.to_numeric(x[od], errors="coerce")
        ts = f"team_success_rate_off_avg{n}"; ods = f"opp_success_rate_def_avg{n}"
        if ts in x.columns and ods in x.columns:
            x[f"offense_vs_def_success_edge_avg{n}"] = pd.to_numeric(x[ts], errors="coerce") - pd.to_numeric(x[ods], errors="coerce")
        tp = f"team_neutral_pace_avg{n}"; op = f"opp_neutral_pace_avg{n}"
        if tp in x.columns and op in x.columns:
            x[f"combined_neutral_seconds_per_play_avg{n}"] = (pd.to_numeric(x[tp], errors="coerce") + pd.to_numeric(x[op], errors="coerce")) / 2.0
        tplay = f"team_plays_est_avg{n}"; oplay = f"opp_plays_est_avg{n}"
        if tplay in x.columns and oplay in x.columns:
            x[f"combined_recent_plays_avg{n}"] = (pd.to_numeric(x[tplay], errors="coerce") + pd.to_numeric(x[oplay], errors="coerce")) / 2.0
    return x, pred


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    blocked = {"season", "week", "team", "opponent", "actual_team_rush_att"}
    cols = []
    for c in frame.columns:
        if c in blocked:
            continue
        v = pd.to_numeric(frame[c], errors="coerce")
        if v.notna().any():
            cols.append(c)
    return sorted(cols)


def _models() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=12.0)),
        ]),
        "gbr": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                random_state=94, n_estimators=160, learning_rate=0.03,
                max_depth=2, min_samples_leaf=10, loss="huber",
            )),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=94, n_estimators=350, max_depth=5,
                min_samples_leaf=8, max_features=0.7, n_jobs=-1,
            )),
        ]),
    }


def _metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    z = pd.DataFrame({"a": pd.to_numeric(actual, errors="coerce"), "p": pd.to_numeric(pred, errors="coerce")}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    err = z["p"] - z["a"]
    corr = float(z["a"].corr(z["p"])) if len(z) > 1 and z["a"].nunique() > 1 and z["p"].nunique() > 1 else np.nan
    return {
        "n": int(len(z)), "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()), "correlation": corr,
    }


def _team_summaries(x: pd.DataFrame, candidate_col: str, label: str) -> pd.DataFrame:
    masks = {
        "all_team_games": pd.Series(True, index=x.index),
        "actual_20_or_less": x["actual_team_rush_att"].le(20),
        "actual_21_29": x["actual_team_rush_att"].between(21, 29),
        "actual_30_plus": x["actual_team_rush_att"].ge(30),
        "actual_35_plus": x["actual_team_rush_att"].ge(35),
    }
    rows = []
    for name, mask in masks.items():
        g = x.loc[mask]
        b = _metrics(g["actual_team_rush_att"], g["baseline_team_rush_att"])
        c = _metrics(g["actual_team_rush_att"], g[candidate_col])
        rows.append({
            "candidate": label, "slice": name, "n": b["n"],
            "baseline_mae": b["mae"], "candidate_mae": c["mae"],
            "mae_gain": b["mae"] - c["mae"],
            "baseline_rmse": b["rmse"], "candidate_rmse": c["rmse"],
            "baseline_bias": b["bias"], "candidate_bias": c["bias"],
            "baseline_correlation": b["correlation"], "candidate_correlation": c["correlation"],
        })
    return pd.DataFrame(rows)


def _script_auc(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, fn in {
        "high_rush_30_plus": lambda y: y.ge(30).astype(int),
        "low_rush_20_or_less": lambda y: y.le(20).astype(int),
    }.items():
        ytr = fn(pd.to_numeric(train["actual_team_rush_att"], errors="coerce"))
        yte = fn(pd.to_numeric(test["actual_team_rush_att"], errors="coerce"))
        if ytr.nunique() < 2 or yte.nunique() < 2:
            out[f"{label}_auc"] = np.nan
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.35, max_iter=2000)),
        ])
        pipe.fit(train[features], ytr)
        prob = pipe.predict_proba(test[features])[:, 1]
        out[f"{label}_auc"] = float(roc_auc_score(yte, prob))
    return out


def _player_candidate(pred: pd.DataFrame, team_pred: pd.DataFrame, candidate_team_col: str) -> pd.DataFrame:
    rush = pred.loc[pred["market"].astype(str).str.lower().eq("rush_att")].copy()
    rush["actual"] = pd.to_numeric(rush["actual"], errors="coerce")
    rush["ml_proj"] = pd.to_numeric(rush["ml_proj"], errors="coerce")
    team_base = rush.groupby(TEAM_KEYS)["ml_proj"].sum(min_count=1).rename("base_team").reset_index()
    team_actual = rush.groupby(TEAM_KEYS)["actual"].sum(min_count=1).rename("actual_team").reset_index()
    r = rush.merge(team_base, on=TEAM_KEYS, how="left", validate="many_to_one")
    r = r.merge(team_actual, on=TEAM_KEYS, how="left", validate="many_to_one")
    r = r.merge(team_pred[TEAM_KEYS + [candidate_team_col]], on=TEAM_KEYS, how="left", validate="many_to_one")
    r["base_team_share"] = np.where(r["base_team"].gt(0), r["ml_proj"] / r["base_team"], 0.0)
    r["candidate_rush_att"] = r["base_team_share"] * r[candidate_team_col]
    rb = r.loc[r["position"].astype(str).str.upper().eq("RB")].copy()
    rb = rb.rename(columns={"actual": "actual_rush_att", "ml_proj": "base_rush_att"})
    rb["actual_team_share"] = np.where(rb["actual_team"].gt(0), rb["actual_rush_att"] / rb["actual_team"], np.nan)
    rb["bellcow_60"] = rb["actual_rush_att"].ge(15) & rb["actual_team_share"].ge(0.60)

    def market_frame(market: str, actual_name: str, base_name: str) -> pd.DataFrame:
        z = pred.loc[
            pred["market"].astype(str).str.lower().eq(market)
            & pred["position"].astype(str).str.upper().eq("RB"),
            PLAYER_KEYS + ["actual", "ml_proj"],
        ].copy()
        return z.rename(columns={"actual": actual_name, "ml_proj": base_name})

    rb = rb.merge(market_frame("rush_yards", "actual_rush_yards", "base_rush_yards"), on=PLAYER_KEYS, how="left", validate="one_to_one")
    rb = rb.merge(market_frame("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards"), on=PLAYER_KEYS, how="left", validate="one_to_one")
    ypc = np.where(rb["base_rush_att"].gt(0.5), rb["base_rush_yards"] / rb["base_rush_att"], np.nan)
    ypc = pd.Series(ypc, index=rb.index).clip(lower=0.0, upper=12.0)
    rb["candidate_rush_yards"] = np.where(ypc.notna(), rb["candidate_rush_att"] * ypc, rb["base_rush_yards"])
    rb["candidate_rush_rec_yards"] = rb["base_rush_rec_yards"] + rb["candidate_rush_yards"] - rb["base_rush_yards"]
    return rb


def _rb_summary(rb: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(rb["actual_rush_att"], errors="coerce")
    masks = {
        "all_rb": pd.Series(True, index=rb.index),
        "actual_0_5": a.le(5), "actual_6_10": a.between(6, 10),
        "actual_11_14": a.between(11, 14), "actual_15_plus": a.ge(15),
        "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25),
        "bellcow_60": rb["bellcow_60"].fillna(False),
    }
    rows = []
    for slice_name, mask in masks.items():
        g = rb.loc[mask]
        for market, actual, base, cand in [
            ("rush_att", "actual_rush_att", "base_rush_att", "candidate_rush_att"),
            ("rush_yards", "actual_rush_yards", "base_rush_yards", "candidate_rush_yards"),
            ("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards", "candidate_rush_rec_yards"),
        ]:
            b = _metrics(g[actual], g[base]); c = _metrics(g[actual], g[cand])
            rows.append({
                "slice": slice_name, "market": market, "n": b["n"],
                "baseline_mae": b["mae"], "candidate_mae": c["mae"],
                "mae_gain": b["mae"] - c["mae"],
                "baseline_bias": b["bias"], "candidate_bias": c["bias"],
                "baseline_correlation": b["correlation"], "candidate_correlation": c["correlation"],
            })
    return pd.DataFrame(rows)


def _legacy_guard(pred: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    all_ry = pred.loc[pred["market"].astype(str).str.lower().eq("rush_yards"), PLAYER_KEYS + ["position", "actual", "ml_proj"]].copy()
    all_ry = all_ry.merge(rb[PLAYER_KEYS + ["candidate_rush_yards"]], on=PLAYER_KEYS, how="left", validate="many_to_one")
    all_ry["candidate"] = np.where(
        all_ry["position"].astype(str).str.upper().eq("RB") & all_ry["candidate_rush_yards"].notna(),
        all_ry["candidate_rush_yards"], all_ry["ml_proj"],
    )
    b = _metrics(all_ry["actual"], all_ry["ml_proj"]); c = _metrics(all_ry["actual"], all_ry["candidate"])
    return pd.DataFrame([{
        "n": b["n"], "baseline_all_player_rush_yards_mae": b["mae"],
        "candidate_all_player_rush_yards_mae": c["mae"], "mae_gain": b["mae"] - c["mae"],
        "baseline_correlation": b["correlation"], "candidate_correlation": c["correlation"],
    }])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m94"))
    args = p.parse_args()
    x24, pred24 = build_features(args.m91_root, 2024)
    x25, pred25 = build_features(args.m91_root, 2025)
    features = sorted(set(_feature_cols(x24)) & set(_feature_cols(x25)))
    if "baseline_team_rush_att" not in features:
        raise RuntimeError("M94 feature set lost baseline_team_rush_att")

    train24 = x24.loc[pd.to_numeric(x24["week"], errors="coerce").le(12)].copy()
    hold24 = x24.loc[pd.to_numeric(x24["week"], errors="coerce").ge(13)].copy()
    ytr = pd.to_numeric(train24["actual_team_rush_att"], errors="coerce")
    grid_rows = []
    fitted_dev: dict[str, Pipeline] = {}
    for name, model in _models().items():
        model.fit(train24[features], ytr)
        fitted_dev[name] = model
        pred = np.clip(model.predict(hold24[features]), 8.0, 50.0)
        m = _metrics(hold24["actual_team_rush_att"], pd.Series(pred, index=hold24.index))
        hi = hold24["actual_team_rush_att"].ge(30)
        hm = _metrics(hold24.loc[hi, "actual_team_rush_att"], pd.Series(pred, index=hold24.index).loc[hi])
        grid_rows.append({"model": name, "holdout_mae": m["mae"], "holdout_bias": m["bias"], "holdout_corr": m["correlation"], "holdout_30plus_mae": hm["mae"]})
    grid = pd.DataFrame(grid_rows).sort_values(["holdout_mae", "holdout_30plus_mae", "model"]).reset_index(drop=True)
    selected = str(grid.iloc[0]["model"])

    # Freeze model family on 2024, then refit coefficients on all 2024 before 2025.
    final_model = _models()[selected]
    final_model.fit(x24[features], pd.to_numeric(x24["actual_team_rush_att"], errors="coerce"))
    x25["candidate_team_rush_att"] = np.clip(final_model.predict(x25[features]), 8.0, 50.0)
    hold24_model = fitted_dev[selected]
    hold24 = hold24.copy()
    hold24["candidate_team_rush_att"] = np.clip(hold24_model.predict(hold24[features]), 8.0, 50.0)

    team24 = _team_summaries(hold24, "candidate_team_rush_att", selected)
    team24.insert(0, "season_scope", "2024_w13_18_holdout")
    team25 = _team_summaries(x25, "candidate_team_rush_att", selected)
    team25.insert(0, "season_scope", "2025_validation")
    team_summary = pd.concat([team24, team25], ignore_index=True)

    script_diag = {
        "2024_holdout": _script_auc(train24, hold24, features),
        "2025_validation": _script_auc(x24, x25, features),
    }
    script_rows = [{"season_scope": k, **v} for k, v in script_diag.items()]
    script_df = pd.DataFrame(script_rows)

    rb25 = _player_candidate(pred25, x25, "candidate_team_rush_att")
    rb_summary = _rb_summary(rb25)
    guard = _legacy_guard(pred25, rb25)

    def gain_team(slice_name: str) -> float:
        q = team25.loc[team25["slice"].eq(slice_name), "mae_gain"]
        return float(q.iloc[0]) if len(q) else np.nan
    def gain_rb(market: str, slice_name: str = "all_rb") -> float:
        q = rb_summary.loc[rb_summary["market"].eq(market) & rb_summary["slice"].eq(slice_name), "mae_gain"]
        return float(q.iloc[0]) if len(q) else np.nan

    pass_gate = (
        gain_team("all_team_games") > 0
        and gain_team("actual_30_plus") > 0
        and gain_rb("rush_att") > 0
        and gain_rb("rush_yards") > 0
        and gain_rb("rush_att", "actual_20_plus") > 0
        and float(guard["mae_gain"].iloc[0]) >= 0
    )
    disposition = pd.DataFrame([{
        "selected_model_from_2024_holdout": selected,
        "feature_count": len(features),
        "development_train_weeks": "2024_01-12",
        "development_holdout_weeks": "2024_13-18",
        "validation_season": 2025,
        "validation_pass": int(pass_gate),
        "disposition": "ADVANCE_TEAM_VOLUME_SIGNAL" if pass_gate else "DO_NOT_ADVANCE_M94_TO_PRODUCTION",
        "note": "Research only; no sportsbook inputs and no production change.",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_dir / "m94_model_grid_2024_holdout.csv", index=False)
    team_summary.to_csv(args.out_dir / "m94_team_volume_summary.csv", index=False)
    script_df.to_csv(args.out_dir / "m94_script_classifier_diagnostics.csv", index=False)
    rb_summary.to_csv(args.out_dir / "m94_rb_validation_summary.csv", index=False)
    guard.to_csv(args.out_dir / "m94_legacy_rushing_guard.csv", index=False)
    disposition.to_csv(args.out_dir / "m94_disposition.csv", index=False)
    x25.to_csv(args.out_dir / "m94_2025_team_volume_trace.csv", index=False)
    rb25.to_csv(args.out_dir / "m94_2025_rb_trace.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.out_dir / "m94_features.csv", index=False)

    print("[rb_m94] 2024 model-family holdout grid")
    print(grid.to_string(index=False))
    print("\n[rb_m94] football-only high/low rush-script discrimination")
    print(script_df.to_string(index=False))
    print("\n[rb_m94] team rushing volume")
    print(team_summary.to_string(index=False))
    print("\n[rb_m94] 2025 RB translation")
    print(rb_summary.loc[rb_summary["slice"].isin(["all_rb", "actual_0_5", "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60"])].to_string(index=False))
    print("\n[rb_m94] legacy all-player rushing guard")
    print(guard.to_string(index=False))
    print("\n[rb_m94] disposition")
    print(disposition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
