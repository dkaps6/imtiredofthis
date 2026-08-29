#!/usr/bin/env python3
"""Migration 67: test new offensive-intent/personnel information after M66.

M66 found a .9715 median residual correlation across nine QB models, ~14.9 yards
of hindsight oracle headroom, and no deployable signal from 127 existing pregame
features.  M67 therefore does not tune another version of the same feature set.
It evaluates newly recovered information families with fixed 2024->2025 tests.

No M67 result directly promotes production logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import scripts.backtest.audit_qb_research_frontier as m66

INVALID_EXISTING = {"coach_change", "coach_tenure_games"}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def metrics(actual, pred, miss_threshold: float | None = None) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "miss": 0}
    e = z.p - z.a
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z.a.corr(z.p)) if len(z) >= 2 else np.nan,
        "miss": int(e.abs().ge(miss_threshold).sum()) if miss_threshold is not None else 0,
    }


def model_specs(target: str):
    return {
        f"ridge_{target}": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=50.0),
        ),
        f"histgb_{target}": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=150,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=15,
                l2_regularization=5.0,
                random_state=66,
            ),
        ),
    }


def usable_cols(x: pd.DataFrame, cols: list[str]) -> list[str]:
    train = x[num(x.season).eq(2024)]
    out = []
    for c in cols:
        if c not in x:
            continue
        x[c] = num(x[c])
        s = train[c]
        if s.notna().sum() >= 100 and s.nunique(dropna=True) > 1:
            out.append(c)
    return out


def prepare(m65_game: pd.DataFrame, state_features: pd.DataFrame, new_features: pd.DataFrame):
    base, existing = m66.merge_feature_universe(m65_game.copy(), state_features.copy())
    existing = [c for c in existing if c not in INVALID_EXISTING]
    nf = new_features.copy()
    nf["season"] = num(nf.season).astype(int)
    nf["week"] = num(nf.week).astype(int)
    nf["team"] = nf.team.astype(str)
    new_cols = [c for c in nf if c not in {"season", "week", "team", "history_games"}]
    x = base.merge(nf[["season", "week", "team"] + new_cols], on=["season", "week", "team"], how="left", validate="many_to_one")
    if len(x) != len(base):
        raise RuntimeError("M67 new-information join changed canonical row count")
    intent = usable_cols(x, [c for c in new_cols if c.startswith("intent_")])
    availability = usable_cols(x, [c for c in new_cols if c.startswith("availability_")])
    participation = usable_cols(x, [c for c in new_cols if c.startswith("personnel_") or c.startswith("continuity_")])
    existing = usable_cols(x, existing)
    families = {
        "pbp_intent_live": intent,
        "injury_availability_live": availability,
        "live_new_combined": list(dict.fromkeys(intent + availability)),
        "participation_personnel_historical_only": participation,
        "all_new_combined": list(dict.fromkeys(intent + availability + participation)),
        "existing_plus_live_new": list(dict.fromkeys(existing + intent + availability)),
        "existing_plus_all_new": list(dict.fromkeys(existing + intent + availability + participation)),
    }
    for name, cols in families.items():
        if not cols:
            raise RuntimeError(f"M67 family {name} has zero usable features")
    return x, families, existing


def family_new_coverage(x: pd.DataFrame, family: str, cols: list[str]) -> float:
    new = [c for c in cols if c.startswith(("intent_", "availability_", "personnel_", "continuity_"))]
    if not new:
        return 1.0
    vals = [float(num(x[c]).notna().mean()) for c in new]
    return float(np.median(vals)) if vals else 1.0


def residual_tests(x: pd.DataFrame, families: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = x[num(x.season).eq(2024)].copy()
    test = x[num(x.season).eq(2025)].copy()
    raw_col = "m64_pass_raw_reference"
    if raw_col not in x:
        raw_col = "mc_proj_attempts_raw_only"
    train["_target"] = num(train.actual) - num(train[raw_col])
    test["_target"] = num(test.actual) - num(test[raw_col])
    raw = metrics(test.actual, test[raw_col], 100)
    rows, preds = [], []
    for family, cols in families.items():
        coverage = family_new_coverage(test, family, cols)
        for model_name, model in model_specs("pass_residual").items():
            model.fit(train[cols], train._target)
            phat = model.predict(test[cols])
            corrected = num(test[raw_col]).to_numpy(float) + phat
            mm = metrics(test.actual, corrected, 100)
            rc = float(pd.Series(test._target.to_numpy(float)).corr(pd.Series(phat)))
            rows.append({
                "family": family, "model": model_name, "feature_count": len(cols), "new_feature_median_coverage": coverage,
                "residual_corr": rc, "corrected_mae": mm["mae"], "corrected_rmse": mm["rmse"], "corrected_corr": mm["corr"],
                "corrected_100plus": mm["miss"], "raw_mae": raw["mae"], "raw_rmse": raw["rmse"], "raw_corr": raw["corr"], "raw_100plus": raw["miss"],
                "mae_gain_vs_raw": raw["mae"] - mm["mae"], "corr_gain_vs_raw": mm["corr"] - raw["corr"],
            })
            q = test[["season", "week", "team", "player_clean_key", "actual"]].copy()
            q["family"] = family; q["model"] = model_name; q["predicted_residual"] = phat; q["corrected_projection"] = corrected
            preds.append(q)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def dbr_tests(x: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    train = x[num(x.season).eq(2024)].copy()
    test = x[num(x.season).eq(2025)].copy()
    target = "m64_actual_dropback_rate"
    base_col = "m64_pred_dropback_rate_neutral"
    train["_target"] = num(train[target]) - num(train[base_col])
    test["_target"] = num(test[target]) - num(test[base_col])
    base = metrics(test[target], test[base_col])
    rows = []
    for family, cols in families.items():
        coverage = family_new_coverage(test, family, cols)
        for model_name, model in model_specs("dbr_residual").items():
            model.fit(train[cols], train._target)
            phat = model.predict(test[cols])
            corrected = np.clip(num(test[base_col]).to_numpy(float) + phat, 0.25, 0.90)
            mm = metrics(test[target], corrected)
            rc = float(pd.Series(test._target.to_numpy(float)).corr(pd.Series(phat)))
            rows.append({
                "family": family, "model": model_name, "feature_count": len(cols), "new_feature_median_coverage": coverage,
                "dbr_residual_corr": rc, "corrected_dbr_mae": mm["mae"], "corrected_dbr_rmse": mm["rmse"], "corrected_dbr_corr": mm["corr"],
                "base_dbr_mae": base["mae"], "base_dbr_rmse": base["rmse"], "base_dbr_corr": base["corr"],
                "dbr_mae_gain": base["mae"] - mm["mae"], "dbr_corr_gain": mm["corr"] - base["corr"],
            })
    return pd.DataFrame(rows)


def attempt_tests(x: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    train = x[num(x.season).eq(2024)].copy()
    test = x[num(x.season).eq(2025)].copy()
    train["_target"] = num(train.actual_pass_att) - num(train.attempts_raw)
    test["_target"] = num(test.actual_pass_att) - num(test.attempts_raw)
    base = metrics(test.actual_pass_att, test.attempts_raw, 10)
    rows = []
    for family, cols in families.items():
        coverage = family_new_coverage(test, family, cols)
        for model_name, model in model_specs("attempt_residual").items():
            model.fit(train[cols], train._target)
            phat = model.predict(test[cols])
            corrected = np.clip(num(test.attempts_raw).to_numpy(float) + phat, 15.0, 55.0)
            mm = metrics(test.actual_pass_att, corrected, 10)
            rc = float(pd.Series(test._target.to_numpy(float)).corr(pd.Series(phat)))
            rows.append({
                "family": family, "model": model_name, "feature_count": len(cols), "new_feature_median_coverage": coverage,
                "attempt_residual_corr": rc, "corrected_attempt_mae": mm["mae"], "corrected_attempt_rmse": mm["rmse"], "corrected_attempt_corr": mm["corr"],
                "corrected_10plus": mm["miss"], "raw_attempt_mae": base["mae"], "raw_attempt_corr": base["corr"], "raw_10plus": base["miss"],
                "attempt_mae_gain": base["mae"] - mm["mae"], "attempt_corr_gain": mm["corr"] - base["corr"],
            })
    return pd.DataFrame(rows)


def univariate_screen(x: pd.DataFrame, new_cols: list[str]) -> pd.DataFrame:
    raw_col = "m64_pass_raw_reference" if "m64_pass_raw_reference" in x else "mc_proj_attempts_raw_only"
    y = {
        "pass_residual": num(x.actual) - num(x[raw_col]),
        "attempt_residual": num(x.actual_pass_att) - num(x.attempts_raw),
        "dbr_residual": num(x.m64_actual_dropback_rate) - num(x.m64_pred_dropback_rate_neutral),
    }
    rows = []
    for c in new_cols:
        s = num(x[c])
        for target, yy in y.items():
            vals = {}
            for season in (2024, 2025):
                mask = num(x.season).eq(season) & s.notna() & yy.notna()
                vals[season] = float(s[mask].corr(yy[mask])) if mask.sum() >= 20 else np.nan
            mask = s.notna() & yy.notna()
            combined = float(s[mask].corr(yy[mask])) if mask.sum() >= 40 else np.nan
            same = np.isfinite(vals[2024]) and np.isfinite(vals[2025]) and vals[2024] * vals[2025] > 0
            strong = bool(same and abs(vals[2024]) >= 0.10 and abs(vals[2025]) >= 0.10 and abs(combined) >= 0.15)
            rows.append({"feature": c, "target": target, "corr_2024": vals[2024], "corr_2025": vals[2025], "corr_combined": combined, "strong_replicated": strong})
    return pd.DataFrame(rows)


def interpretation(resid: pd.DataFrame, dbr: pd.DataFrame, att: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    live_names = {"pbp_intent_live", "injury_availability_live", "live_new_combined", "existing_plus_live_new"}
    hist_names = {"participation_personnel_historical_only", "all_new_combined", "existing_plus_all_new"}
    passes = []
    for family in sorted(set(resid.family) | set(dbr.family) | set(att.family)):
        r = resid[resid.family.eq(family)]
        d = dbr[dbr.family.eq(family)]
        a = att[att.family.eq(family)]
        coverage = float(pd.concat([r.new_feature_median_coverage, d.new_feature_median_coverage, a.new_feature_median_coverage]).median())
        pass_ok = bool(((r.residual_corr >= .20) & (r.mae_gain_vs_raw >= 1.0) & (r.corr_gain_vs_raw >= .03) & (r.corrected_100plus <= r.raw_100plus)).any())
        dbr_ok = bool(((d.dbr_residual_corr >= .20) & (d.dbr_mae_gain >= .0075) & (d.dbr_corr_gain >= .10)).any())
        att_ok = bool(((a.attempt_residual_corr >= .20) & (a.attempt_mae_gain >= .25) & (a.attempt_corr_gain >= .05) & (a.corrected_10plus <= np.floor(a.raw_10plus * .95))).any())
        if coverage >= .75 and (pass_ok or dbr_ok or att_ok):
            passes.append((family, pass_ok, dbr_ok, att_ok, coverage))
    live = [p for p in passes if p[0] in live_names]
    hist = [p for p in passes if p[0] in hist_names]
    if live:
        verdict = "live_new_information_breakthrough_followup"
    elif hist:
        verdict = "historical_personnel_signal_acquire_live_source"
    else:
        verdict = "seek_other_new_information_playcaller_leverage_transition"
    return pd.DataFrame([{
        "live_new_information_actionable": bool(live),
        "historical_only_information_actionable": bool(hist),
        "actionable_families": "|".join(p[0] for p in passes),
        "strong_replicated_new_feature_target_pairs": int(uni.strong_replicated.sum()),
        "m67_interpretation": verdict,
    }])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m65-game-level", type=Path, required=True)
    p.add_argument("--m65-state-features", type=Path, required=True)
    p.add_argument("--new-features", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    g, sf, nf = read(args.m65_game_level), read(args.m65_state_features), read(args.new_features)
    x, families, existing = prepare(g, sf, nf)
    new_cols = list(dict.fromkeys([c for cols in families.values() for c in cols if c.startswith(("intent_", "availability_", "personnel_", "continuity_"))]))
    resid, pred = residual_tests(x, families)
    dbr = dbr_tests(x, families)
    att = attempt_tests(x, families)
    uni = univariate_screen(x, new_cols)
    verdict = interpretation(resid, dbr, att, uni)
    pd.DataFrame([{"family": k, "feature_count": len(v), "features": "|".join(v)} for k, v in families.items()]).to_csv(args.out_dir / "m67_feature_families.csv", index=False)
    resid.to_csv(args.out_dir / "m67_pass_residual_family_metrics.csv", index=False)
    dbr.to_csv(args.out_dir / "m67_dropback_family_metrics.csv", index=False)
    att.to_csv(args.out_dir / "m67_attempt_family_metrics.csv", index=False)
    uni.sort_values(["strong_replicated", "target", "corr_combined"], ascending=[False, True, False]).to_csv(args.out_dir / "m67_univariate_new_information_screen.csv", index=False)
    pred.to_csv(args.out_dir / "m67_2025_pass_residual_predictions.csv", index=False)
    verdict.to_csv(args.out_dir / "m67_precommitted_interpretation.csv", index=False)
    print("=== M67 PRECOMMITTED INTERPRETATION ===")
    print(verdict.to_string(index=False))
    print("=== M67 PASS RESIDUAL ===")
    print(resid.to_string(index=False))
    print("=== M67 DROPBACK ===")
    print(dbr.to_string(index=False))
    print("=== M67 ATTEMPTS ===")
    print(att.to_string(index=False))
    print("=== M67 STRONG REPLICATED NEW FEATURES ===")
    print(uni[uni.strong_replicated].sort_values("corr_combined", key=lambda s: s.abs(), ascending=False).head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
