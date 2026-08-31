"""M95C: RB quality vs blocking/environment decomposition.

Research-only. M95C freezes the successful M95B leakage-safe trace and asks a
narrow football question: can we separate what the rushing environment gives a
back from what the back creates himself, and does that separation improve
forward rushing-yard / efficiency prediction beyond raw rushing efficiency?

No sportsbook inputs. No production code changes. No 2025-driven feature or
hyperparameter search.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import scripts.backtest.evaluate_rb_offense_defense_matchup as b

TARGET_SPLITS = [
    ("train_2023_test_2024", [2023], 2024),
    ("train_2023_24_test_2025", [2023, 2024], 2025),
]

# The opportunity / identity control is frozen from M95B. These variables are
# intentionally present in every M95C family so the experiment is about
# efficiency attribution rather than rediscovering who gets the football.
ROLE = list(b.ROLE_FEATURES)

# Traditional / outcome-level rushing efficiency. These describe what happened
# but do not cleanly distinguish blocking from runner creation.
RAW_EFFICIENCY = [
    "player_pbp_ypa_avg3", "player_pbp_ypa_avg5",
    "player_pbp_epa_avg3", "player_pbp_epa_avg5",
    "player_pbp_success_avg3", "player_pbp_success_avg5",
    "player_pbp_first_down_rate_avg3", "player_pbp_first_down_rate_avg5",
    "player_pbp_stuff_rate_avg3", "player_pbp_stuff_rate_avg5",
    "player_pbp_explosive10_rate_avg3", "player_pbp_explosive10_rate_avg5",
    "player_pbp_explosive15_rate_avg3", "player_pbp_explosive15_rate_avg5",
    "player_pbp_explosive20_rate_avg3", "player_pbp_explosive20_rate_avg5",
]

# Environment / blocking opportunity: what is available before the runner has
# to create extra yardage. Stacked-box frequency and TLOS are player-observed
# context variables; expected yards is NGS's context-sensitive expectation.
ENVIRONMENT = [
    "pfr_ybc_per_att_avg3", "pfr_ybc_per_att_avg5",
    "ngs_expected_yards_per_att_avg3", "ngs_expected_yards_per_att_avg5",
    "ngs_percent_attempts_gte_eight_defenders_avg3", "ngs_percent_attempts_gte_eight_defenders_avg5",
    "ngs_avg_time_to_los_avg3", "ngs_avg_time_to_los_avg5",
    "team_pfr_ybc_per_att_avg3", "team_pfr_ybc_per_att_avg5",
    "team_pbp_stuff_rate_avg3", "team_pbp_stuff_rate_avg5",
]

# Runner-created value: after-contact production, broken tackles, and NGS
# performance over expectation. RYOE is especially valuable because expected
# rush yards already accounts for contextual difficulty.
CREATED = [
    "pfr_yac_per_att_avg3", "pfr_yac_per_att_avg5",
    "pfr_brk_tkl_per_att_avg3", "pfr_brk_tkl_per_att_avg5",
    "ngs_ryoe_per_att_avg3", "ngs_ryoe_per_att_avg5",
    "ngs_rush_pct_over_expected_avg3", "ngs_rush_pct_over_expected_avg5",
]

DERIVED = [
    "rel_ybc_vs_team_avg3", "rel_ybc_vs_team_avg5",
    "rel_yac_vs_team_avg3", "rel_yac_vs_team_avg5",
]

FAMILY_ORDER = [
    "role_baseline",
    "role_plus_raw_efficiency",
    "role_plus_environment",
    "role_plus_created",
    "role_plus_decomposition",
    "role_plus_decomposition_and_raw",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_trace(root: Path) -> pd.DataFrame:
    paths = list(root.rglob("m95b_rb_matchup_trace.csv"))
    if not paths:
        raise RuntimeError(f"missing m95b_rb_matchup_trace.csv under {root}")
    x = pd.read_csv(paths[0], low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x = x.loc[x["season"].isin([2023, 2024, 2025])].copy()

    # Leakage-safe relative player-vs-team environment/creation measures. Every
    # input is already a strictly lagged rolling feature from frozen M95B.
    for n in (3, 5):
        pybc = f"pfr_ybc_per_att_avg{n}"
        tybc = f"team_pfr_ybc_per_att_avg{n}"
        pyac = f"pfr_yac_per_att_avg{n}"
        tyac = f"team_pfr_yac_per_att_avg{n}"
        x[f"rel_ybc_vs_team_avg{n}"] = num(x[pybc]) - num(x[tybc]) if pybc in x and tybc in x else np.nan
        x[f"rel_yac_vs_team_avg{n}"] = num(x[pyac]) - num(x[tyac]) if pyac in x and tyac in x else np.nan

    # Recreate targets defensively if an older frozen trace lacks them.
    if "actual_ypc" not in x:
        x["actual_ypc"] = np.where(num(x["actual_carries"]).gt(0), num(x["actual_rush_yards"]) / num(x["actual_carries"]), np.nan)
    for cutoff in (75, 100):
        c = f"actual_rush_{cutoff}plus"
        if c not in x:
            x[c] = num(x["actual_rush_yards"]).ge(cutoff).astype(int)
    if "actual_player_explosive20" not in x:
        x["actual_player_explosive20"] = np.nan
    return x.reset_index(drop=True)


def available(x: pd.DataFrame, cols: list[str], min_nonnull: int = 50) -> list[str]:
    return [c for c in cols if c in x.columns and num(x[c]).notna().sum() >= min_nonnull]


def families(x: pd.DataFrame) -> dict[str, list[str]]:
    role = available(x, ROLE)
    raw = available(x, RAW_EFFICIENCY)
    env = available(x, ENVIRONMENT)
    created = available(x, CREATED)
    derived = available(x, DERIVED)
    return {
        "role_baseline": role,
        "role_plus_raw_efficiency": role + raw,
        "role_plus_environment": role + env + [c for c in derived if c.startswith("rel_ybc")],
        "role_plus_created": role + created + [c for c in derived if c.startswith("rel_yac")],
        "role_plus_decomposition": role + env + created + derived,
        "role_plus_decomposition_and_raw": role + raw + env + created + derived,
    }


def reg_metrics(actual, pred):
    z = pd.DataFrame({"actual": num(actual), "pred": np.asarray(pred)}).dropna()
    if not len(z):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = z["pred"] - z["actual"]
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(math.sqrt(np.square(e).mean())),
        "bias": float(e.mean()),
        "corr": float(z["actual"].corr(z["pred"])) if z["actual"].nunique() > 1 and z["pred"].nunique() > 1 else np.nan,
    }


def fit_all(x: pd.DataFrame):
    fams = families(x)
    results = []
    coefs = []
    preds = []
    regressions = [
        ("carries", "actual_carries", 0),
        ("rush_yards", "actual_rush_yards", 0),
        ("ypc_3plus", "actual_ypc", 3),
        ("ypc_8plus", "actual_ypc", 8),
    ]
    classes = [
        ("rush_75plus_auc", "actual_rush_75plus"),
        ("rush_100plus_auc", "actual_rush_100plus"),
        ("explosive20_auc", "actual_player_explosive20"),
    ]

    for split, train_seasons, test_season in TARGET_SPLITS:
        tr0 = x.loc[x["season"].isin(train_seasons) & x["pregame_role"].ne("unknown")].copy()
        te0 = x.loc[x["season"].eq(test_season) & x["pregame_role"].ne("unknown")].copy()
        for family in FAMILY_ORDER:
            feats = [c for c in fams[family] if c in tr0 and num(tr0[c]).notna().any()]
            Xtr0 = tr0[feats].apply(pd.to_numeric, errors="coerce")
            Xte0 = te0[feats].apply(pd.to_numeric, errors="coerce")

            for target, col, min_carries in regressions:
                if min_carries:
                    mtr = num(tr0["actual_carries"]).ge(min_carries)
                    mte = num(te0["actual_carries"]).ge(min_carries)
                else:
                    mtr = pd.Series(True, index=tr0.index)
                    mte = pd.Series(True, index=te0.index)
                ytr = num(tr0.loc[mtr, col])
                valid = ytr.notna()
                if valid.sum() < 50 or mte.sum() < 20:
                    continue
                model = b.ridge()
                model.fit(Xtr0.loc[mtr].loc[valid], ytr.loc[valid])
                pred = model.predict(Xte0.loc[mte])
                met = reg_metrics(te0.loc[mte, col], pred)
                results.append({
                    "split": split,
                    "train_seasons": ",".join(map(str, train_seasons)),
                    "test_season": test_season,
                    "family": family,
                    "target": target,
                    "feature_count": len(feats),
                    **met,
                })
                if target in {"rush_yards", "ypc_8plus"}:
                    for f, v in zip(feats, np.ravel(model.named_steps["model"].coef_)):
                        coefs.append({
                            "split": split, "family": family, "target": target,
                            "feature": f, "standardized_coefficient": float(v),
                            "abs_coefficient": abs(float(v)),
                        })
                if target in {"carries", "rush_yards"}:
                    for idx, p in zip(te0.loc[mte].index, pred):
                        preds.append({
                            "split": split, "test_season": test_season,
                            "row_index": int(idx), "family": family,
                            "target": target, "prediction": float(p),
                        })

            for target, col in classes:
                ytr = num(tr0[col])
                yte = num(te0[col])
                valid_tr = ytr.notna()
                valid_te = yte.notna()
                if valid_tr.sum() < 50 or valid_te.sum() < 20:
                    continue
                if ytr.loc[valid_tr].nunique() < 2 or yte.loc[valid_te].nunique() < 2:
                    continue
                model = b.logit()
                model.fit(Xtr0.loc[valid_tr], ytr.loc[valid_tr].astype(int))
                prob = model.predict_proba(Xte0.loc[valid_te])[:, 1]
                auc = float(roc_auc_score(yte.loc[valid_te].astype(int), prob))
                results.append({
                    "split": split,
                    "train_seasons": ",".join(map(str, train_seasons)),
                    "test_season": test_season,
                    "family": family,
                    "target": target,
                    "feature_count": len(feats),
                    "n": int(valid_te.sum()),
                    "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": auc,
                })

    return pd.DataFrame(results), pd.DataFrame(coefs), pd.DataFrame(preds), fams


def comparison(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, target), g in results.groupby(["split", "target"]):
        base = g.loc[g["family"].eq("role_baseline")]
        raw = g.loc[g["family"].eq("role_plus_raw_efficiency")]
        for family in FAMILY_ORDER:
            q = g.loc[g["family"].eq(family)]
            if q.empty:
                continue
            is_auc = target.endswith("_auc")
            value = float(q.iloc[0]["corr"] if is_auc else q.iloc[0]["mae"])
            base_value = float(base.iloc[0]["corr"] if is_auc else base.iloc[0]["mae"]) if len(base) else np.nan
            raw_value = float(raw.iloc[0]["corr"] if is_auc else raw.iloc[0]["mae"]) if len(raw) else np.nan
            if is_auc:
                vs_base = value - base_value if np.isfinite(base_value) else np.nan
                vs_raw = value - raw_value if np.isfinite(raw_value) else np.nan
                metric = "auc_gain"
            else:
                vs_base = base_value - value if np.isfinite(base_value) else np.nan
                vs_raw = raw_value - value if np.isfinite(raw_value) else np.nan
                metric = "mae_gain"
            rows.append({
                "split": split, "target": target, "family": family,
                "metric": metric, "value": value,
                "gain_vs_role": vs_base, "gain_vs_raw_efficiency": vs_raw,
            })
    return pd.DataFrame(rows)


def feature_coverage(x: pd.DataFrame, fams: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for family, feats in fams.items():
        for c in feats:
            if c in ROLE:
                source = "role_control"
            elif c in RAW_EFFICIENCY:
                source = "raw_efficiency"
            elif c in ENVIRONMENT or c.startswith("rel_ybc"):
                source = "environment"
            elif c in CREATED or c.startswith("rel_yac"):
                source = "runner_created"
            else:
                source = "other"
            rows.append({
                "family": family,
                "feature": c,
                "source_group": source,
                "nonnull_rows": int(num(x[c]).notna().sum()),
                "coverage": float(num(x[c]).notna().mean()),
            })
    return pd.DataFrame(rows).drop_duplicates(["family", "feature"])


def creator_environment_slices(x: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    q = x.loc[x["season"].eq(2025)].copy()
    # Pregame-only diagnostic archetypes. Quantiles are used only to describe
    # test-set behavior after predictions; they are never model features/gates.
    env = num(q.get("pfr_ybc_per_att_avg5", pd.Series(index=q.index, dtype=float)))
    create = num(q.get("ngs_ryoe_per_att_avg5", pd.Series(index=q.index, dtype=float)))
    if env.notna().sum() >= 30:
        loe, hie = env.quantile([.33, .67])
    else:
        loe = hie = np.nan
    if create.notna().sum() >= 30:
        loc, hic = create.quantile([.33, .67])
    else:
        loc = hic = np.nan
    q["archetype"] = "middle"
    if np.isfinite(loe) and np.isfinite(hic):
        q.loc[env.le(loe) & create.ge(hic), "archetype"] = "creator_low_ybc_high_ryoe"
    if np.isfinite(hie) and np.isfinite(loc):
        q.loc[env.ge(hie) & create.le(loc), "archetype"] = "environment_high_ybc_low_ryoe"

    rows = []
    p = preds.loc[(preds["test_season"].eq(2025)) & preds["target"].eq("rush_yards")]
    for family, g in p.groupby("family"):
        mp = g.set_index("row_index")["prediction"]
        z = q.loc[q.index.isin(mp.index)].copy()
        z["pred"] = mp.reindex(z.index)
        for name in ["creator_low_ybc_high_ryoe", "environment_high_ybc_low_ryoe", "middle"]:
            s = z.loc[z["archetype"].eq(name), ["actual_rush_yards", "pred"]].dropna()
            if len(s) < 8:
                continue
            e = s["pred"] - num(s["actual_rush_yards"])
            rows.append({
                "family": family, "slice": name, "n": int(len(s)),
                "mae": float(e.abs().mean()), "bias": float(e.mean()),
                "corr": float(num(s["actual_rush_yards"]).corr(s["pred"])) if s["actual_rush_yards"].nunique() > 1 and s["pred"].nunique() > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def disposition(results: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    def val(split, family, target, col="mae"):
        q = results.loc[
            results["split"].eq(split) & results["family"].eq(family) & results["target"].eq(target)
        ]
        return float(q.iloc[0][col]) if len(q) else np.nan

    fam = "role_plus_decomposition_and_raw"
    raw = "role_plus_raw_efficiency"
    ry24_raw = val("train_2023_test_2024", raw, "rush_yards")
    ry24_dec = val("train_2023_test_2024", fam, "rush_yards")
    ry25_raw = val("train_2023_24_test_2025", raw, "rush_yards")
    ry25_dec = val("train_2023_24_test_2025", fam, "rush_yards")
    ypc25_raw = val("train_2023_24_test_2025", raw, "ypc_8plus")
    ypc25_dec = val("train_2023_24_test_2025", fam, "ypc_8plus")
    c25_raw = val("train_2023_24_test_2025", raw, "carries")
    c25_dec = val("train_2023_24_test_2025", fam, "carries")
    auc100_raw = val("train_2023_24_test_2025", raw, "rush_100plus_auc", "corr")
    auc100_dec = val("train_2023_24_test_2025", fam, "rush_100plus_auc", "corr")
    aucx_raw = val("train_2023_24_test_2025", raw, "explosive20_auc", "corr")
    aucx_dec = val("train_2023_24_test_2025", fam, "explosive20_auc", "corr")

    stable_rush = bool(np.isfinite(ry24_raw) and np.isfinite(ry24_dec) and np.isfinite(ry25_raw) and np.isfinite(ry25_dec)
                       and ry24_dec < ry24_raw and ry25_dec < ry25_raw)
    efficiency_support = bool(np.isfinite(ypc25_raw) and np.isfinite(ypc25_dec) and ypc25_dec <= ypc25_raw)
    carry_guard = bool(not (np.isfinite(c25_raw) and np.isfinite(c25_dec)) or c25_dec <= c25_raw + 0.05)
    tail_support = bool((np.isfinite(auc100_raw) and np.isfinite(auc100_dec) and auc100_dec >= auc100_raw) or
                        (np.isfinite(aucx_raw) and np.isfinite(aucx_dec) and aucx_dec >= aucx_raw))
    advance = stable_rush and efficiency_support and carry_guard and tail_support
    return pd.DataFrame([{
        "rush_yards_raw_2024": ry24_raw,
        "rush_yards_decomposed_2024": ry24_dec,
        "rush_yards_raw_2025": ry25_raw,
        "rush_yards_decomposed_2025": ry25_dec,
        "ypc8_raw_2025": ypc25_raw,
        "ypc8_decomposed_2025": ypc25_dec,
        "carries_raw_2025": c25_raw,
        "carries_decomposed_2025": c25_dec,
        "rush100_auc_raw_2025": auc100_raw,
        "rush100_auc_decomposed_2025": auc100_dec,
        "explosive20_auc_raw_2025": aucx_raw,
        "explosive20_auc_decomposed_2025": aucx_dec,
        "stable_rush_gain_both_forward_years": int(stable_rush),
        "efficiency_support": int(efficiency_support),
        "carry_guard": int(carry_guard),
        "tail_support": int(tail_support),
        "disposition": "ADVANCE_M95C_QUALITY_ENVIRONMENT_DECOMPOSITION" if advance else "RETAIN_M95B_OFFENSE_PROFILE",
        "production_change": 0,
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m95c"))
    a = ap.parse_args()

    x = load_trace(a.m95b_root)
    results, coefs, preds, fams = fit_all(x)
    comp = comparison(results)
    cov = feature_coverage(x, fams)
    slices = creator_environment_slices(x, preds)
    disp = disposition(results, comp)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(a.out_dir / "m95c_model_comparison.csv", index=False)
    comp.to_csv(a.out_dir / "m95c_gain_vs_controls.csv", index=False)
    cov.to_csv(a.out_dir / "m95c_feature_coverage.csv", index=False)
    coefs.sort_values("abs_coefficient", ascending=False).to_csv(a.out_dir / "m95c_standardized_coefficients.csv", index=False)
    preds.to_csv(a.out_dir / "m95c_prediction_trace.csv", index=False)
    slices.to_csv(a.out_dir / "m95c_creator_environment_slices.csv", index=False)
    disp.to_csv(a.out_dir / "m95c_disposition.csv", index=False)

    print("[m95c] disposition\n", disp.to_string(index=False))
    print("\n[m95c] forward model comparison\n", results.to_string(index=False))
    print("\n[m95c] gains vs role/raw\n", comp.to_string(index=False))
    print("\n[m95c] creator/environment slices\n", slices.to_string(index=False))
    print("\n[m95c] source-group coverage\n", cov.groupby("source_group")["coverage"].agg(["count", "mean", "min", "max"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
