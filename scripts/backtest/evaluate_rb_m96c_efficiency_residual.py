"""M96C: M94C-anchored RB efficiency residual synthesis.

Research-only. Protocol frozen in docs/migrations/M96C_RB_M94C_EFFICIENCY_RESIDUAL.md.
M94C carries/yard center are fixed. Point candidates predict only leakage-safe
YPC residuals. X is a tail-only explosive-context audit.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALIASES = {"audricestime": "audricestim"}
KEYS = ["season", "week", "team", "player_join_key"]
START_TEST_WEEK = 6
MIN_TRAIN_ROWS = 150
RIDGE_ALPHA = 10.0
LOGIT_C = 0.1
MIN_ACTUAL_CARRIES_FOR_EFF_TRAIN = 5

E_BLOCK = [
    "pfr_ybc_per_att_avg3", "pfr_ybc_per_att_avg5",
    "ngs_expected_yards_per_att_avg3", "ngs_expected_yards_per_att_avg5",
    "ngs_percent_attempts_gte_eight_defenders_avg3", "ngs_percent_attempts_gte_eight_defenders_avg5",
    "ngs_avg_time_to_los_avg3", "ngs_avg_time_to_los_avg5",
    "team_pfr_ybc_per_att_avg3", "team_pfr_ybc_per_att_avg5",
    "team_pbp_stuff_rate_avg3", "team_pbp_stuff_rate_avg5",
    "rel_ybc_vs_team_avg3", "rel_ybc_vs_team_avg5",
]
P_BLOCK = [
    "pfr_yac_per_att_avg3", "pfr_yac_per_att_avg5",
    "pfr_brk_tkl_per_att_avg3", "pfr_brk_tkl_per_att_avg5",
    "ngs_ryoe_per_att_avg3", "ngs_ryoe_per_att_avg5",
    "ngs_rush_pct_over_expected_avg3", "ngs_rush_pct_over_expected_avg5",
    "rel_yac_vs_team_avg3", "rel_yac_vs_team_avg5",
]
D_BLOCK = [
    "def_rush_ypa_allowed_avg3", "def_rush_ypa_allowed_avg5",
    "def_rush_epa_allowed_avg3", "def_rush_epa_allowed_avg5",
    "def_rush_success_allowed_avg3", "def_rush_success_allowed_avg5",
    "def_rush_first_down_rate_allowed_avg3", "def_rush_first_down_rate_allowed_avg5",
    "def_non_scramble_ypa_allowed_avg3", "def_non_scramble_ypa_allowed_avg5",
    "def_stuff_rate_allowed_avg3", "def_stuff_rate_allowed_avg5",
    "def_rb_ypc_allowed_avg3", "def_rb_ypc_allowed_avg5",
    "def_rb_over_prior5_rush_yards_allowed_avg3", "def_rb_over_prior5_rush_yards_allowed_avg5",
]
X_BLOCK = [
    "player_pbp_explosive10_rate_avg3", "player_pbp_explosive10_rate_avg5",
    "player_pbp_explosive15_rate_avg3", "player_pbp_explosive15_rate_avg5",
    "player_pbp_explosive20_rate_avg3", "player_pbp_explosive20_rate_avg5",
    "team_pbp_explosive10_rate_avg3", "team_pbp_explosive10_rate_avg5",
    "team_pbp_explosive20_rate_avg3", "team_pbp_explosive20_rate_avg5",
    "def_explosive10_rate_allowed_avg3", "def_explosive10_rate_allowed_avg5",
    "def_explosive15_rate_allowed_avg3", "def_explosive15_rate_allowed_avg5",
    "def_explosive20_rate_allowed_avg3", "def_explosive20_rate_allowed_avg5",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def canon_team(x):
    s = "" if pd.isna(x) else str(x).upper().strip()
    return {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAX": "JAC"}.get(s, s)


def clean_player_key(x):
    s = "" if pd.isna(x) else str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return ALIASES.get(s, s)


def prep(x):
    z = x.copy()
    z.columns = [str(c).lower() for c in z.columns]
    z["season"] = num(z["season"]).astype(int)
    z["week"] = num(z["week"]).astype(int)
    z["team"] = z["team"].map(canon_team)
    if "player_clean_key" not in z.columns:
        raise RuntimeError("missing player_clean_key")
    z["player_join_key"] = z["player_clean_key"].map(clean_player_key)
    return z


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def point_metrics(actual, pred):
    q = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan,
                "actual_mean": np.nan, "pred_mean": np.nan}
    err = q["pred"] - q["actual"]
    corr = q["actual"].corr(q["pred"]) if len(q) >= 3 and q["actual"].nunique() > 1 and q["pred"].nunique() > 1 else np.nan
    return {
        "n": int(len(q)), "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))), "bias": float(err.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
        "actual_mean": float(q["actual"].mean()), "pred_mean": float(q["pred"].mean()),
    }


def prob_metrics(y, p):
    q = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if q.empty:
        return {"n": 0, "events": 0, "base_rate": np.nan, "mean_prob": np.nan,
                "auc": np.nan, "brier": np.nan, "logloss": np.nan}
    yy = q["y"].astype(int)
    pp = q["p"].clip(1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan
    return {"n": int(len(q)), "events": int(yy.sum()), "base_rate": float(yy.mean()),
            "mean_prob": float(pp.mean()), "auc": auc,
            "brier": float(np.mean(np.square(pp - yy))),
            "logloss": float(log_loss(yy, pp, labels=[0, 1]))}


def ridge_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def logit_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=LOGIT_C, max_iter=2500, random_state=9603)),
    ])


def available(z, cols):
    return [c for c in cols if c in z.columns and num(z[c]).notna().sum() >= 50]


def load_join(m94c_root: Path, m95d_root: Path):
    d = prep(pd.read_csv(find_one(m95d_root, "m95d_rb_environment_trace.csv"), low_memory=False))
    c = prep(pd.read_csv(find_one(m94c_root, "m94c_2025_rb_trace.csv"), low_memory=False))
    d = d.loc[d["season"].eq(2025)].copy()
    c = c.loc[c["season"].eq(2025)].copy()
    if d.duplicated(KEYS).any() or c.duplicated(KEYS).any():
        raise RuntimeError("duplicate player-game keys in M96C source")
    keep = KEYS + ["candidate_rush_yards", "candidate_rush_att", "actual_rush_yards", "actual_rush_att"]
    j = d.merge(c[keep], on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_m94c"))
    coverage = len(j) / max(len(d), 1)
    if coverage < 0.97:
        raise RuntimeError(f"M96C join coverage below 97% of M95D 2025 rows: {len(j)}/{len(d)}")
    yard_diff = np.abs(num(j["actual_rush_yards"]) - num(j["actual_rush_yards_m94c"]))
    att_diff = np.abs(num(j["actual_carries"]) - num(j["actual_rush_att"])) if "actual_carries" in j else pd.Series([0.0])
    if yard_diff.max() > 1e-6:
        raise RuntimeError(f"M96C yard truth parity failed: {yard_diff.max()}")
    if att_diff.max() > 1e-6:
        raise RuntimeError(f"M96C carry truth parity failed: {att_diff.max()}")
    audit = pd.DataFrame([
        {"source": "m95d_2025", "rows": len(d), "joined_rows": len(j), "coverage": coverage},
        {"source": "m94c_2025", "rows": len(c), "joined_rows": len(j), "coverage": len(j)/max(len(c),1)},
    ])
    audit["max_yard_truth_diff"] = float(yard_diff.max())
    audit["max_carry_truth_diff"] = float(att_diff.max())
    return j, audit


def add_targets(j):
    z = j.copy()
    c_att = num(z["candidate_rush_att"])
    c_yards = num(z["candidate_rush_yards"])
    z["m94c_implied_ypc"] = np.where(c_att.abs() > 1e-9, c_yards / c_att, np.nan)
    if not np.isfinite(z["m94c_implied_ypc"].dropna()).all():
        raise RuntimeError("M96C non-finite M94C implied YPC")
    actual_att = num(z["actual_rush_att"])
    actual_yards = num(z["actual_rush_yards_m94c"])
    z["actual_ypc_train"] = np.where(actual_att.ge(MIN_ACTUAL_CARRIES_FOR_EFF_TRAIN), actual_yards / actual_att, np.nan)
    z["efficiency_residual_ypc"] = z["actual_ypc_train"] - z["m94c_implied_ypc"]
    z["actual_75"] = actual_yards.ge(75).astype(int)
    z["actual_100"] = actual_yards.ge(100).astype(int)
    return z


def block_contract(z):
    blocks = {"E": available(z, E_BLOCK), "P": available(z, P_BLOCK),
              "D": available(z, D_BLOCK), "X": available(z, X_BLOCK)}
    for k in ["E", "P", "D", "X"]:
        if not blocks[k]:
            raise RuntimeError(f"M96C block {k} has no available features")
    arms = {
        "E": blocks["E"], "P": blocks["P"], "D": blocks["D"],
        "E+P": blocks["E"] + blocks["P"], "E+D": blocks["E"] + blocks["D"],
        "P+D": blocks["P"] + blocks["D"], "E+P+D": blocks["E"] + blocks["P"] + blocks["D"],
    }
    rows = []
    for name, feats in blocks.items():
        rows.append({"block": name, "feature_count": len(feats), "features": "|".join(feats),
                     "min_nonnull": min(int(num(z[c]).notna().sum()) for c in feats),
                     "max_nonnull": max(int(num(z[c]).notna().sum()) for c in feats)})
    return blocks, arms, pd.DataFrame(rows)


def expanding_point_oof(z, arms):
    q = z.copy()
    q["pred_C"] = num(q["candidate_rush_yards"])
    correction_rows = []
    for arm in arms:
        q[f"pred_{arm}"] = np.nan
        q[f"delta_{arm}"] = np.nan
    for week in range(START_TEST_WEEK, 19):
        train = q.loc[num(q["week"]).lt(week) & num(q["efficiency_residual_ypc"]).notna()].copy()
        test = q.loc[num(q["week"]).eq(week)].copy()
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        lo, hi = np.nanquantile(num(train["efficiency_residual_ypc"]), [0.05, 0.95])
        y = num(train["efficiency_residual_ypc"]).clip(lo, hi)
        for arm, feats in arms.items():
            model = ridge_model()
            model.fit(train[feats], y)
            delta = np.clip(model.predict(test[feats]), lo, hi)
            pred = num(test["candidate_rush_yards"]) + num(test["candidate_rush_att"]) * delta
            q.loc[test.index, f"delta_{arm}"] = delta
            q.loc[test.index, f"pred_{arm}"] = pred
            correction_rows.append({"week": week, "arm": arm, "train_n": len(train),
                                    "clip_lo": float(lo), "clip_hi": float(hi),
                                    "mean_delta": float(np.mean(delta)),
                                    "mean_abs_delta": float(np.mean(np.abs(delta)))})
    return q, pd.DataFrame(correction_rows)


def slice_masks(z):
    a = num(z["actual_rush_att"])
    return {"all_rb": pd.Series(True, index=z.index), "actual_0_5": a.between(0, 5),
            "actual_6_10": a.between(6, 10), "actual_11_14": a.between(11, 14),
            "actual_15_19": a.between(15, 19), "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25)}


def point_tables(oof, arms):
    evalz = oof.loc[num(oof["week"]).ge(START_TEST_WEEK)].copy()
    rows = []
    scopes = {"weeks6_18": pd.Series(True, index=evalz.index), "weeks13_18": num(evalz["week"]).ge(13)}
    arm_cols = {"C": "pred_C", **{arm: f"pred_{arm}" for arm in arms}}
    for scope, sm in scopes.items():
        s = evalz.loc[sm].copy()
        for sl, mask in slice_masks(s).items():
            g = s.loc[mask]
            for arm, col in arm_cols.items():
                rows.append({"scope": scope, "slice": sl, "arm": arm,
                             **point_metrics(g["actual_rush_yards_m94c"], g[col])})
    return pd.DataFrame(rows)


def point_gate(point, arms):
    def row(scope, sl, arm):
        q = point.loc[(point.scope.eq(scope)) & (point.slice.eq(sl)) & (point.arm.eq(arm))]
        if q.empty:
            raise RuntimeError(f"missing M96C point row {scope}/{sl}/{arm}")
        return q.iloc[0]
    base = row("weeks6_18", "all_rb", "C")
    gates = []
    for arm, feats in arms.items():
        cand = row("weeks6_18", "all_rb", arm)
        late = row("weeks13_18", "all_rb", arm)
        late_base = row("weeks13_18", "all_rb", "C")
        slice_regs = []
        slice_gains = []
        for sl in ["actual_0_5", "actual_6_10", "actual_11_14", "actual_15_19", "actual_20_plus"]:
            b = row("weeks6_18", sl, "C"); c = row("weeks6_18", sl, arm)
            if int(b["n"]) >= 50:
                slice_regs.append(float(c["mae"] - b["mae"]))
                slice_gains.append(float(b["mae"] - c["mae"]))
        max_reg = max(slice_regs) if slice_regs else np.nan
        mae_gain = float(base["mae"] - cand["mae"])
        rmse_gain = float(base["rmse"] - cand["rmse"])
        bias_ok = abs(float(cand["bias"])) <= abs(float(base["bias"])) + 1.0
        late_reg = float(late["mae"] - late_base["mae"])
        passed = mae_gain >= 0.25 and rmse_gain >= 0 and bias_ok and max_reg <= 1.0 and late_reg <= 0.50
        conditional = (not passed) and any(g >= 0.50 for g in slice_gains) and any(g <= -0.50 for g in slice_gains)
        gates.append({"arm": arm, "feature_count": len(feats), "mae_gain": mae_gain,
                      "rmse_gain": rmse_gain, "base_bias": float(base["bias"]), "candidate_bias": float(cand["bias"]),
                      "max_gated_slice_mae_regression": max_reg, "late_all_mae_regression": late_reg,
                      "global_gate_pass": int(passed), "conditional_clue": int(conditional)})
    g = pd.DataFrame(gates)
    passing = g.loc[g["global_gate_pass"].eq(1)].copy()
    selected = None
    if not passing.empty:
        best = float(passing["mae_gain"].max())
        pool = passing.loc[passing["mae_gain"].ge(best - 0.10)].copy()
        pool = pool.sort_values(["feature_count", "mae_gain", "arm"], ascending=[True, False, True])
        selected = str(pool.iloc[0]["arm"])
    g["selected_global_arm"] = selected if selected else "NONE"
    return g, selected


def expanding_tail_oof(z, x_feats):
    q = z.copy()
    out = []
    for threshold, target in [(75, "actual_75"), (100, "actual_100")]:
        p_base = pd.Series(np.nan, index=q.index, dtype=float)
        p_x = pd.Series(np.nan, index=q.index, dtype=float)
        for week in range(START_TEST_WEEK, 19):
            train = q.loc[num(q["week"]).lt(week)].copy()
            test = q.loc[num(q["week"]).eq(week)].copy()
            if len(train) < MIN_TRAIN_ROWS or test.empty or train[target].nunique() < 2:
                continue
            for feats, holder in [(["candidate_rush_yards"], p_base), (["candidate_rush_yards"] + x_feats, p_x)]:
                model = logit_model()
                model.fit(train[feats], train[target].astype(int))
                holder.loc[test.index] = model.predict_proba(test[feats])[:, 1]
        for scope, mask in {"weeks6_18": num(q["week"]).ge(START_TEST_WEEK), "weeks13_18": num(q["week"]).ge(13)}.items():
            for arm, p in [("C", p_base), ("C+X", p_x)]:
                out.append({"threshold": threshold, "scope": scope, "arm": arm,
                            **prob_metrics(q.loc[mask, target], p.loc[mask])})
    return pd.DataFrame(out)


def tail_gate(tail):
    details = []
    material_any = False
    safe = True
    for threshold in [75, 100]:
        b = tail.loc[(tail.threshold.eq(threshold)) & (tail.scope.eq("weeks6_18")) & (tail.arm.eq("C"))].iloc[0]
        x = tail.loc[(tail.threshold.eq(threshold)) & (tail.scope.eq("weeks6_18")) & (tail.arm.eq("C+X"))].iloc[0]
        auc_gain = float(x["auc"] - b["auc"])
        brier_gain = float(b["brier"] - x["brier"])
        material_any = material_any or auc_gain >= 0.01 or brier_gain >= 0.001
        safe = safe and auc_gain >= -0.01 and brier_gain >= -0.002
        details.append((threshold, auc_gain, brier_gain))
    retain = material_any and safe
    return pd.DataFrame([{"module": "X", "tail_gate_pass": int(retain),
                          "detail": "; ".join(f"{t}:auc_gain={a:.6f},brier_gain={b:.6f}" for t,a,b in details)}])


def capability_ledger(point_gate_df, tail_gate_df):
    rows = [{"module": "C", "status": "RETAIN", "detail": "Frozen M94C carry/yard point anchor"}]
    for module in ["E", "P", "D"]:
        affected = point_gate_df.loc[point_gate_df["arm"].str.split("+", regex=False).apply(lambda xs: module in xs)]
        if (affected["global_gate_pass"] == 1).any():
            status = "RETAIN_IN_PASSING_COMBINATION"
        elif (affected["conditional_clue"] == 1).any():
            status = "CONDITIONAL_CLUE"
        else:
            status = "REJECT_GLOBAL"
        rows.append({"module": module, "status": status,
                     "detail": "Point residual block; see m96c_point_gate.csv for arm-level ablations"})
    xpass = int(tail_gate_df.iloc[0]["tail_gate_pass"])
    rows.append({"module": "X", "status": "RETAIN_TAIL_ONLY" if xpass else "REJECT_TAIL_INCREMENT",
                 "detail": str(tail_gate_df.iloc[0]["detail"])})
    return pd.DataFrame(rows)


def casebook(oof, arms):
    q = oof.loc[num(oof["week"]).ge(START_TEST_WEEK)].copy()
    base_err = (num(q["pred_C"]) - num(q["actual_rush_yards_m94c"])).abs()
    rows = []
    for arm in arms:
        cand_err = (num(q[f"pred_{arm}"]) - num(q["actual_rush_yards_m94c"])).abs()
        cols = KEYS + ["player", "actual_rush_att", "actual_rush_yards_m94c", "candidate_rush_att", "candidate_rush_yards", f"pred_{arm}", f"delta_{arm}"]
        q2 = q[cols].copy(); q2["arm"] = arm; q2["mae_improvement"] = base_err - cand_err
        rows.append(q2.nlargest(8, "mae_improvement")); rows.append(q2.nsmallest(8, "mae_improvement"))
    return pd.concat(rows, ignore_index=True).sort_values(["arm", "mae_improvement"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--m95d-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    joined, source = load_join(args.m94c_root, args.m95d_root)
    joined = add_targets(joined)
    blocks, arms, features = block_contract(joined)
    oof, corrections = expanding_point_oof(joined, arms)
    point = point_tables(oof, arms)
    gates, selected = point_gate(point, arms)
    tail = expanding_tail_oof(joined, blocks["X"])
    xgate = tail_gate(tail)
    ledger = capability_ledger(gates, xgate)
    cases = casebook(oof, arms)

    conditional_any = bool(gates["conditional_clue"].eq(1).any())
    if selected:
        disposition = "M96C_GLOBAL_EFFICIENCY_ARM_RETAINED_REQUIRE_PROSPECTIVE_CONFIRMATION"
        next_step = "M96D_PROSPECTIVE_OR_SEALED_CONFIRMATION_OF_SELECTED_EFFICIENCY_ARM"
    elif conditional_any:
        disposition = "M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED"
        next_step = "M96D_PRECOMMITTED_CONDITIONAL_EFFICIENCY_ROUTING_AUDIT"
    else:
        disposition = "M96C_NO_SAFE_EFFICIENCY_INCREMENT_RETAIN_M94C"
        next_step = "PROSPECTIVE_CONFIRMATION_WITHOUT_MORE_RETROSPECTIVE_EFFICIENCY_TUNING"
    disp = pd.DataFrame([{"selected_global_arm": selected if selected else "NONE",
        "x_tail_retained": int(xgate.iloc[0]["tail_gate_pass"]), "conditional_signal": int(conditional_any),
        "disposition": disposition, "next_step": next_step, "model_fit": 1, "feature_search": 0,
        "weight_search": 0, "hyperparameter_search": 0, "sportsbook_inputs": 0, "production_change": 0}])

    source.to_csv(args.out_dir / "m96c_source_audit.csv", index=False)
    features.to_csv(args.out_dir / "m96c_feature_blocks.csv", index=False)
    corrections.to_csv(args.out_dir / "m96c_weekly_corrections.csv", index=False)
    point.to_csv(args.out_dir / "m96c_point_ablation.csv", index=False)
    gates.to_csv(args.out_dir / "m96c_point_gate.csv", index=False)
    tail.to_csv(args.out_dir / "m96c_tail_x_ablation.csv", index=False)
    xgate.to_csv(args.out_dir / "m96c_tail_x_gate.csv", index=False)
    ledger.to_csv(args.out_dir / "m96c_capability_ledger.csv", index=False)
    cases.to_csv(args.out_dir / "m96c_casebook.csv", index=False)
    keep = KEYS + ["player", "actual_rush_att", "actual_rush_yards_m94c", "candidate_rush_att", "candidate_rush_yards", "m94c_implied_ypc"]
    for arm in arms: keep += [f"delta_{arm}", f"pred_{arm}"]
    oof[[c for c in keep if c in oof.columns]].to_csv(args.out_dir / "m96c_oof_trace.csv", index=False)
    disp.to_csv(args.out_dir / "m96c_disposition.csv", index=False)

    print("=== M96C source audit ==="); print(source.to_string(index=False))
    print("=== M96C feature blocks ==="); print(features[["block","feature_count","min_nonnull","max_nonnull"]].to_string(index=False))
    print("=== M96C all-RB point ablation ==="); print(point.loc[point["slice"].eq("all_rb")].to_string(index=False))
    print("=== M96C point gates ==="); print(gates.to_string(index=False))
    print("=== M96C X tail ==="); print(tail.to_string(index=False))
    print("=== M96C capability ledger ==="); print(ledger.to_string(index=False))
    print("=== M96C disposition ==="); print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
