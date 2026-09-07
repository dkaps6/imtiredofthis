#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPECTED_ROWS = 884
CONTEXT = [
    "qb_prior_attempts", "qb_prior_ypa", "off_true_proe", "off_neutral_pace",
    "def_pass_epa_allowed", "def_success_allowed", "def_ypa_allowed",
    "off_plays", "off_pass_rate", "def_pass_rate_faced",
]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(v) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def corr(a, b, method: str) -> float:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method=method))


def score(name: str, g: pd.DataFrame, pred_col: str) -> dict:
    z = g.loc[num(g[pred_col]).notna() & num(g["attempt_mechanism_share"]).notna()].copy()
    if z.empty:
        return {"view": name, "n": 0, "pearson": np.nan, "spearman": np.nan}
    p = num(z[pred_col])
    y = num(z["attempt_mechanism_share"])
    q1, q4 = float(p.quantile(.25)), float(p.quantile(.75))
    lo, hi = p.le(q1), p.ge(q4)
    att_dom = z["mechanism_state"].eq("ATTEMPTS_DOMINANT")
    return {
        "view": name,
        "n": int(len(z)),
        "pearson": corr(p, y, "pearson"),
        "spearman": corr(p, y, "spearman"),
        "pred_q1": q1,
        "pred_q4": q4,
        "actual_share_q1": float(y.loc[lo].mean()),
        "actual_share_q4": float(y.loc[hi].mean()),
        "actual_share_q4_minus_q1_gap": float(y.loc[hi].mean() - y.loc[lo].mean()),
        "attempts_dominant_rate_q1": float(att_dom.loc[lo].mean()),
        "attempts_dominant_rate_q4": float(att_dom.loc[hi].mean()),
        "attempts_dominant_q4_minus_q1_gap": float(att_dom.loc[hi].mean() - att_dom.loc[lo].mean()),
        "w2_18_spearman": corr(p.loc[z.week.between(2,18)], y.loc[z.week.between(2,18)], "spearman"),
        "w13_18_spearman": corr(p.loc[z.week.between(13,18)], y.loc[z.week.between(13,18)], "spearman"),
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    pipe.fit(train[features], train["attempt_mechanism_share"])
    return np.clip(pipe.predict(test[features]), 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qb-mechanism-root", type=Path, required=True)
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    mech = pd.read_csv(one(a.qb_mechanism_root, "qb_mechanism_casebook.csv"), low_memory=False)
    feat = pd.read_csv(one(a.m89_root, "m89_2024_2025_synthesis_trace.csv"), low_memory=False)
    mech.columns = [str(c).strip().lower() for c in mech.columns]
    feat.columns = [str(c).strip().lower() for c in feat.columns]
    if len(mech) != EXPECTED_ROWS:
        raise RuntimeError(f"QB mechanism row drift expected={EXPECTED_ROWS} got={len(mech)}")

    keys = ["season", "week", "team", "player_clean_key"]
    need_m = set(keys + ["attempt_component", "ypa_component"])
    need_f = set(keys + CONTEXT)
    if need_m - set(mech.columns):
        raise RuntimeError(f"mechanism casebook missing {sorted(need_m-set(mech.columns))}")
    if need_f - set(feat.columns):
        raise RuntimeError(f"M89 synthesis trace missing {sorted(need_f-set(feat.columns))}")

    for d in (mech, feat):
        d["season"] = num(d["season"])
        d["week"] = num(d["week"])
        d["team"] = d["team"].fillna("").astype(str).str.upper().str.strip()
        d["player_clean_key"] = d["player_clean_key"].fillna("").astype(str)

    f = feat.loc[feat.season.isin([2024,2025]), keys + CONTEXT].copy()
    if f.duplicated(keys).any():
        raise RuntimeError("duplicate M89 synthesis-trace keys for 2024-2025")
    x = mech.merge(f, on=keys, how="left", validate="one_to_one", indicator=True)
    if len(x) != EXPECTED_ROWS or not x["_merge"].eq("both").all():
        raise RuntimeError(f"mechanism/feature alignment failed rows={len(x)} matched={int(x['_merge'].eq('both').sum())}")
    x = x.drop(columns=["_merge"])

    x["attempt_component"] = num(x["attempt_component"])
    x["ypa_component"] = num(x["ypa_component"])
    den = x["attempt_component"].abs() + x["ypa_component"].abs()
    x["attempt_mechanism_share"] = np.where(den.gt(0), x["attempt_component"].abs() / den, np.nan)
    x["mechanism_state"] = "MIXED"
    x.loc[x["attempt_component"].abs().ge(1.25*x["ypa_component"].abs()), "mechanism_state"] = "ATTEMPTS_DOMINANT"
    x.loc[x["ypa_component"].abs().ge(1.25*x["attempt_component"].abs()), "mechanism_state"] = "YPA_DOMINANT"

    x = x.sort_values(["player_clean_key", "season", "week", "team"], kind="stable").reset_index(drop=True)
    gp = x.groupby("player_clean_key", sort=False)
    x["prior4_player_attempt_share"] = gp["attempt_mechanism_share"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=3).mean()
    )

    train = x.loc[x.season.eq(2024) & x.attempt_mechanism_share.notna()].copy()
    test = x.loc[x.season.eq(2025) & x.attempt_mechanism_share.notna()].copy()
    if len(train) < 300 or len(test) < 300:
        raise RuntimeError(f"insufficient temporal rows train={len(train)} test={len(test)}")

    views = {
        "PLAYER_ONLY": ["prior4_player_attempt_share"],
        "CONTEXT_ONLY": CONTEXT,
        "COMBINED": ["prior4_player_attempt_share"] + CONTEXT,
    }
    score_rows = []
    for name, features in views.items():
        col = f"pred_{name.lower()}"
        test[col] = fit_predict(train, test, features)
        score_rows.append(score(name, test, col))

    scores = pd.DataFrame(score_rows)
    combined = next(r for r in score_rows if r["view"] == "COMBINED")
    context = next(r for r in score_rows if r["view"] == "CONTEXT_ONLY")
    raw_player_spearman = corr(
        test["prior4_player_attempt_share"], test["attempt_mechanism_share"], "spearman"
    )
    raw_player_n = int(test["prior4_player_attempt_share"].notna().sum())

    individual_value = bool(
        (np.isfinite(combined["spearman"]) and np.isfinite(context["spearman"]) and combined["spearman"] >= context["spearman"] + 0.03)
        or (np.isfinite(raw_player_spearman) and raw_player_spearman >= 0.10)
    )
    gates = {
        "test_n_ge_300": bool(combined["n"] >= 300),
        "combined_spearman_ge_0_15": bool(np.isfinite(combined["spearman"]) and combined["spearman"] >= 0.15),
        "combined_q4_q1_share_gap_ge_0_10": bool(np.isfinite(combined["actual_share_q4_minus_q1_gap"]) and combined["actual_share_q4_minus_q1_gap"] >= 0.10),
        "combined_attempt_dom_rate_gap_ge_0_12": bool(np.isfinite(combined["attempts_dominant_q4_minus_q1_gap"]) and combined["attempts_dominant_q4_minus_q1_gap"] >= 0.12),
        "combined_w2_18_spearman_positive": bool(np.isfinite(combined["w2_18_spearman"]) and combined["w2_18_spearman"] > 0),
        "combined_w13_18_spearman_positive": bool(np.isfinite(combined["w13_18_spearman"]) and combined["w13_18_spearman"] > 0),
        "individual_value_condition": individual_value,
    }
    passed = all(gates.values())
    disposition = "QB_PLAYER_CONTEXT_MECHANISM_ROUTER_DISCOVERY_PASS" if passed else "NO_ACTIONABLE_QB_PLAYER_CONTEXT_MECHANISM_ROUTER"

    result = {
        "migration": "QB_R1_PLAYER_CONTEXT_MECHANISM_ROUTER",
        "source_file": "m89_2024_2025_synthesis_trace.csv",
        "context_features": CONTEXT,
        "rows": int(len(x)),
        "train_2024_rows": int(len(train)),
        "test_2025_rows": int(len(test)),
        "mechanism_state_counts_all": {str(k): int(v) for k,v in x.mechanism_state.value_counts().to_dict().items()},
        "raw_player_prior_2025_n": raw_player_n,
        "raw_player_prior_2025_spearman": raw_player_spearman,
        "combined_minus_context_spearman": float(combined["spearman"]-context["spearman"]) if np.isfinite(combined["spearman"]) and np.isfinite(context["spearman"]) else np.nan,
        "views": {r["view"]: {k:v for k,v in r.items() if k!="view"} for r in score_rows},
        "gates": gates,
        "sportsbook_inputs_used": False,
        "model_fitting_used": True,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_r1_mechanism_router_casebook.csv", index=False)
    test.to_csv(a.out_dir / "qb_r1_2025_oos_scores.csv", index=False)
    scores.to_csv(a.out_dir / "qb_r1_view_metrics.csv", index=False)
    (a.out_dir / "qb_r1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(scores.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
