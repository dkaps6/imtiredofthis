"""M95O: precommitted agreement-gated stable-workhorse 20+ tail candidate.

Derived from M95N diagnostic evidence. This candidate does not retune M95K to
opened 2023 labels. It uses a fixed 2024 W13-15 pregame reference distribution to
decide whether frozen M95F current-context and frozen M95K historical-feed evidence
agree. Discordant stable-workhorse rows remain exactly M95F. On aligned rows only,
the frozen M95K 20+ ranking is used and mean-anchored back to the aligned M95F
probability mass. Stable-workhorse 25+ remains M95F. No sportsbook or production
change. M94C central carries are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scripts.backtest.evaluate_rb_feed_tendency_carry_ceiling import (
    PLAYER_KEYS, SPECS, add_composites, available, build_feed_features,
    candidate_probs, lower, mean_anchor, num, pipe, prep,
)

FEED20 = [
    "player_season_rate20_eb", "player_season_q95_eb",
    "team_lead_season_rate20_eb", "team_lead_season_q95_eb",
]
DEV_WEEKS = (13, 15)
EVAL_WEEKS = (16, 18)
SHRINK_K = 4.0
SPEC = "feed_compact_env"
C = 0.03


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def prob_metrics(y, p) -> dict:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if z.empty:
        return {"n": 0, "base_rate": np.nan, "mean_prob": np.nan,
                "auc": np.nan, "brier": np.nan, "logloss": np.nan,
                "positive_events": 0}
    yy = z.y.astype(int)
    pp = z.p.clip(1e-6, 1 - 1e-6)
    return {
        "n": int(len(z)),
        "base_rate": float(yy.mean()),
        "mean_prob": float(pp.mean()),
        "auc": float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan,
        "brier": float(np.mean((pp - yy) ** 2)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
        "positive_events": int(yy.sum()),
    }


def fixed_cdf(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = np.sort(num(reference).dropna().to_numpy(dtype=float))
    vals = num(values)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(ref) == 0:
        return out
    ok = vals.notna()
    out.loc[ok] = np.searchsorted(ref, vals.loc[ok].to_numpy(dtype=float), side="right") / len(ref)
    return out.clip(0, 1)


def add_fixed_gate_scores(z: pd.DataFrame, dev_ref: pd.DataFrame) -> pd.DataFrame:
    x = z.copy()
    feed_parts = []
    for c in FEED20:
        if c not in x.columns or c not in dev_ref.columns:
            raise RuntimeError(f"M95O missing gate field {c}")
        feed_parts.append(fixed_cdf(x[c], dev_ref[c]).rename(c))
    x["feed_score_fixed"] = pd.concat(feed_parts, axis=1).mean(axis=1)
    x["context_score_fixed"] = fixed_cdf(x["p20_base"], dev_ref["p20_base"])
    hi_feed = x["feed_score_fixed"].ge(0.50)
    hi_context = x["context_score_fixed"].ge(0.50)
    x["m95o_regime"] = np.select(
        [hi_feed & hi_context, hi_feed & ~hi_context, ~hi_feed & hi_context],
        ["aligned_high", "history_only", "context_only"],
        default="aligned_low",
    )
    x["m95o_aligned"] = x["m95o_regime"].isin(["aligned_high", "aligned_low"]).astype(int)
    return x


def apply_gate(z: pd.DataFrame, dev_ref: pd.DataFrame, k_col: str) -> pd.DataFrame:
    x = add_fixed_gate_scores(z, dev_ref)
    if k_col not in x.columns:
        raise RuntimeError(f"M95O missing frozen K probability {k_col}")
    x["p20_base"] = num(x["p20_base"]).clip(1e-6, 1 - 1e-6)
    x["p20_k"] = num(x[k_col]).clip(1e-6, 1 - 1e-6)
    x["p20_m95o"] = x["p20_base"]
    eligible = x["m95o_aligned"].eq(1) & x["p20_k"].notna() & x["p20_base"].notna()
    if eligible.sum() >= 2:
        anchored = mean_anchor(
            x.loc[eligible, "p20_k"].to_numpy(dtype=float),
            float(x.loc[eligible, "p20_base"].mean()),
        )
        x.loc[eligible, "p20_m95o"] = anchored
    x["p25_m95o"] = num(x["p25_base"])
    x["delta20_m95o"] = x["p20_m95o"] - x["p20_base"]
    return x


def stable_standardize(df: pd.DataFrame, season: int, k_col: str) -> pd.DataFrame:
    x = lower(df)
    req = [*PLAYER_KEYS, "actual_carries", "stable_workhorse_m95k", "p20_base", "p25_base", k_col, *FEED20]
    missing = [c for c in req if c not in x.columns]
    if missing:
        raise RuntimeError(f"M95O trace missing required fields: {missing}")
    x = x.loc[num(x["season"]).eq(season) & num(x["stable_workhorse_m95k"]).eq(1)].copy()
    x["actual_carries"] = num(x["actual_carries"])
    x["actual_20plus"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus"] = x["actual_carries"].ge(25).astype(int)
    x["p20_base"] = num(x["p20_base"])
    x["p25_base"] = num(x["p25_base"])
    x[k_col] = num(x[k_col])
    return x


def metric_rows(scope: str, x: pd.DataFrame) -> list[dict]:
    y = x["actual_20plus"]
    out = []
    for model, col in [("m95f", "p20_base"), ("m95k_frozen", "p20_k"), ("m95o_gate", "p20_m95o")]:
        out.append({"scope": scope, "target": "actual_20plus", "model": model, **prob_metrics(y, x[col])})
    return out


def regime_rows(scope: str, x: pd.DataFrame) -> list[dict]:
    rows = []
    for reg, q in x.groupby("m95o_regime"):
        b = prob_metrics(q["actual_20plus"], q["p20_base"])
        k = prob_metrics(q["actual_20plus"], q["p20_k"])
        o = prob_metrics(q["actual_20plus"], q["p20_m95o"])
        rows.append({
            "scope": scope, "regime": reg, "n": int(len(q)),
            "positive_events": int(q["actual_20plus"].sum()),
            "event_rate": float(q["actual_20plus"].mean()) if len(q) else np.nan,
            "mean_context_score": float(num(q["context_score_fixed"]).mean()),
            "mean_feed_score": float(num(q["feed_score_fixed"]).mean()),
            "mean_m95o_shift": float(num(q["delta20_m95o"]).mean()),
            "base_auc": b["auc"], "k_auc": k["auc"], "m95o_auc": o["auc"],
            "base_brier": b["brier"], "k_brier": k["brier"], "m95o_brier": o["brier"],
        })
    return rows


def make_2024(m95b_root: Path, m95g_root: Path, m94c_root: Path):
    history = pd.read_csv(find_one(m95b_root, "m95b_rb_matchup_trace.csv"), low_memory=False)
    g24 = pd.read_csv(find_one(m95g_root, "m95g_2024_holdout_trace.csv"), low_memory=False)
    t24 = pd.read_csv(find_one(m94c_root, "m94c_2024_holdout_trace.csv"), low_memory=False)
    b24all = prep(g24, t24)
    b24 = b24all.loc[
        b24all["stable_workhorse_m95k"].eq(1) & num(b24all["week"]).between(13, 18)
    ].copy()
    ff = build_feed_features(history, SHRINK_K)
    z24 = add_composites(b24.merge(ff, on=PLAYER_KEYS, how="left", validate="one_to_one"))
    dev = z24.loc[num(z24["week"]).between(*DEV_WEEKS)].copy()
    sel = z24.loc[num(z24["week"]).between(*EVAL_WEEKS)].copy()
    fs = available(dev, SPECS[SPEC])
    model = pipe(C)
    model.fit(dev[fs], num(dev["actual_carries"]).ge(20).astype(int))
    p20, _ = candidate_probs(model, sel, fs)
    sel["p20_k_2024"] = p20
    sel["actual_20plus"] = num(sel["actual_carries"]).ge(20).astype(int)
    sel["actual_25plus"] = num(sel["actual_carries"]).ge(25).astype(int)
    return dev, sel, fs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--m95g-root", type=Path, required=True)
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--m95k-root", type=Path, required=True)
    ap.add_argument("--m95l-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dev24, sel24, fs = make_2024(args.m95b_root, args.m95g_root, args.m94c_root)
    x24 = sel24.copy()
    x24["p20_k"] = num(x24["p20_k_2024"])
    x24 = apply_gate(x24, dev24, "p20_k")

    k25 = pd.read_csv(find_one(args.m95k_root, "m95k_2025_trace.csv"), low_memory=False)
    l23 = pd.read_csv(find_one(args.m95l_root, "m95l_2023_confirmation_trace.csv"), low_memory=False)
    x25 = stable_standardize(k25, 2025, "p20_m95k")
    x23 = stable_standardize(l23, 2023, "p20_m95l")
    x25 = x25.rename(columns={"p20_m95k": "p20_k"})
    x23 = x23.rename(columns={"p20_m95l": "p20_k"})
    x25 = apply_gate(x25, dev24, "p20_k")
    x23 = apply_gate(x23, dev24, "p20_k")

    frames = {
        "2024_w16_18_development_selection": x24,
        "2025_full_research": x25,
        "2025_w13_18_research": x25.loc[num(x25["week"]).between(13, 18)].copy(),
        "2023_w13_18_opened": x23.loc[num(x23["week"]).between(13, 18)].copy(),
    }

    metrics = pd.DataFrame([r for scope, z in frames.items() for r in metric_rows(scope, z)])
    regimes = pd.DataFrame([r for scope, z in frames.items() for r in regime_rows(scope, z)])

    year_rows = []
    for scope, z in frames.items():
        year_rows.append({
            "scope": scope, "n": len(z), "unique_players": z["player_clean_key"].nunique(),
            "event20_rate": float(z["actual_20plus"].mean()),
            "event25_rate": float(z["actual_25plus"].mean()),
            "mean_actual_carries": float(num(z["actual_carries"]).mean()),
            "mean_m95f_p20": float(num(z["p20_base"]).mean()),
            "mean_m95o_p20": float(num(z["p20_m95o"]).mean()),
            "aligned_share": float(z["m95o_aligned"].mean()),
        })
    year_context = pd.DataFrame(year_rows)

    def row(scope: str, model: str):
        q = metrics.loc[metrics.scope.eq(scope) & metrics.model.eq(model)]
        if len(q) != 1:
            raise RuntimeError(f"missing metric row {scope}/{model}")
        return q.iloc[0]

    s23b, s23o = row("2023_w13_18_opened", "m95f"), row("2023_w13_18_opened", "m95o_gate")
    s25fb, s25fo = row("2025_full_research", "m95f"), row("2025_full_research", "m95o_gate")
    s25lb, s25lo = row("2025_w13_18_research", "m95f"), row("2025_w13_18_research", "m95o_gate")
    s24b, s24o = row("2024_w16_18_development_selection", "m95f"), row("2024_w16_18_development_selection", "m95o_gate")

    mass_errs = [abs(float(num(z["p20_m95o"]).mean() - num(z["p20_base"]).mean())) for z in frames.values()]
    mass_preserved = int(max(mass_errs) < 1e-9)

    guard23 = int(s23o["auc"] >= s23b["auc"] - 0.02 and s23o["brier"] <= s23b["brier"] + 0.005)
    retain25full = int(s25fo["auc"] >= s25fb["auc"] + 0.02 and s25fo["brier"] <= s25fb["brier"])
    retain25late = int(s25lo["auc"] >= s25lb["auc"] and s25lo["brier"] <= s25lb["brier"] + 0.005)
    advance = int(guard23 and retain25full and retain25late and mass_preserved)
    disposition = "ADVANCE_M95O_TO_NEW_UNTOUCHED_CONFIRMATION_PROTOCOL" if advance else "RETAIN_M95O_AS_DIAGNOSTIC_DO_NOT_PROMOTE"

    disp = pd.DataFrame([{
        "m95o_role": "research_candidate_not_production",
        "gate_2023_no_material_regression": guard23,
        "gate_2025_full_retains_value": retain25full,
        "gate_2025_late_nonnegative": retain25late,
        "stable_probability_mass_preserved": mass_preserved,
        "retrospective_research_pass": advance,
        "m95f_2023_auc": float(s23b["auc"]), "m95o_2023_auc": float(s23o["auc"]),
        "m95f_2023_brier": float(s23b["brier"]), "m95o_2023_brier": float(s23o["brier"]),
        "m95f_2025_full_auc": float(s25fb["auc"]), "m95o_2025_full_auc": float(s25fo["auc"]),
        "m95f_2025_full_brier": float(s25fb["brier"]), "m95o_2025_full_brier": float(s25fo["brier"]),
        "m95f_2025_late_auc": float(s25lb["auc"]), "m95o_2025_late_auc": float(s25lo["auc"]),
        "m95f_2024_sel_auc": float(s24b["auc"]), "m95o_2024_sel_auc": float(s24o["auc"]),
        "development_year": 2024,
        "development_note": "2024_used_by_m95k_selection_not_independent_confirmation",
        "opened_years_note": "2023_w13_18_and_2025_are_research_evidence_not_pristine_confirmation",
        "feature_search": 0, "coefficient_search": 0, "sportsbook_inputs": 0,
        "production_change": 0, "m94c_central_carries_changed": 0,
        "stable_25_changed": 0, "disposition": disposition,
    }])

    method = pd.DataFrame([{
        "development_reference": "2024_w13_15",
        "evaluation_2024": "w16_18_development_selection_only",
        "frozen_k": SHRINK_K, "frozen_spec": SPEC, "frozen_C": C,
        "frozen_features": "|".join(fs),
        "gate_feed_fields": "|".join(FEED20),
        "gate_threshold": "fixed_empirical_CDF_median_from_2024_w13_15_no_outcomes",
        "aligned_behavior": "frozen_m95k_ranking_logit_mean_anchored_to_aligned_m95f_mass",
        "discordant_behavior": "exact_m95f",
        "stable20_mass_preservation": 1,
        "stable25_behavior": "exact_m95f",
    }])

    metrics.to_csv(args.out_dir / "m95o_probability_metrics.csv", index=False)
    regimes.to_csv(args.out_dir / "m95o_regime_metrics.csv", index=False)
    year_context.to_csv(args.out_dir / "m95o_year_context.csv", index=False)
    disp.to_csv(args.out_dir / "m95o_disposition.csv", index=False)
    method.to_csv(args.out_dir / "m95o_method_audit.csv", index=False)
    for name, z in frames.items():
        z.to_csv(args.out_dir / f"m95o_trace_{name}.csv", index=False)

    print("[m95o] year context")
    print(year_context.to_string(index=False))
    print("\n[m95o] probability metrics")
    print(metrics.to_string(index=False))
    print("\n[m95o] regime metrics")
    print(regimes.to_string(index=False))
    print("\n[m95o] disposition")
    print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
