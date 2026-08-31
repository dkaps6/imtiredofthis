"""M95N: conditional player-game micro-regime audit.

Research-only descriptive audit. M95M established that the frozen M95K stable-
workhorse reranker is cross-season nonstationary. M95N tests a narrower question:
do current pregame model context and historical feed/ceiling information behave
more consistently when they agree, while disagreement requires a different
response function?

This migration fits no model, searches no features/coefficients, changes no
probabilities, and makes no production change. All regime inputs are pregame-only.
Outcome labels are used only after regime assignment for evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]

FEED20 = [
    "player_season_rate20_eb", "player_season_q95_eb",
    "team_lead_season_rate20_eb", "team_lead_season_q95_eb",
]
FEED25 = [
    "player_season_rate25_eb", "player_season_q95_eb",
    "team_lead_season_rate25_eb", "team_lead_season_q95_eb",
]
VOLUME = ["candidate_team_rush_att", "pred_lead_play_share"]
CONCENTRATION_POS = ["rb_rb_share_avg3", "team_top1_share_avg3"]
CONCENTRATION_NEG = ["team_rb_used_avg3"]
MATCHUP = [
    "def_vulnerability_score", "def_rb_20plus_carry_rate_allowed_avg5",
    "def_top_rb_carries_allowed_avg5", "def_rb_carries_allowed_avg5",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c).lower() for c in x.columns]; return x


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return hits[0]


def prob_metrics(y, p) -> dict:
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if z.empty:
        return {"n": 0, "base_rate": np.nan, "mean_prob": np.nan, "auc": np.nan, "brier": np.nan, "logloss": np.nan, "positive_events": 0}
    yy = z.y.astype(int); pp = z.p.clip(1e-6, 1 - 1e-6)
    return {
        "n": int(len(z)), "base_rate": float(yy.mean()), "mean_prob": float(pp.mean()),
        "auc": float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan,
        "brier": float(np.mean((pp - yy) ** 2)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
        "positive_events": int(yy.sum()),
    }


def standardize(df: pd.DataFrame, season: int, candidate_suffix: str) -> pd.DataFrame:
    x = lower(df)
    req = [*PLAYER_KEYS, "actual_carries", "stable_workhorse_m95k", "p20_base", "p25_base"]
    req += sorted(set(FEED20 + FEED25 + VOLUME + CONCENTRATION_POS + CONCENTRATION_NEG + MATCHUP + ["rb_rb_share_avg1", "rb_rb_share_avg5"]))
    missing = [c for c in req if c not in x.columns]
    if missing:
        raise RuntimeError(f"M95N trace missing required columns: {missing}")
    c20, c25 = f"p20_{candidate_suffix}", f"p25_{candidate_suffix}"
    if c20 not in x.columns or c25 not in x.columns:
        raise RuntimeError(f"M95N missing candidate columns {c20}/{c25}")
    x = x.loc[num(x["season"]).eq(season) & num(x["stable_workhorse_m95k"]).eq(1)].copy()
    x["actual_carries"] = num(x["actual_carries"])
    x["actual_20plus"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus"] = x["actual_carries"].ge(25).astype(int)
    x["candidate_p20"] = num(x[c20]); x["candidate_p25"] = num(x[c25])
    x["p20_base"] = num(x["p20_base"]); x["p25_base"] = num(x["p25_base"])
    x["delta20"] = x["candidate_p20"] - x["p20_base"]
    x["delta25"] = x["candidate_p25"] - x["p25_base"]
    return x


def rank_mean(z: pd.DataFrame, cols: list[str], inverse: set[str] | None = None) -> pd.Series:
    inverse = inverse or set(); arr = []
    for c in cols:
        r = num(z[c]).rank(pct=True, method="average")
        if c in inverse: r = 1.0 - r
        arr.append(r.rename(c))
    return pd.concat(arr, axis=1).mean(axis=1)


def assign_regimes(z0: pd.DataFrame, scope: str, th: int) -> pd.DataFrame:
    z = z0.copy(); feed_cols = FEED20 if th == 20 else FEED25
    z["feed_score"] = rank_mean(z, feed_cols)
    z["baseline_context_rank"] = num(z[f"p{th}_base"]).rank(pct=True, method="average")
    high_context = z["baseline_context_rank"].ge(0.50)
    high_feed = z["feed_score"].ge(0.50)
    z["micro_regime"] = np.select(
        [high_context & high_feed, ~high_context & high_feed, high_context & ~high_feed],
        ["aligned_high", "history_only", "context_only"],
        default="aligned_low",
    )
    z["volume_score"] = rank_mean(z, VOLUME)
    z["concentration_score"] = rank_mean(z, CONCENTRATION_POS + CONCENTRATION_NEG, set(CONCENTRATION_NEG))
    z["matchup_score"] = rank_mean(z, MATCHUP)
    z["role_momentum"] = num(z["rb_rb_share_avg1"]) - num(z["rb_rb_share_avg5"])
    z["role_momentum_rank"] = num(z["role_momentum"]).rank(pct=True, method="average")
    z["scope"] = scope; z["target"] = f"actual_{th}plus"
    return z


def regime_metrics(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, assigned = [], []
    for scope, z0 in frames.items():
        for th in (20, 25):
            z = assign_regimes(z0, scope, th); assigned.append(z)
            for reg, q in z.groupby("micro_regime"):
                b = prob_metrics(q[f"actual_{th}plus"], q[f"p{th}_base"])
                c = prob_metrics(q[f"actual_{th}plus"], q[f"candidate_p{th}"])
                rows.append({
                    "scope": scope, "target": f"actual_{th}plus", "micro_regime": reg,
                    "n": int(len(q)), "positive_events": int(q[f"actual_{th}plus"].sum()),
                    "event_rate": float(q[f"actual_{th}plus"].mean()),
                    "mean_feed_score": float(q["feed_score"].mean()),
                    "mean_baseline_context_rank": float(q["baseline_context_rank"].mean()),
                    "mean_candidate_shift": float(num(q[f"delta{th}"]).mean()),
                    "base_auc": b["auc"], "candidate_auc": c["auc"],
                    "auc_gain": c["auc"] - b["auc"] if np.isfinite(c["auc"]) and np.isfinite(b["auc"]) else np.nan,
                    "base_brier": b["brier"], "candidate_brier": c["brier"],
                    "brier_gain": b["brier"] - c["brier"],
                })
    return pd.DataFrame(rows), pd.concat(assigned, ignore_index=True)


def secondary_gate_table(assigned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dims = ["volume_score", "concentration_score", "matchup_score", "role_momentum_rank"]
    q0 = assigned.loc[assigned["micro_regime"].isin(["history_only", "context_only"])].copy()
    for (scope, target, reg), q in q0.groupby(["scope", "target", "micro_regime"]):
        for dim in dims:
            hi = num(q[dim]).ge(0.50)
            for label, mask in [("low", ~hi), ("high", hi)]:
                qq = q.loc[mask]
                truth = qq[target]
                rows.append({
                    "scope": scope, "target": target, "micro_regime": reg,
                    "secondary_dimension": dim, "secondary_level": label,
                    "n": int(len(qq)), "positive_events": int(truth.sum()) if len(qq) else 0,
                    "event_rate": float(truth.mean()) if len(qq) else np.nan,
                })
    return pd.DataFrame(rows)


def get_rate(m: pd.DataFrame, scope: str, target: str, regime: str) -> float:
    q = m.loc[m.scope.eq(scope) & m.target.eq(target) & m.micro_regime.eq(regime)]
    return float(q.event_rate.iloc[0]) if len(q) else np.nan


def disposition(metrics: pd.DataFrame) -> pd.DataFrame:
    s23, s25 = "2023_w13_18", "2025_w13_18"
    t = "actual_20plus"
    ah23, al23 = get_rate(metrics, s23, t, "aligned_high"), get_rate(metrics, s23, t, "aligned_low")
    ah25, al25 = get_rate(metrics, s25, t, "aligned_high"), get_rate(metrics, s25, t, "aligned_low")
    co23, ho23 = get_rate(metrics, s23, t, "context_only"), get_rate(metrics, s23, t, "history_only")
    co25, ho25 = get_rate(metrics, s25, t, "context_only"), get_rate(metrics, s25, t, "history_only")
    aligned_order_stable = int(np.isfinite([ah23, al23, ah25, al25]).all() and ah23 > al23 and ah25 > al25)
    discordant_flip = int(np.isfinite([co23, ho23, co25, ho25]).all() and co23 > ho23 and ho25 > co25)
    same_window25_events = int(metrics.loc[metrics.scope.isin([s23, s25]) & metrics.target.eq("actual_25plus"), "positive_events"].sum())
    micro_regime_evidence = int(aligned_order_stable and discordant_flip)
    return pd.DataFrame([{
        "m95n_role": "diagnostic_only_no_model_fit",
        "aligned_20plus_order_stable": aligned_order_stable,
        "discordant_20plus_preference_flips_by_season": discordant_flip,
        "aligned_high_rate_2023": ah23, "aligned_low_rate_2023": al23,
        "aligned_high_rate_2025": ah25, "aligned_low_rate_2025": al25,
        "context_only_rate_2023": co23, "history_only_rate_2023": ho23,
        "context_only_rate_2025": co25, "history_only_rate_2025": ho25,
        "same_window_25plus_events": same_window25_events,
        "micro_regime_dependence_supported": micro_regime_evidence,
        "feature_search": 0, "coefficient_search": 0, "new_model_fit": 0,
        "sportsbook_inputs": 0, "production_change": 0,
        "interpretation": "agreement_is_stable_signal_disagreement_requires_conditional_response" if micro_regime_evidence else "no_clear_micro_regime_structure",
        "next_rule": "if supported, test a precommitted agreement-gated architecture; do not select a discordant expert using opened 2023 labels",
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95k-root", type=Path, required=True)
    ap.add_argument("--m95l-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    k = pd.read_csv(find_one(args.m95k_root, "m95k_2025_trace.csv"), low_memory=False)
    l = pd.read_csv(find_one(args.m95l_root, "m95l_2023_confirmation_trace.csv"), low_memory=False)
    x25 = standardize(k, 2025, "m95k")
    x23 = standardize(l, 2023, "m95l")
    frames = {
        "2023_w13_18": x23.loc[num(x23.week).between(13, 18)].copy(),
        "2025_w13_18": x25.loc[num(x25.week).between(13, 18)].copy(),
        "2025_full_research": x25.copy(),
    }

    metrics, assigned = regime_metrics(frames)
    secondary = secondary_gate_table(assigned)
    disp = disposition(metrics)
    case_cols = [
        "scope", "target", "season", "week", "team", "player_clean_key", "actual_carries",
        "micro_regime", "feed_score", "baseline_context_rank", "volume_score",
        "concentration_score", "matchup_score", "role_momentum", "p20_base", "candidate_p20",
        "delta20", "p25_base", "candidate_p25", "delta25",
    ]
    cases = assigned.loc[assigned.micro_regime.isin(["history_only", "context_only"]), [c for c in case_cols if c in assigned.columns]].copy()

    metrics.to_csv(args.out_dir / "m95n_micro_regime_metrics.csv", index=False)
    secondary.to_csv(args.out_dir / "m95n_secondary_gate_audit.csv", index=False)
    cases.to_csv(args.out_dir / "m95n_discordant_casebook.csv", index=False)
    disp.to_csv(args.out_dir / "m95n_disposition.csv", index=False)
    pd.DataFrame([{
        "regime_definition": "within-scope median split of frozen M95F baseline context rank and fixed historical feed score",
        "feed20_fields": "|".join(FEED20), "feed25_fields": "|".join(FEED25),
        "volume_fields": "|".join(VOLUME),
        "concentration_fields": "|".join(CONCENTRATION_POS + CONCENTRATION_NEG),
        "matchup_fields": "|".join(MATCHUP),
        "outcome_used_for_regime_assignment": 0, "model_fit": 0,
    }]).to_csv(args.out_dir / "m95n_method_audit.csv", index=False)

    print("[m95n] micro-regime metrics")
    print(metrics.to_string(index=False))
    print("\n[m95n] disposition")
    print(disp.to_string(index=False))
    print("\n[m95n] secondary discordant audit")
    print(secondary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
