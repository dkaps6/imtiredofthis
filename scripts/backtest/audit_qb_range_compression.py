#!/usr/bin/env python3
"""Migration 57: audit whether caps/shrinkage suppress useful QB range.

Diagnostic only. Re-fits the exact walk-forward attempts and contextual-YPA
residual models from Migration 53, then exposes raw, cap-only, shrink-only, and
cap+shrink stages before any production change is considered.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.diagnose_qb_gamescript_attribution import FORCE_PASS, MARKET, OPP_OFFENSE, PACE, TEAM_TENDENCY, prepare
from scripts.backtest.fit_qb_gamescript_attempts_walkforward import metrics, num, read
from scripts.backtest.fit_qb_joint_attempts_ypa_walkforward import add_qb_and_matchup_context

ATT_CAP = 5.0
YPA_CAP = 1.5
SHRINK = 0.60


def fit_raw_residual(train, test, features, target, alpha=30.0):
    usable = [f for f in features if f in train and num(train[f]).notna().sum() >= 10 and num(train[f]).nunique() > 1]
    if not usable:
        return pd.Series(float(num(target).mean()), index=test.index)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
    model.fit(train[usable], num(target))
    return pd.Series(model.predict(test[usable]), index=test.index)


def stages(base, raw, cap, lo, hi):
    base, raw = num(base), num(raw)
    return {
        "base": base.clip(lo, hi),
        "raw": (base + raw).clip(lo, hi),
        "cap_only": (base + raw.clip(-cap, cap)).clip(lo, hi),
        "shrink_only": (base + raw * SHRINK).clip(lo, hi),
        "cap_shrink": (base + raw.clip(-cap, cap) * SHRINK).clip(lo, hi),
    }


def dist_metrics(actual, pred):
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty: return {}
    out = metrics(z.a, z.p)
    astd, pstd = float(z.a.std(ddof=0)), float(z.p.std(ddof=0))
    out.update({
        "actual_sd": astd, "pred_sd": pstd, "sd_ratio": pstd / astd if astd > 0 else np.nan,
        "pred_p10": float(z.p.quantile(.10)), "pred_p50": float(z.p.quantile(.50)), "pred_p90": float(z.p.quantile(.90)),
        "actual_p10": float(z.a.quantile(.10)), "actual_p50": float(z.a.quantile(.50)), "actual_p90": float(z.a.quantile(.90)),
    })
    return out


def auc(actual, score, threshold):
    z = pd.DataFrame({"a": num(actual), "s": num(score)}).dropna(); y = z.a.ge(threshold).astype(int)
    if len(z) < 20 or y.nunique() < 2: return np.nan
    try: return float(roc_auc_score(y, z.s))
    except Exception: return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True); p.add_argument("--market-trace", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True); p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-train", type=int, default=80); a = p.parse_args()

    x = prepare(a.market_trace, a.team_weekly, a.season)
    x = add_qb_and_matchup_context(x, read(a.player_logs), read(a.team_weekly), read(a.weather), a.season)
    attempt_features = MARKET + ["pred_attempts"] + PACE + TEAM_TENDENCY + OPP_OFFENSE + FORCE_PASS
    qb_features = ["pred_ypa", "qb_recent_ypa", "qb_recent_pass_att", "qb_recent_completion_pct", "qb_recent_td_rate", "qb_recent_int_rate", "qb_recent_epa_per_att"]
    matchup = ["team_pressure_rate_allowed", "team_success_rate_off", "opp_pressure_rate_generated", "opp_def_pass_epa", "opp_success_rate_def", "opp_explosive_play_rate_allowed", "opp_coverage_man_rate", "opp_coverage_zone_rate", "market_total", "market_team_implied", "market_opp_implied", "market_spread", "controlled_environment"]
    ypa_features = qb_features + matchup
    x["att_raw_delta"] = np.nan; x["ypa_raw_delta"] = np.nan; scored = []
    for week in sorted(num(x.week).dropna().astype(int).unique()):
        train, test = x[num(x.week) < week].copy(), x[num(x.week) == week].copy()
        if len(train) < a.min_train or test.empty: continue
        x.loc[test.index, "att_raw_delta"] = fit_raw_residual(train, test, attempt_features, num(train.actual_pass_att) - num(train.pred_attempts))
        x.loc[test.index, "ypa_raw_delta"] = fit_raw_residual(train, test, ypa_features, num(train.actual_ypa) - num(train.pred_ypa))
        scored.extend(test.index.tolist())
    x = x.loc[sorted(set(scored))].copy()
    if x.empty: raise RuntimeError("no OOS rows")

    att = stages(x.pred_attempts, x.att_raw_delta, ATT_CAP, 18, 48); ypa = stages(x.pred_ypa, x.ypa_raw_delta, YPA_CAP, 4.5, 10.5)
    for n, s in att.items(): x[f"attempts_{n}"] = s
    for n, s in ypa.items(): x[f"ypa_{n}"] = s
    combos = {
        "current_components": ("base", "base"), "joint_cap_shrink": ("cap_shrink", "cap_shrink"),
        "attempts_raw_only": ("raw", "cap_shrink"), "attempts_cap_only": ("cap_only", "cap_shrink"), "attempts_shrink_only": ("shrink_only", "cap_shrink"),
        "ypa_raw_only": ("cap_shrink", "raw"), "ypa_cap_only": ("cap_shrink", "cap_only"), "ypa_shrink_only": ("cap_shrink", "shrink_only"),
        "both_raw": ("raw", "raw"), "both_cap_only": ("cap_only", "cap_only"), "both_shrink_only": ("shrink_only", "shrink_only"),
    }
    for n, (aa, yy) in combos.items():
        x[f"pass_yards_{n}"] = num(x.mc_proj) * num(x[f"attempts_{aa}"]) / num(x.pred_attempts).replace(0, np.nan) * num(x[f"ypa_{yy}"]) / num(x.pred_ypa).replace(0, np.nan)

    rows = []
    for s in ["base", "raw", "cap_only", "shrink_only", "cap_shrink"]:
        r = {"season": a.season, "target": "attempts", "candidate": s, **dist_metrics(x.actual_pass_att, x[f"attempts_{s}"])}
        r["auc_40plus"] = auc(x.actual_pass_att, x[f"attempts_{s}"], 40); high = num(x.actual_pass_att).ge(40)
        r["mae_actual_40plus"] = float((num(x[f"attempts_{s}"]) - num(x.actual_pass_att))[high].abs().mean()); rows.append(r)
        rows.append({"season": a.season, "target": "ypa", "candidate": s, **dist_metrics(x.actual_ypa, x[f"ypa_{s}"])})
    for n in combos:
        r = {"season": a.season, "target": "pass_yards", "candidate": n, **dist_metrics(x.actual_pass_yards_raw, x[f"pass_yards_{n}"])}
        r["auc_300plus"] = auc(x.actual_pass_yards_raw, x[f"pass_yards_{n}"], 300); e = num(x[f"pass_yards_{n}"]) - num(x.actual_pass_yards_raw)
        r["catastrophic_100plus"] = int(e.abs().ge(100).sum()); r["under_100plus"] = int(e.le(-100).sum()); r["over_100plus"] = int(e.ge(100).sum()); rows.append(r)

    bind = pd.DataFrame([{ "season": a.season, "n": len(x), "attempt_raw_delta_sd": float(num(x.att_raw_delta).std(ddof=0)), "attempt_raw_delta_p10": float(num(x.att_raw_delta).quantile(.10)), "attempt_raw_delta_p90": float(num(x.att_raw_delta).quantile(.90)), "attempt_cap_bind_rate": float(num(x.att_raw_delta).abs().gt(ATT_CAP).mean()), "ypa_raw_delta_sd": float(num(x.ypa_raw_delta).std(ddof=0)), "ypa_raw_delta_p10": float(num(x.ypa_raw_delta).quantile(.10)), "ypa_raw_delta_p90": float(num(x.ypa_raw_delta).quantile(.90)), "ypa_cap_bind_rate": float(num(x.ypa_raw_delta).abs().gt(YPA_CAP).mean()) }])
    tier = pd.qcut(num(x.qb_recent_ypa).rank(method="first"), 3, labels=["lower_qb_history", "middle_qb_history", "higher_qb_history"])
    tiers = []
    for label, g in x.groupby(tier, observed=True):
        for c in ["joint_cap_shrink", "attempts_raw_only", "ypa_raw_only", "both_raw", "both_cap_only", "both_shrink_only"]:
            tiers.append({"season": a.season, "qb_tier": str(label), "candidate": c, **dist_metrics(g.actual_pass_yards_raw, g[f"pass_yards_{c}"])})
    a.out_dir.mkdir(parents=True, exist_ok=True); x.to_csv(a.out_dir/f"qb_range_compression_trace_{a.season}.csv", index=False)
    pd.DataFrame(rows).to_csv(a.out_dir/f"qb_range_compression_summary_{a.season}.csv", index=False); bind.to_csv(a.out_dir/f"qb_range_compression_bind_{a.season}.csv", index=False); pd.DataFrame(tiers).to_csv(a.out_dir/f"qb_range_compression_qb_tiers_{a.season}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print("\n=== BIND RATES ==="); print(bind.to_string(index=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
