"""M95J: regime-specific RB workload conversion.

Research-only. M95I isolated two workload-tail populations: vacancy/role-transition
backs, where recipient-specific deep concentration is strong, and stable incumbent
workhorses, where role identity is already known but the specific high-volume week
is not. M95J freezes the M95I vacancy branch, leaves M95F as the ordinary baseline,
and fits a compact 20+ conversion model for stable workhorses from 2024 only.

The 25+ stable probability is not independently fit on a tiny rare-event sample;
it preserves M95F's conditional 25|20 ratio and scales it by the new 20+ probability.
M94C remains the central carry estimate. No sportsbook input or production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 95110
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]
C_GRID = (0.01, 0.03, 0.10, 0.30)

SPECS = {
    "tail_only": ["p20_logit"],
    "script_core": [
        "p20_logit", "candidate_team_rush_att", "pred_off_plays",
        "pred_lead_play_share", "pred_trail_play_share",
        "gs_team_neutral_rush_rate_avg3", "team_qb_rush_share_avg3",
    ],
    "script_matchup": [
        "p20_logit", "candidate_team_rush_att", "pred_off_plays",
        "pred_lead_play_share", "pred_trail_play_share",
        "gs_team_neutral_rush_rate_avg3", "team_qb_rush_share_avg3",
        "def_rb_20plus_carry_rate_allowed_avg3", "def_rb_carries_allowed_avg3",
        "def_rush_epa_allowed_avg3", "def_rush_success_allowed_avg3",
        "def_rush_ypa_allowed_avg3",
    ],
    "script_competition": [
        "p20_logit", "candidate_team_rush_att", "pred_off_plays",
        "pred_lead_play_share", "pred_trail_play_share",
        "gs_team_neutral_rush_rate_avg3", "team_qb_rush_share_avg3",
        "team_top1_share_avg3", "team_rb_used_avg3",
        "share_trend_1v5", "carry_trend_1v5",
        "self_inj_questionable", "self_practice_dnp", "self_practice_limited",
        "depth_rank", "depth_promotion",
    ],
    "script_full": [
        "p20_logit", "candidate_team_rush_att", "pred_off_plays",
        "pred_lead_play_share", "pred_neutral_play_share", "pred_trail_play_share",
        "gs_team_neutral_rush_rate_avg3", "gs_team_lead_rush_rate_avg3",
        "gs_team_trail_rush_rate_avg3", "team_qb_rush_share_avg3",
        "def_rb_20plus_carry_rate_allowed_avg3", "def_rb_carries_allowed_avg3",
        "def_rush_epa_allowed_avg3", "def_rush_success_allowed_avg3",
        "def_rush_ypa_allowed_avg3", "team_top1_share_avg3", "team_rb_used_avg3",
        "share_trend_1v5", "carry_trend_1v5",
        "self_inj_questionable", "self_practice_dnp", "self_practice_limited",
        "depth_rank", "depth_promotion", "opp_success_rate_off_avg3",
        "team_success_rate_off_avg3", "home", "pred_mean_margin", "pred_final_margin",
    ],
}


def num(s): return pd.to_numeric(s, errors="coerce")

def lower(df):
    x = df.copy(); x.columns = [str(c).lower() for c in x.columns]; return x

def find_one(root, name):
    hits = list(root.rglob(name))
    if len(hits) != 1: raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return hits[0]

def pipe(c):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=3000, random_state=SEED)),
    ])

def prob_metrics(y, p):
    z = pd.DataFrame({"y": num(y), "p": num(pd.Series(p, index=getattr(y, "index", None)))}).dropna()
    yy = z.y.astype(int); pp = z.p.clip(1e-6, 1 - 1e-6)
    return {
        "n": len(z), "base_rate": float(yy.mean()), "mean_prob": float(pp.mean()),
        "auc": float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan,
        "brier": float(np.mean((pp - yy) ** 2)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
    }

def stable_workhorse(z):
    trend = num(z.rb_rb_share_avg1) - num(z.rb_rb_share_avg5)
    return (
        num(z.role_is_workhorse).fillna(0).eq(1)
        & num(z.prior_top1_unavailable).fillna(0).eq(0)
        & num(z.target_was_prior_top1).fillna(0).eq(1)
        & trend.ge(-0.10)
        & num(z.self_inj_out).fillna(0).eq(0)
        & num(z.self_inj_doubtful).fillna(0).eq(0)
    )

def prep(g, team):
    z = lower(g); team = lower(team)
    wanted = [
        "pred_off_plays", "pred_lead_play_share", "pred_neutral_play_share", "pred_trail_play_share",
        "pred_mean_margin", "pred_final_margin", "gs_team_neutral_rush_rate_avg3",
        "gs_team_lead_rush_rate_avg3", "gs_team_trail_rush_rate_avg3",
        "opp_success_rate_off_avg3", "team_success_rate_off_avg3",
    ]
    add = TEAM_KEYS + [c for c in wanted if c in team.columns and c not in z.columns]
    z = z.merge(team[add].drop_duplicates(TEAM_KEYS), on=TEAM_KEYS, how="left", validate="many_to_one")
    z["actual_carries"] = num(z.actual_rush_att)
    z["share_trend_1v5"] = num(z.rb_rb_share_avg1) - num(z.rb_rb_share_avg5)
    z["carry_trend_1v5"] = num(z.rb_carries_avg1) - num(z.rb_carries_avg5)
    z["p20_base"] = num(z.cal_prob_20).clip(1e-5, 1 - 1e-5)
    z["p25_base"] = num(z.cal_prob_25).clip(1e-5, 1 - 1e-5)
    z["p20_logit"] = np.log(z.p20_base / (1 - z.p20_base))
    z["stable_workhorse_m95j"] = stable_workhorse(z).astype(int)
    z["vacancy_m95j"] = num(z.prior_top1_unavailable).fillna(0).eq(1).astype(int)
    return z

def usable(df, feats):
    out = [c for c in feats if c in df.columns and num(df[c]).notna().sum() >= 10 and num(df[c]).nunique(dropna=True) > 1]
    if not out: raise RuntimeError("no usable M95J features")
    return out

def select_model(dev, sel):
    yd = num(dev.actual_carries).ge(20).astype(int); ys = num(sel.actual_carries).ge(20).astype(int)
    base = prob_metrics(ys, sel.p20_base); rows = []
    for spec, fs0 in SPECS.items():
        fs = usable(dev, fs0)
        for c in C_GRID:
            m = pipe(c); m.fit(dev[fs], yd); p = m.predict_proba(sel[fs])[:, 1]
            met = prob_metrics(ys, p)
            eligible = int(met["brier"] <= base["brier"] and (not np.isfinite(base["auc"]) or met["auc"] >= base["auc"] - 0.02))
            rows.append({"spec": spec, "C": c, "feature_count": len(fs), **met,
                         "baseline_auc": base["auc"], "baseline_brier": base["brier"], "baseline_logloss": base["logloss"],
                         "auc_gain": met["auc"] - base["auc"], "brier_gain": base["brier"] - met["brier"],
                         "logloss_gain": base["logloss"] - met["logloss"], "eligible": eligible})
    grid = pd.DataFrame(rows); pool = grid.loc[grid.eligible.eq(1)].copy()
    if pool.empty: pool = grid.copy()
    chosen = pool.sort_values(["brier", "auc", "logloss"], ascending=[True, False, True]).iloc[0].to_dict()
    return grid, chosen

def probability_table(z):
    rows = []
    masks = {
        "all": pd.Series(True, index=z.index),
        "stable_workhorse": z.stable_workhorse_m95j.eq(1),
        "vacancy": z.vacancy_m95j.eq(1),
        "other": ~z.stable_workhorse_m95j.eq(1) & ~z.vacancy_m95j.eq(1),
    }
    for target, bcol, ncol, th in [("actual_20plus", "p20_base", "p20_m95j", 20), ("actual_25plus", "p25_base", "p25_m95j", 25)]:
        truth = num(z.actual_carries).ge(th).astype(int)
        for sl, mask in masks.items():
            for model, col in [("m95f", bcol), ("m95j_regime", ncol)]:
                rows.append({"scope": "2025_untouched_validation", "target": target, "slice": sl, "model": model,
                             **prob_metrics(truth.loc[mask], z.loc[mask, col])})
    return pd.DataFrame(rows)

def carry_table(z):
    a = num(z.actual_carries); rows = []
    masks = {"all_rb": pd.Series(True, index=z.index), "actual_0_5": a.between(0,5), "actual_6_10": a.between(6,10),
             "actual_11_14": a.between(11,14), "actual_15_plus": a.ge(15), "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25)}
    for sl, mask in masks.items():
        q = z.loc[mask]; e = num(q.m94c_rush_att) - num(q.actual_carries)
        rows.append({"scope": "2025_untouched_validation", "slice": sl, "n": len(q),
                     "m94c_mae": float(np.abs(e).mean()), "m95j_mae": float(np.abs(e).mean()), "mae_gain": 0.0,
                     "m94c_bias": float(e.mean()), "m95j_bias": float(e.mean())})
    return pd.DataFrame(rows)

def frequency_25(z, team):
    hit = z.loc[num(z.actual_carries).ge(25), PLAYER_KEYS].copy()
    tm = lower(team)[TEAM_KEYS + ["opponent"]].drop_duplicates(TEAM_KEYS)
    h = hit.merge(tm, on=TEAM_KEYS, how="left", validate="many_to_one")
    h["game_key"] = h.apply(lambda r: f"{int(r.season)}-{int(r.week)}-{'-'.join(sorted([str(r.team), str(r.opponent)]))}", axis=1)
    tg = len(tm); games = tg // 2; events = len(hit); unique_games = h.game_key.nunique()
    return pd.DataFrame([{"season": 2025, "rb_player_games": len(z), "rb_25plus_events": events,
                          "rb_player_game_rate": events / len(z), "team_games": tg, "team_game_25plus_rate": events / tg,
                          "nfl_games": games, "nfl_games_with_25plus_rb": unique_games,
                          "nfl_game_rate": unique_games / games, "games_per_25plus_event": games / max(unique_games,1)}])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m95g-root", type=Path, required=True); ap.add_argument("--m95i-root", type=Path, required=True)
    ap.add_argument("--m94c-root", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    g24 = pd.read_csv(find_one(args.m95g_root, "m95g_2024_holdout_trace.csv"), low_memory=False)
    g25 = pd.read_csv(find_one(args.m95g_root, "m95g_2025_rb_trace.csv"), low_memory=False)
    i25 = lower(pd.read_csv(find_one(args.m95i_root, "m95i_2025_trace.csv"), low_memory=False))
    t24 = pd.read_csv(find_one(args.m94c_root, "m94c_2024_holdout_trace.csv"), low_memory=False)
    t25 = pd.read_csv(find_one(args.m94c_root, "m94c_2025_team_trace.csv"), low_memory=False)
    d24 = prep(g24, t24); d25 = prep(g25, t25)
    d25 = d25.merge(i25[PLAYER_KEYS + ["p20_joint", "p25_joint"]].drop_duplicates(PLAYER_KEYS), on=PLAYER_KEYS, how="left", validate="one_to_one")
    dev = d24.loc[d24.stable_workhorse_m95j.eq(1) & num(d24.week).between(13,15)].copy()
    sel = d24.loc[d24.stable_workhorse_m95j.eq(1) & num(d24.week).between(16,18)].copy()
    full = d24.loc[d24.stable_workhorse_m95j.eq(1) & num(d24.week).between(13,18)].copy()
    grid, chosen = select_model(dev, sel); feats = usable(full, SPECS[str(chosen["spec"])])
    model = pipe(float(chosen["C"])); model.fit(full[feats], num(full.actual_carries).ge(20).astype(int))
    stable = d25.stable_workhorse_m95j.eq(1); vacancy = d25.vacancy_m95j.eq(1)
    d25["p20_stable_model"] = np.nan; d25.loc[stable, "p20_stable_model"] = model.predict_proba(d25.loc[stable, feats])[:,1]
    ratio = (num(d25.loc[stable, "p20_stable_model"]) / num(d25.loc[stable, "p20_base"])).clip(0.10, 10.0)
    d25["p25_stable_model"] = np.nan; d25.loc[stable, "p25_stable_model"] = np.minimum(num(d25.loc[stable,"p25_base"]) * ratio, num(d25.loc[stable,"p20_stable_model"]))
    d25["p20_m95j"] = num(d25.p20_base); d25["p25_m95j"] = num(d25.p25_base)
    d25.loc[vacancy,"p20_m95j"] = num(d25.loc[vacancy,"p20_joint"]); d25.loc[vacancy,"p25_m95j"] = num(d25.loc[vacancy,"p25_joint"])
    d25.loc[stable,"p20_m95j"] = num(d25.loc[stable,"p20_stable_model"]); d25.loc[stable,"p25_m95j"] = num(d25.loc[stable,"p25_stable_model"])
    d25["p25_m95j"] = np.minimum(num(d25.p25_m95j), num(d25.p20_m95j)); d25["m95j_rush_att"] = num(d25.m94c_rush_att)
    pm = probability_table(d25); ct = carry_table(d25); freq = frequency_25(d25, t25)
    def r(t,s,m): return pm.loc[pm.target.eq(t)&pm.slice.eq(s)&pm.model.eq(m)].iloc[0]
    s20b,s20n=r("actual_20plus","stable_workhorse","m95f"),r("actual_20plus","stable_workhorse","m95j_regime")
    s25b,s25n=r("actual_25plus","stable_workhorse","m95f"),r("actual_25plus","stable_workhorse","m95j_regime")
    v20b,v20n=r("actual_20plus","vacancy","m95f"),r("actual_20plus","vacancy","m95j_regime")
    v25b,v25n=r("actual_25plus","vacancy","m95f"),r("actual_25plus","vacancy","m95j_regime")
    a20b,a20n=r("actual_20plus","all","m95f"),r("actual_20plus","all","m95j_regime")
    a25b,a25n=r("actual_25plus","all","m95f"),r("actual_25plus","all","m95j_regime")
    stable20_pass=int(s20n.brier<s20b.brier and s20n.logloss<s20b.logloss and s20n.auc>=s20b.auc-.005)
    stable25_pass=int(s25n.brier<s25b.brier and s25n.auc>=s25b.auc-.005)
    vacancy20=int(v20n.brier<=v20b.brier); vacancy25=int(v25n.brier<=v25b.brier and v25n.auc>=v25b.auc)
    all20=int(a20n.brier<a20b.brier and a20n.auc>=a20b.auc-.002); all25=int(a25n.brier<=a25b.brier and a25n.auc>=a25b.auc-.002)
    valid=int(stable20_pass and stable25_pass and vacancy20 and vacancy25 and all20 and all25)
    disp="ADVANCE_M95J_REGIME_CONVERSION_TO_INTEGRATION_NOT_PRODUCTION" if valid else "RETAIN_M95J_AS_DIAGNOSTIC_DO_NOT_PROMOTE"
    selout=pd.DataFrame([{"stable_spec":chosen["spec"],"stable_C":chosen["C"],"selection_auc":chosen["auc"],"selection_brier":chosen["brier"],
                          "baseline_selection_auc":chosen["baseline_auc"],"baseline_selection_brier":chosen["baseline_brier"],
                          "stable_features":"|".join(feats),"vacancy_branch":"frozen_m95i_joint","other_branch":"frozen_m95f",
                          "stable_25_method":"preserve_m95f_conditional_25_given_20_ratio","central_carries":"m94c_preserved"}])
    dout=pd.DataFrame([{"stable20_pass":stable20_pass,"stable25_pass":stable25_pass,"vacancy20_preserved":vacancy20,"vacancy25_preserved":vacancy25,
                        "all20_pass":all20,"all25_pass":all25,"validation_pass":valid,
                        "stable20_auc_gain":s20n.auc-s20b.auc,"stable20_brier_gain":s20b.brier-s20n.brier,
                        "stable25_auc_gain":s25n.auc-s25b.auc,"stable25_brier_gain":s25b.brier-s25n.brier,
                        "vacancy25_auc_gain":v25n.auc-v25b.auc,"vacancy25_brier_gain":v25b.brier-v25n.brier,
                        "all20_auc_gain":a20n.auc-a20b.auc,"all20_brier_gain":a20b.brier-a20n.brier,
                        "all25_auc_gain":a25n.auc-a25b.auc,"all25_brier_gain":a25b.brier-a25n.brier,
                        "m94c_central_reference_preserved":1,"sportsbook_inputs":0,"production_change":0,"disposition":disp}])
    grid.to_csv(args.out_dir/"m95j_2024_stable_candidate_grid.csv",index=False); selout.to_csv(args.out_dir/"m95j_selected_architecture.csv",index=False)
    pm.to_csv(args.out_dir/"m95j_2025_probability_metrics.csv",index=False); ct.to_csv(args.out_dir/"m95j_2025_carry_metrics.csv",index=False)
    freq.to_csv(args.out_dir/"m95j_25plus_frequency.csv",index=False); dout.to_csv(args.out_dir/"m95j_disposition.csv",index=False)
    d25[PLAYER_KEYS+["actual_carries","m94c_rush_att","stable_workhorse_m95j","vacancy_m95j","p20_base","p20_m95j","p25_base","p25_m95j"]].to_csv(args.out_dir/"m95j_2025_trace.csv",index=False)
    print("[m95j] 25+ frequency\n",freq.to_string(index=False)); print("\n[m95j] selected\n",selout.to_string(index=False)); print("\n[m95j] probabilities\n",pm.to_string(index=False)); print("\n[m95j] disposition\n",dout.to_string(index=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
