"""M95K: stable-workhorse feed-tendency / carry-ceiling model.

Research-only. M95J established that vacancy/role-transition and stable-incumbent
workloads are different problems. M95K freezes the successful M95I vacancy branch
and tests whether leakage-safe player/team high-workload propensity can rerank
20+/25+ risk among already-established workhorses.

Key safeguard: M95K is a mass-preserving reranker. For stable workhorses it is
not allowed to increase the aggregate M95F 20+/25+ probability mass. It may only
redistribute that mass toward backs/weeks with stronger historical feed tendency,
carry ceiling, and current football environment. M94C remains the central carry
estimate. No sportsbook input and no production code changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 95111
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]
SHRINK_GRID = (4.0, 8.0, 12.0)
C_GRID = (0.01, 0.03, 0.10, 0.30)

COMPACT_BASE = ["p20_logit"]
COMPACT_FEED = [
    "p20_logit", "feed20_rate", "feed25_rate", "carry_ceiling90",
    "carry_ceiling95", "sample_depth",
]
CURRENT_FEED = [
    "p20_logit", "player_season_rate20_eb", "player_season_rate25_eb",
    "player_season_q90_eb", "player_season_q95_eb",
    "team_lead_season_rate20_eb", "team_lead_season_rate25_eb",
    "team_lead_season_q90_eb", "team_lead_season_q95_eb",
]
COMPACT_ENV = COMPACT_FEED + [
    "candidate_team_rush_att", "pred_off_plays", "pred_lead_play_share",
    "pred_trail_play_share", "gs_team_neutral_rush_rate_avg3",
    "gs_team_lead_rush_rate_avg3", "team_qb_rush_share_avg3",
    "team_top1_share_avg3", "team_rb_used_avg3",
]
SPECS = {
    "baseline_recalibration": COMPACT_BASE,
    "feed_compact": COMPACT_FEED,
    "feed_current": CURRENT_FEED,
    "feed_compact_env": COMPACT_ENV,
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def lower(df):
    x = df.copy()
    x.columns = [str(c).lower() for c in x.columns]
    return x


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return hits[0]


def pipe(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=3000, random_state=SEED)),
    ])


def prob_metrics(y, p) -> dict:
    z = pd.DataFrame({"y": num(y), "p": num(pd.Series(p, index=getattr(y, "index", None)))}).dropna()
    if z.empty:
        return {"n": 0, "base_rate": np.nan, "mean_prob": np.nan, "auc": np.nan, "brier": np.nan, "logloss": np.nan}
    yy = z.y.astype(int)
    pp = z.p.clip(1e-6, 1 - 1e-6)
    return {
        "n": int(len(z)), "base_rate": float(yy.mean()), "mean_prob": float(pp.mean()),
        "auc": float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan,
        "brier": float(np.mean((pp - yy) ** 2)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
    }


def stable_workhorse(z: pd.DataFrame) -> pd.Series:
    trend = num(z["rb_rb_share_avg1"]) - num(z["rb_rb_share_avg5"])
    return (
        num(z["role_is_workhorse"]).fillna(0).eq(1)
        & num(z["prior_top1_unavailable"]).fillna(0).eq(0)
        & num(z["target_was_prior_top1"]).fillna(0).eq(1)
        & trend.ge(-0.10)
        & num(z["self_inj_out"]).fillna(0).eq(0)
        & num(z["self_inj_doubtful"]).fillna(0).eq(0)
    )


def prep(g: pd.DataFrame, team: pd.DataFrame) -> pd.DataFrame:
    z = lower(g); team = lower(team)
    wanted = [
        "candidate_team_rush_att", "pred_off_plays", "pred_lead_play_share",
        "pred_neutral_play_share", "pred_trail_play_share", "pred_mean_margin",
        "pred_final_margin", "gs_team_neutral_rush_rate_avg3",
        "gs_team_lead_rush_rate_avg3", "gs_team_trail_rush_rate_avg3",
        "opp_success_rate_off_avg3", "team_success_rate_off_avg3",
    ]
    add = TEAM_KEYS + [c for c in wanted if c in team.columns and c not in z.columns]
    z = z.merge(team[add].drop_duplicates(TEAM_KEYS), on=TEAM_KEYS, how="left", validate="many_to_one")
    z["actual_carries"] = num(z["actual_carries"] if "actual_carries" in z.columns else z["actual_rush_att"])
    z["p20_base"] = num(z["cal_prob_20"]).clip(1e-5, 1 - 1e-5)
    z["p25_base"] = num(z["cal_prob_25"]).clip(1e-5, 1 - 1e-5)
    z["p20_logit"] = np.log(z["p20_base"] / (1 - z["p20_base"]))
    z["stable_workhorse_m95k"] = stable_workhorse(z).astype(int)
    z["vacancy_m95k"] = num(z["prior_top1_unavailable"]).fillna(0).eq(1).astype(int)
    return z


def quant(arr: list[float], q: float) -> float:
    return float(np.quantile(arr, q)) if arr else np.nan


def build_feed_features(history: pd.DataFrame, shrink: float) -> pd.DataFrame:
    """Create strictly pregame player/team feed priors by season-week batch."""
    h = lower(history)[PLAYER_KEYS + ["actual_carries"]].copy()
    h["actual_carries"] = num(h["actual_carries"])
    h = h.dropna(subset=["actual_carries"]).sort_values(PLAYER_KEYS).reset_index(drop=True)
    team_week = (
        h.sort_values(["season", "week", "team", "actual_carries"], ascending=[True, True, True, False])
        .groupby(TEAM_KEYS, as_index=False).first()[TEAM_KEYS + ["actual_carries"]]
        .rename(columns={"actual_carries": "lead_carries"})
    )
    hg = {(int(s), int(w)): g for (s, w), g in h.groupby(["season", "week"])}
    tg = {(int(s), int(w)): g for (s, w), g in team_week.groupby(["season", "week"])}
    player_all = defaultdict(list); player_season = defaultdict(list)
    team_all = defaultdict(list); team_season = defaultdict(list)
    league_leads = []; league_leads_season = defaultdict(list); rows = []

    for s, w in sorted(hg):
        current = hg[(s, w)]; lg = league_leads; lgs = league_leads_season[s]
        global20 = float(np.mean(np.asarray(lg) >= 20)) if lg else 0.15
        global25 = float(np.mean(np.asarray(lg) >= 25)) if lg else 0.04
        global90 = quant(lg, 0.90) if lg else 20.0; global95 = quant(lg, 0.95) if lg else 23.0
        season20 = float(np.mean(np.asarray(lgs) >= 20)) if lgs else global20
        season25 = float(np.mean(np.asarray(lgs) >= 25)) if lgs else global25
        season90 = quant(lgs, 0.90) if lgs else global90; season95 = quant(lgs, 0.95) if lgs else global95

        def stats(arr, pr20, pr25, pq90, pq95, prefix):
            n = len(arr); wt = n / (n + shrink)
            raw90 = quant(arr, 0.90) if arr else pq90; raw95 = quant(arr, 0.95) if arr else pq95
            return {
                f"{prefix}_n": n,
                f"{prefix}_rate20_eb": (sum(x >= 20 for x in arr) + shrink * pr20) / (n + shrink),
                f"{prefix}_rate25_eb": (sum(x >= 25 for x in arr) + shrink * pr25) / (n + shrink),
                f"{prefix}_q90_eb": wt * raw90 + (1 - wt) * pq90,
                f"{prefix}_q95_eb": wt * raw95 + (1 - wt) * pq95,
                f"{prefix}_max": max(arr) if arr else pq95,
            }

        for _, r in current.iterrows():
            pk = str(r["player_clean_key"]); team = str(r["team"])
            out = {"season": s, "week": w, "team": team, "player_clean_key": pk}
            out.update(stats(player_all[pk], global20, global25, global90, global95, "player_career"))
            out.update(stats(player_season[(s, pk)], season20, season25, season90, season95, "player_season"))
            out.update(stats(team_all[team], global20, global25, global90, global95, "team_lead_all"))
            out.update(stats(team_season[(s, team)], season20, season25, season90, season95, "team_lead_season"))
            rows.append(out)

        for _, r in current.iterrows():
            pk = str(r["player_clean_key"]); team = str(r["team"]); x = float(r["actual_carries"])
            player_all[pk].append(x); player_season[(s, pk)].append(x)
        for _, r in tg[(s, w)].iterrows():
            team = str(r["team"]); x = float(r["lead_carries"])
            team_all[team].append(x); team_season[(s, team)].append(x)
            league_leads.append(x); league_leads_season[s].append(x)
    return pd.DataFrame(rows)


def add_composites(z: pd.DataFrame) -> pd.DataFrame:
    x = z.copy()
    x["feed20_rate"] = x[["player_career_rate20_eb", "player_season_rate20_eb", "team_lead_all_rate20_eb", "team_lead_season_rate20_eb"]].mean(axis=1)
    x["feed25_rate"] = x[["player_career_rate25_eb", "player_season_rate25_eb", "team_lead_all_rate25_eb", "team_lead_season_rate25_eb"]].mean(axis=1)
    x["carry_ceiling90"] = x[["player_career_q90_eb", "player_season_q90_eb", "team_lead_all_q90_eb", "team_lead_season_q90_eb"]].mean(axis=1)
    x["carry_ceiling95"] = x[["player_career_q95_eb", "player_season_q95_eb", "team_lead_all_q95_eb", "team_lead_season_q95_eb"]].mean(axis=1)
    x["sample_depth"] = np.log1p(x[["player_career_n", "player_season_n", "team_lead_all_n", "team_lead_season_n"]].mean(axis=1))
    return x


def solve_mean_delta(p: np.ndarray, target_mean: float) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p)); lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2; meanp = float((1 / (1 + np.exp(-np.clip(lp + mid, -35, 35)))).mean())
        if meanp < target_mean: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def mean_anchor(p: np.ndarray, target_mean: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6); d = solve_mean_delta(p, target_mean)
    lp = np.log(p / (1 - p)) + d
    return np.clip(1 / (1 + np.exp(-np.clip(lp, -35, 35))), 1e-6, 1 - 1e-6)


def available(df: pd.DataFrame, fs: list[str]) -> list[str]:
    out = [c for c in fs if c in df.columns and num(df[c]).notna().sum() >= 10 and num(df[c]).nunique(dropna=True) > 1]
    if not out: raise RuntimeError("M95K candidate has no usable features")
    return out


def candidate_probs(model: Pipeline, x: pd.DataFrame, fs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw20 = model.predict_proba(x[fs])[:, 1]
    p20 = mean_anchor(raw20, float(num(x["p20_base"]).mean()))
    rel = np.clip(p20 / num(x["p20_base"]).to_numpy(dtype=float), 0.10, 10.0)
    raw25 = np.minimum(num(x["p25_base"]).to_numpy(dtype=float) * rel, p20)
    p25 = mean_anchor(raw25, float(num(x["p25_base"]).mean()))
    return p20, np.minimum(p25, p20)


def select_architecture(base24: pd.DataFrame, history: pd.DataFrame):
    sel0 = base24.loc[num(base24["week"]).between(16, 18)].copy()
    base20 = prob_metrics(num(sel0["actual_carries"]).ge(20).astype(int), sel0["p20_base"])
    base25 = prob_metrics(num(sel0["actual_carries"]).ge(25).astype(int), sel0["p25_base"])
    rows = []; frames = {}
    for shrink in SHRINK_GRID:
        ff = build_feed_features(history, shrink)
        z = add_composites(base24.merge(ff, on=PLAYER_KEYS, how="left", validate="one_to_one")); frames[shrink] = z
        dev = z.loc[num(z["week"]).between(13, 15)].copy(); sel = z.loc[num(z["week"]).between(16, 18)].copy()
        ydev = num(dev["actual_carries"]).ge(20).astype(int); y20 = num(sel["actual_carries"]).ge(20).astype(int); y25 = num(sel["actual_carries"]).ge(25).astype(int)
        for spec, fs0 in SPECS.items():
            fs = available(dev, fs0)
            for c in C_GRID:
                model = pipe(c); model.fit(dev[fs], ydev); p20, p25 = candidate_probs(model, sel, fs)
                m20 = prob_metrics(y20, p20); m25 = prob_metrics(y25, p25)
                eligible = int(m20["brier"] <= base20["brier"] and m20["auc"] >= base20["auc"] and m25["brier"] <= base25["brier"] and m25["auc"] >= base25["auc"] - 0.02)
                rows.append({
                    "shrink_k": shrink, "spec": spec, "C": c, "feature_count": len(fs),
                    "auc20": m20["auc"], "brier20": m20["brier"], "logloss20": m20["logloss"],
                    "auc25": m25["auc"], "brier25": m25["brier"], "logloss25": m25["logloss"],
                    "baseline_auc20": base20["auc"], "baseline_brier20": base20["brier"],
                    "baseline_auc25": base25["auc"], "baseline_brier25": base25["brier"],
                    "auc20_gain": m20["auc"] - base20["auc"], "brier20_gain": base20["brier"] - m20["brier"],
                    "auc25_gain": m25["auc"] - base25["auc"], "brier25_gain": base25["brier"] - m25["brier"], "eligible": eligible,
                })
    grid = pd.DataFrame(rows); pool = grid.loc[grid["eligible"].eq(1)].copy()
    if pool.empty: pool = grid.copy()
    chosen = pool.sort_values(["brier20", "brier25", "auc20"], ascending=[True, True, False]).iloc[0].to_dict()
    return grid, chosen, frames[float(chosen["shrink_k"])]


def probability_table(z: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "all": pd.Series(True, index=z.index), "stable_workhorse": z["stable_workhorse_m95k"].eq(1),
        "vacancy": z["vacancy_m95k"].eq(1), "other": ~z["stable_workhorse_m95k"].eq(1) & ~z["vacancy_m95k"].eq(1),
    }
    rows = []
    for th in (20, 25):
        truth = num(z["actual_carries"]).ge(th).astype(int)
        for sl, mask in masks.items():
            for model, col in [("m95f", f"p{th}_base"), ("m95k_regime", f"p{th}_m95k")]:
                rows.append({"scope": "2025_validation_reused_not_pristine", "target": f"actual_{th}plus", "slice": sl, "model": model, **prob_metrics(truth.loc[mask], z.loc[mask, col])})
    return pd.DataFrame(rows)


def feed_signal_audit(z: pd.DataFrame) -> pd.DataFrame:
    s = z.loc[z["stable_workhorse_m95k"].eq(1)].copy(); rows = []
    for th in (20, 25):
        y = num(s["actual_carries"]).ge(th).astype(int)
        for c in ["feed20_rate", "feed25_rate", "carry_ceiling90", "carry_ceiling95", "player_season_rate20_eb", "player_season_rate25_eb", "player_season_q90_eb", "player_season_q95_eb", "team_lead_season_rate20_eb", "team_lead_season_rate25_eb", "team_lead_season_q90_eb", "team_lead_season_q95_eb"]:
            if c not in s.columns: continue
            v = num(s[c]); ok = y.notna() & v.notna()
            auc = float(roc_auc_score(y.loc[ok], v.loc[ok])) if ok.sum() > 5 and y.loc[ok].nunique() > 1 and v.loc[ok].nunique() > 1 else np.nan
            rows.append({"target": f"actual_{th}plus", "signal": c, "n": int(ok.sum()), "auc": auc})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--m95b-root", type=Path, required=True); ap.add_argument("--m95g-root", type=Path, required=True)
    ap.add_argument("--m95i-root", type=Path, required=True); ap.add_argument("--m94c-root", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(find_one(args.m95b_root, "m95b_rb_matchup_trace.csv"), low_memory=False)
    g24 = pd.read_csv(find_one(args.m95g_root, "m95g_2024_holdout_trace.csv"), low_memory=False); g25 = pd.read_csv(find_one(args.m95g_root, "m95g_2025_rb_trace.csv"), low_memory=False)
    i25 = lower(pd.read_csv(find_one(args.m95i_root, "m95i_2025_trace.csv"), low_memory=False))
    t24 = pd.read_csv(find_one(args.m94c_root, "m94c_2024_holdout_trace.csv"), low_memory=False); t25 = pd.read_csv(find_one(args.m94c_root, "m94c_2025_team_trace.csv"), low_memory=False)
    legacy = lower(pd.read_csv(find_one(args.m94c_root, "m94c_legacy_guard.csv"), low_memory=False))

    b24all = prep(g24, t24); b25all = prep(g25, t25)
    b24 = b24all.loc[b24all["stable_workhorse_m95k"].eq(1) & num(b24all["week"]).between(13, 18)].copy()
    grid, chosen, z24 = select_architecture(b24, history)
    shrink = float(chosen["shrink_k"]); spec = str(chosen["spec"]); c = float(chosen["C"]); fs = available(z24, SPECS[spec])
    model = pipe(c); model.fit(z24[fs], num(z24["actual_carries"]).ge(20).astype(int))

    ff = build_feed_features(history, shrink); z25 = add_composites(b25all.merge(ff, on=PLAYER_KEYS, how="left", validate="one_to_one"))
    z25 = z25.merge(i25[PLAYER_KEYS + ["p20_joint", "p25_joint"]].drop_duplicates(PLAYER_KEYS), on=PLAYER_KEYS, how="left", validate="one_to_one")
    stable = z25["stable_workhorse_m95k"].eq(1); vacancy = z25["vacancy_m95k"].eq(1)
    p20s, p25s = candidate_probs(model, z25.loc[stable].copy(), fs)
    z25["p20_m95k"] = num(z25["p20_base"]); z25["p25_m95k"] = num(z25["p25_base"])
    z25.loc[vacancy, "p20_m95k"] = num(z25.loc[vacancy, "p20_joint"]); z25.loc[vacancy, "p25_m95k"] = num(z25.loc[vacancy, "p25_joint"])
    z25.loc[stable, "p20_m95k"] = p20s; z25.loc[stable, "p25_m95k"] = p25s
    z25["p25_m95k"] = np.minimum(num(z25["p25_m95k"]), num(z25["p20_m95k"])); z25["m95k_rush_att"] = num(z25["m94c_rush_att"])

    pm = probability_table(z25); sig = feed_signal_audit(z25)
    def get(target, sl, model_name): return pm.loc[(pm.target.eq(target)) & (pm.slice.eq(sl)) & (pm.model.eq(model_name))].iloc[0]
    s20b = get("actual_20plus", "stable_workhorse", "m95f"); s20c = get("actual_20plus", "stable_workhorse", "m95k_regime")
    s25b = get("actual_25plus", "stable_workhorse", "m95f"); s25c = get("actual_25plus", "stable_workhorse", "m95k_regime")
    a20b = get("actual_20plus", "all", "m95f"); a20c = get("actual_20plus", "all", "m95k_regime")
    a25b = get("actual_25plus", "all", "m95f"); a25c = get("actual_25plus", "all", "m95k_regime")
    v25b = get("actual_25plus", "vacancy", "m95f"); v25c = get("actual_25plus", "vacancy", "m95k_regime")
    stable20_pass = int(s20c.auc >= s20b.auc + 0.02 and s20c.brier <= s20b.brier - 0.005)
    stable25_pass = int(s25c.auc >= s25b.auc + 0.01 and s25c.brier <= s25b.brier - 0.001)
    all20_pass = int(a20c.auc >= a20b.auc + 0.002 and a20c.brier <= a20b.brier - 0.001)
    all25_pass = int(a25c.auc >= a25b.auc + 0.002 and a25c.brier <= a25b.brier - 0.0001)
    vacancy25_preserved = int(v25c.auc >= 0.90 and v25c.brier <= v25b.brier)
    stable_mass_preserved = int(abs(float(s20c.mean_prob - s20b.mean_prob)) < 1e-9 and abs(float(s25c.mean_prob - s25b.mean_prob)) < 1e-9)
    scientific_pass = int(stable20_pass and stable25_pass and all20_pass and all25_pass and vacancy25_preserved and stable_mass_preserved)
    legacy_gain = float(num(legacy["mae_gain"]).iloc[0]) if "mae_gain" in legacy.columns else np.nan
    production_gate_pass = int(scientific_pass and np.isfinite(legacy_gain) and legacy_gain >= 0)
    disposition = "ADVANCE_M95K_TAIL_ARCHITECTURE_TO_SEALED_CONFIRMATION" if scientific_pass else "RETAIN_M95K_AS_DIAGNOSTIC_DO_NOT_PROMOTE"

    selected = pd.DataFrame([{"shrink_k": shrink, "spec": spec, "C": c, "features": "|".join(fs), "mass_preserving_rerank": 1, "stable_25_method": "m95f_conditional_ratio_plus_mass_anchor", "vacancy_branch": "frozen_m95i_joint", "other_branch": "frozen_m95f", "central_carries": "m94c_preserved"}])
    disp = pd.DataFrame([{
        "stable20_pass": stable20_pass, "stable25_pass": stable25_pass, "all20_pass": all20_pass, "all25_pass": all25_pass,
        "vacancy25_preserved": vacancy25_preserved, "stable_mass_preserved": stable_mass_preserved, "scientific_pass": scientific_pass,
        "stable20_auc_gain": float(s20c.auc - s20b.auc), "stable20_brier_gain": float(s20b.brier - s20c.brier),
        "stable25_auc_gain": float(s25c.auc - s25b.auc), "stable25_brier_gain": float(s25b.brier - s25c.brier),
        "all20_auc_gain": float(a20c.auc - a20b.auc), "all20_brier_gain": float(a20b.brier - a20c.brier),
        "all25_auc_gain": float(a25c.auc - a25b.auc), "all25_brier_gain": float(a25b.brier - a25c.brier),
        "legacy_guard_gain_inherited": legacy_gain, "production_gate_pass": production_gate_pass,
        "m94c_central_reference_preserved": 1, "sportsbook_inputs": 0, "production_change": 0,
        "validation_note": "2025_reused_research_validation_not_pristine_final_confirmation", "disposition": disposition,
    }])
    source = pd.DataFrame([{"history_rows": len(history), "stable_2024_rows": len(z24), "stable_2025_rows": int(stable.sum()), "feature_snapshot_rule": "strictly_prior_season_week_batch", "empirical_bayes_shrink_k": shrink}])
    grid.to_csv(args.out_dir / "m95k_2024_selection_grid.csv", index=False); selected.to_csv(args.out_dir / "m95k_selected_architecture.csv", index=False)
    pm.to_csv(args.out_dir / "m95k_2025_probability_metrics.csv", index=False); sig.to_csv(args.out_dir / "m95k_feed_signal_audit.csv", index=False)
    disp.to_csv(args.out_dir / "m95k_disposition.csv", index=False); source.to_csv(args.out_dir / "m95k_source_audit.csv", index=False); z25.to_csv(args.out_dir / "m95k_2025_trace.csv", index=False)
    print("[m95k] selected"); print(selected.to_string(index=False)); print("\n[m95k] probabilities"); print(pm.to_string(index=False)); print("\n[m95k] disposition"); print(disp.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
