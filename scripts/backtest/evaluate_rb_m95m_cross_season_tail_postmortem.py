"""M95M: descriptive cross-season postmortem of the M95K sealed-confirmation failure.

Research-only diagnostic. M95L opened the sealed 2023 W13-18 confirmation and the
frozen M95K stable-workhorse reranker failed. M95M must not tune, refit, select or
promote a replacement. It compares the already-produced 2023 and 2025 traces to
identify whether the failure is driven by cross-season nonstationarity, late-season
window effects, signal-direction reversals, sample-depth dependence, population
composition, or a small set of player-week reranking mistakes.

No sportsbook inputs. No production change. No new model coefficients.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
FEED_SIGNALS = [
    "feed20_rate", "feed25_rate", "carry_ceiling90", "carry_ceiling95", "sample_depth",
    "player_career_rate20_eb", "player_career_rate25_eb", "player_career_q90_eb", "player_career_q95_eb",
    "player_season_rate20_eb", "player_season_rate25_eb", "player_season_q90_eb", "player_season_q95_eb",
    "team_lead_all_rate20_eb", "team_lead_all_rate25_eb", "team_lead_all_q90_eb", "team_lead_all_q95_eb",
    "team_lead_season_rate20_eb", "team_lead_season_rate25_eb", "team_lead_season_q90_eb", "team_lead_season_q95_eb",
]
ENV_SIGNALS = [
    "candidate_team_rush_att", "pred_off_plays", "pred_lead_play_share", "pred_trail_play_share",
    "gs_team_neutral_rush_rate_avg3", "gs_team_lead_rush_rate_avg3", "team_qb_rush_share_avg3",
    "team_top1_share_avg3", "team_rb_used_avg3", "m94c_rush_att",
]


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c).lower() for c in x.columns]; return x


def num(s):
    return pd.to_numeric(s, errors="coerce")


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
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
        "logloss": float(log_loss(yy, pp, labels=[0, 1])), "positive_events": int(yy.sum()),
    }


def standardize_trace(df: pd.DataFrame, season: int, candidate_suffix: str) -> pd.DataFrame:
    x = lower(df)
    required = [*PLAYER_KEYS, "actual_carries", "stable_workhorse_m95k", "p20_base", "p25_base"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise RuntimeError(f"trace missing required columns: {missing}")
    c20 = f"p20_{candidate_suffix}"; c25 = f"p25_{candidate_suffix}"
    if c20 not in x.columns or c25 not in x.columns:
        raise RuntimeError(f"trace missing candidate probability columns {c20}/{c25}")
    x = x.loc[num(x["season"]).eq(season) & num(x["stable_workhorse_m95k"]).eq(1)].copy()
    x["candidate_p20"] = num(x[c20]); x["candidate_p25"] = num(x[c25])
    x["p20_base"] = num(x["p20_base"]); x["p25_base"] = num(x["p25_base"])
    x["actual_carries"] = num(x["actual_carries"])
    x["actual_20plus"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus"] = x["actual_carries"].ge(25).astype(int)
    x["delta20"] = x["candidate_p20"] - x["p20_base"]
    x["delta25"] = x["candidate_p25"] - x["p25_base"]
    return x


def scope_frames(x23: pd.DataFrame, x25: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "2023_w13_18_sealed": x23.loc[num(x23["week"]).between(13, 18)].copy(),
        "2025_full_research": x25.copy(),
        "2025_w13_18_same_window": x25.loc[num(x25["week"]).between(13, 18)].copy(),
    }


def population_table(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scope, z in scopes.items():
        row = {
            "scope": scope, "n": int(len(z)), "unique_players": int(z["player_clean_key"].nunique()),
            "unique_teams": int(z["team"].nunique()), "week_min": int(num(z["week"]).min()) if len(z) else np.nan,
            "week_max": int(num(z["week"]).max()) if len(z) else np.nan,
            "mean_actual_carries": float(num(z["actual_carries"]).mean()) if len(z) else np.nan,
        }
        for th in (20, 25):
            y = z[f"actual_{th}plus"]
            b = prob_metrics(y, z[f"p{th}_base"]); c = prob_metrics(y, z[f"candidate_p{th}"])
            for key, val in b.items(): row[f"base_{th}_{key}"] = val
            for key, val in c.items(): row[f"cand_{th}_{key}"] = val
            row[f"gain_{th}_auc"] = c["auc"] - b["auc"] if np.isfinite(c["auc"]) and np.isfinite(b["auc"]) else np.nan
            row[f"gain_{th}_brier"] = b["brier"] - c["brier"]
        rows.append(row)
    return pd.DataFrame(rows)


def signal_table(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    all_signals = FEED_SIGNALS + ENV_SIGNALS
    for scope, z in scopes.items():
        for sig in all_signals:
            if sig not in z.columns: continue
            v = num(z[sig]); carries = num(z["actual_carries"]); okc = v.notna() & carries.notna()
            rho = np.nan
            if okc.sum() >= 8 and v.loc[okc].nunique() > 1 and carries.loc[okc].nunique() > 1:
                rho = float(spearmanr(v.loc[okc], carries.loc[okc]).statistic)
            for th in (20, 25):
                y = z[f"actual_{th}plus"]; ok = v.notna() & y.notna()
                auc = np.nan
                if ok.sum() >= 8 and y.loc[ok].nunique() > 1 and v.loc[ok].nunique() > 1:
                    auc = float(roc_auc_score(y.loc[ok].astype(int), v.loc[ok]))
                rows.append({
                    "scope": scope, "signal_family": "feed_ceiling" if sig in FEED_SIGNALS else "environment",
                    "signal": sig, "target": f"actual_{th}plus", "n": int(ok.sum()), "auc": auc,
                    "auc_vs_random": auc - 0.5 if np.isfinite(auc) else np.nan, "spearman_actual_carries": rho,
                })
    out = pd.DataFrame(rows)
    if out.empty: return out
    wide = out.pivot_table(index=["signal_family", "signal", "target"], columns="scope", values="auc", aggfunc="first").reset_index()
    if "2023_w13_18_sealed" in wide and "2025_full_research" in wide:
        wide["auc_change_2023_minus_2025full"] = wide["2023_w13_18_sealed"] - wide["2025_full_research"]
    if "2023_w13_18_sealed" in wide and "2025_w13_18_same_window" in wide:
        wide["auc_change_2023_minus_2025samewindow"] = wide["2023_w13_18_sealed"] - wide["2025_w13_18_same_window"]
    return out.merge(wide, on=["signal_family", "signal", "target"], how="left", suffixes=("", "_cross"))


def rerank_cases(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    keep_signals = [c for c in ["feed20_rate", "feed25_rate", "carry_ceiling90", "carry_ceiling95", "sample_depth", "player_season_q95_eb", "team_lead_season_q95_eb", "candidate_team_rush_att", "m94c_rush_att"]]
    for scope, z in scopes.items():
        for th in (20, 25):
            q = z.copy(); y = q[f"actual_{th}plus"].astype(int); d = q[f"delta{th}"]
            q["rerank_help_score"] = np.where(y.eq(1), d, -d)
            q["target"] = f"actual_{th}plus"; q["scope"] = scope
            q["baseline_prob"] = q[f"p{th}_base"]; q["candidate_prob"] = q[f"candidate_p{th}"]; q["prob_delta"] = d
            q["actual_event"] = y
            cols = ["scope", "target", "season", "week", "team", "player_clean_key", "actual_carries", "actual_event", "baseline_prob", "candidate_prob", "prob_delta", "rerank_help_score"] + [c for c in keep_signals if c in q.columns]
            harmful = q.nsmallest(min(20, len(q)), "rerank_help_score")[cols].copy(); harmful["case_type"] = "most_harmful"
            helpful = q.nlargest(min(20, len(q)), "rerank_help_score")[cols].copy(); helpful["case_type"] = "most_helpful"
            rows.extend([harmful, helpful])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def depth_slices(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scope, z0 in scopes.items():
        if "sample_depth" not in z0.columns: continue
        z = z0.copy(); sd = num(z["sample_depth"])
        try:
            z["depth_bin"] = pd.qcut(sd.rank(method="first"), q=3, labels=["low", "mid", "high"])
        except Exception:
            continue
        for depth_bin, q in z.groupby("depth_bin", observed=True):
            for th in (20, 25):
                b = prob_metrics(q[f"actual_{th}plus"], q[f"p{th}_base"]); c = prob_metrics(q[f"actual_{th}plus"], q[f"candidate_p{th}"])
                rows.append({
                    "scope": scope, "depth_bin": str(depth_bin), "target": f"actual_{th}plus", "n": int(len(q)),
                    "base_rate": float(q[f"actual_{th}plus"].mean()), "sample_depth_mean": float(num(q["sample_depth"]).mean()),
                    "base_auc": b["auc"], "candidate_auc": c["auc"],
                    "auc_gain": c["auc"] - b["auc"] if np.isfinite(c["auc"]) and np.isfinite(b["auc"]) else np.nan,
                    "base_brier": b["brier"], "candidate_brier": c["brier"], "brier_gain": b["brier"] - c["brier"],
                })
    return pd.DataFrame(rows)


def player_concentration(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scope, z in scopes.items():
        agg = z.groupby(["player_clean_key"], as_index=False).agg(
            team=("team", "last"), games=("week", "size"), mean_carries=("actual_carries", "mean"),
            events20=("actual_20plus", "sum"), events25=("actual_25plus", "sum"),
            rate20=("actual_20plus", "mean"), rate25=("actual_25plus", "mean"),
            mean_delta20=("delta20", "mean"), mean_delta25=("delta25", "mean"),
        )
        agg["scope"] = scope
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def overlap_table(x23: pd.DataFrame, x25: pd.DataFrame) -> pd.DataFrame:
    common = sorted(set(x23["player_clean_key"].dropna()) & set(x25["player_clean_key"].dropna()))
    rows = []
    for p in common:
        a = x23.loc[x23["player_clean_key"].eq(p)]; b = x25.loc[x25["player_clean_key"].eq(p)]
        row = {
            "player_clean_key": p, "games_2023": len(a), "games_2025": len(b),
            "rate20_2023": float(a["actual_20plus"].mean()), "rate20_2025": float(b["actual_20plus"].mean()),
            "rate25_2023": float(a["actual_25plus"].mean()), "rate25_2025": float(b["actual_25plus"].mean()),
            "mean_carries_2023": float(num(a["actual_carries"]).mean()), "mean_carries_2025": float(num(b["actual_carries"]).mean()),
            "mean_delta20_2023": float(num(a["delta20"]).mean()), "mean_delta20_2025": float(num(b["delta20"]).mean()),
            "mean_delta25_2023": float(num(a["delta25"]).mean()), "mean_delta25_2025": float(num(b["delta25"]).mean()),
        }
        for sig in ["carry_ceiling95", "player_season_q95_eb", "team_lead_season_q95_eb", "sample_depth"]:
            if sig in a.columns and sig in b.columns:
                row[f"{sig}_2023"] = float(num(a[sig]).mean()); row[f"{sig}_2025"] = float(num(b[sig]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def disposition(pop: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    def prow(scope):
        q = pop.loc[pop["scope"].eq(scope)]
        return q.iloc[0] if len(q) else None
    p23 = prow("2023_w13_18_sealed"); p25full = prow("2025_full_research"); p25late = prow("2025_w13_18_same_window")
    g23_20 = float(p23["gain_20_auc"]); g23_25 = float(p23["gain_25_auc"])
    g25f_20 = float(p25full["gain_20_auc"]); g25f_25 = float(p25full["gain_25_auc"])
    g25l_20 = float(p25late["gain_20_auc"]) if p25late is not None else np.nan
    g25l_25 = float(p25late["gain_25_auc"]) if p25late is not None else np.nan
    if np.isfinite(g25l_20) and g25l_20 > 0 and g23_20 < 0:
        pattern = "cross_season_nonstationarity_same_window"
    elif g25f_20 > 0 and np.isfinite(g25l_20) and g25l_20 <= 0 and g23_20 < 0:
        pattern = "late_season_or_window_concentration"
    else:
        pattern = "mixed_nonstationarity_requires_case_review"
    flips = 0; strong25 = 0
    if not sig.empty:
        u = sig.loc[(sig["signal_family"].eq("feed_ceiling")) & (sig["target"].eq("actual_25plus"))].drop_duplicates(["signal", "target"])
        if "2025_full_research" in u.columns and "2023_w13_18_sealed" in u.columns:
            flips = int(((num(u["2025_full_research"]) > 0.55) & (num(u["2023_w13_18_sealed"]) < 0.50)).sum())
            strong25 = int((num(u["2025_full_research"]) > 0.60).sum())
    return pd.DataFrame([{
        "m95k_sealed_status": "failed_in_m95l", "m95m_role": "postmortem_only_no_model_change",
        "primary_pattern": pattern, "m95k_2025_full_auc_gain20": g25f_20, "m95k_2025_full_auc_gain25": g25f_25,
        "m95k_2025_w13_18_auc_gain20": g25l_20, "m95k_2025_w13_18_auc_gain25": g25l_25,
        "m95k_2023_w13_18_auc_gain20": g23_20, "m95k_2023_w13_18_auc_gain25": g23_25,
        "feed_signals_strong25_in_2025": strong25, "feed_signals_2025strong_but_2023_inverse": flips,
        "feature_search": 0, "coefficient_search": 0, "new_model_fit": 0, "sportsbook_inputs": 0, "production_change": 0,
        "next_rule": "use postmortem evidence to define a new hypothesis; do not retune M95K on opened 2023 confirmation labels",
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95k-root", type=Path, required=True)
    ap.add_argument("--m95l-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    k25 = pd.read_csv(find_one(args.m95k_root, "m95k_2025_trace.csv"), low_memory=False)
    l23 = pd.read_csv(find_one(args.m95l_root, "m95l_2023_confirmation_trace.csv"), low_memory=False)
    x25 = standardize_trace(k25, 2025, "m95k")
    x23 = standardize_trace(l23, 2023, "m95l")
    scopes = scope_frames(x23, x25)

    pop = population_table(scopes)
    sig = signal_table(scopes)
    cases = rerank_cases(scopes)
    depth = depth_slices(scopes)
    players = player_concentration(scopes)
    overlap = overlap_table(scopes["2023_w13_18_sealed"], scopes["2025_full_research"])
    disp = disposition(pop, sig)

    pop.to_csv(args.out_dir / "m95m_population_comparison.csv", index=False)
    sig.to_csv(args.out_dir / "m95m_signal_stability.csv", index=False)
    cases.to_csv(args.out_dir / "m95m_rerank_casebook.csv", index=False)
    depth.to_csv(args.out_dir / "m95m_sample_depth_slices.csv", index=False)
    players.to_csv(args.out_dir / "m95m_player_event_concentration.csv", index=False)
    overlap.to_csv(args.out_dir / "m95m_cross_season_player_overlap.csv", index=False)
    disp.to_csv(args.out_dir / "m95m_disposition.csv", index=False)

    print("[m95m] population comparison")
    print(pop.to_string(index=False))
    print("\n[m95m] disposition")
    print(disp.to_string(index=False))
    print("\n[m95m] strongest cross-season feed-signal changes")
    if not sig.empty and "auc_change_2023_minus_2025full" in sig.columns:
        q = sig.loc[sig["signal_family"].eq("feed_ceiling")].drop_duplicates(["signal", "target"]).copy()
        q["abs_change"] = num(q["auc_change_2023_minus_2025full"]).abs()
        print(q.nlargest(20, "abs_change")[["signal", "target", "2025_full_research", "2025_w13_18_same_window", "2023_w13_18_sealed", "auc_change_2023_minus_2025full"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
