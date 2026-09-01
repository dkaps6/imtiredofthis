#!/usr/bin/env python3
"""M95T: final constrained dual-layer stable-workhorse carry-tail candidate.

Population mass responds only to fast pregame league workload state. Player
reranking reuses M95K leakage-safe feed/carry-ceiling semantics and is conditional,
bounded, and exactly mass-preserving before the population adjustment.

Research only. M94C central carries are unchanged. No sportsbook inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

from scripts.player_form_v2 import _normalize_weekly
from scripts.backtest.evaluate_rb_feed_tendency_carry_ceiling import (
    add_composites,
    build_feed_features,
    mean_anchor,
)

KEYS = ["season", "week", "team", "player_clean_key"]
EPS = 1e-6
FEED_SHRINK_K = 4.0
RATIO_LO = 0.70
RATIO_HI = 1.30
MASS_SHRINK = 0.50
RANK_BLEND = 0.50
RANK_DELTA_CAP = 0.25


def num(s):
    return pd.to_numeric(s, errors="coerce")


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return hits[0]


def metrics(y, p):
    z = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if z.empty:
        return {"n": 0, "events": 0, "actual_rate": np.nan, "mean_prob": np.nan,
                "calibration_gap": np.nan, "abs_calibration_gap": np.nan,
                "auc": np.nan, "brier": np.nan, "logloss": np.nan}
    yy = z.y.astype(int)
    pp = z.p.clip(EPS, 1 - EPS)
    actual = float(yy.mean())
    meanp = float(pp.mean())
    return {
        "n": int(len(z)), "events": int(yy.sum()), "actual_rate": actual,
        "mean_prob": meanp, "calibration_gap": actual - meanp,
        "abs_calibration_gap": abs(actual - meanp),
        "auc": float(roc_auc_score(yy, pp)) if yy.nunique() > 1 else np.nan,
        "brier": float(np.mean((pp - yy) ** 2)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
    }


def auc_score(y, score):
    z = pd.DataFrame({"y": num(y), "s": num(score)}).dropna()
    if len(z) < 5 or z.y.nunique() < 2 or z.s.nunique() < 2:
        return np.nan
    return float(roc_auc_score(z.y.astype(int), z.s.astype(float)))


def load_exact_panel(rdir: Path) -> pd.DataFrame:
    p = pd.read_csv(find_one(rdir, "m95r_exact_panel.csv"), low_memory=False)
    p.columns = [str(c).lower() for c in p.columns]
    p["season"] = num(p["season"]).astype(int)
    p["week"] = num(p["week"]).astype(int)
    p["p20_base"] = num(p["p20_base"]).clip(EPS, 1 - EPS)
    p["actual_carries"] = num(p["actual_carries"])
    p["actual_20plus"] = num(p["actual_20plus"]).astype(int)
    if p.duplicated(KEYS).any():
        raise RuntimeError("duplicate rows in M95R exact panel")
    return p


def load_p25(qdir: Path, kdir: Path, ldir: Path) -> pd.DataFrame:
    rows = []
    q = pd.read_csv(find_one(qdir, "m95q_enriched_holdouts.csv"), low_memory=False)
    q.columns = [str(c).lower() for c in q.columns]
    q = q.loc[num(q["season"]).isin([2020, 2021, 2022, 2024]) & num(q["stable_workhorse_m95k"]).eq(1)].copy()
    q["p25_base"] = num(q["cal_prob_25"])
    rows.append(q[KEYS + ["p25_base"]])

    l = pd.read_csv(find_one(ldir, "m95l_2023_confirmation_trace.csv"), low_memory=False)
    l.columns = [str(c).lower() for c in l.columns]
    l = l.loc[num(l["stable_workhorse_m95k"]).eq(1)].copy()
    rows.append(l[KEYS + ["p25_base"]])

    k = pd.read_csv(find_one(kdir, "m95k_2025_trace.csv"), low_memory=False)
    k.columns = [str(c).lower() for c in k.columns]
    k = k.loc[num(k["stable_workhorse_m95k"]).eq(1)].copy()
    rows.append(k[KEYS + ["p25_base"]])

    out = pd.concat(rows, ignore_index=True, sort=False)
    out["season"] = num(out["season"]).astype(int)
    out["week"] = num(out["week"]).astype(int)
    out["p25_base"] = num(out["p25_base"]).clip(EPS, 1 - EPS)
    out = out.sort_values(KEYS).drop_duplicates(KEYS, keep="first")
    return out


def load_cached_rb_history(cache_dir: Path) -> pd.DataFrame:
    frames = []
    for season in range(2018, 2026):
        path = cache_dir / f"player_weekly_{season}.parquet"
        if not path.exists():
            raise RuntimeError(f"missing frozen player cache {path}")
        raw = pd.read_parquet(path)
        z = _normalize_weekly(raw, season)
        z = z.loc[z["position"].astype(str).str.upper().eq("RB")].copy()
        max_week = 17 if season <= 2020 else 18
        z = z.loc[num(z["week"]).between(1, max_week)].copy()
        z["actual_carries"] = num(z["rushes"])
        z["actual_rush_yards"] = num(z["rush_yards"])
        z = z[KEYS + ["actual_carries", "actual_rush_yards"]]
        if z.duplicated(KEYS).any():
            dup = z.loc[z.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
            raise RuntimeError(f"duplicate cached RB player-week rows: {dup}")
        frames.append(z)
    h = pd.concat(frames, ignore_index=True, sort=False).sort_values(KEYS).reset_index(drop=True)
    return h


def build_league_anchors(history: pd.DataFrame) -> pd.DataFrame:
    lead = (
        history.sort_values(KEYS + ["actual_carries"], ascending=[True, True, True, True, False])
        .groupby(["season", "week", "team"], as_index=False)
        .first()[["season", "week", "team", "actual_carries"]]
    )
    wk = lead.groupby(["season", "week"], as_index=False).agg(
        league_week_lead20=("actual_carries", lambda x: float((num(x) >= 20).mean())),
        teams=("team", "nunique"),
    )
    rows = []
    for season, g in wk.groupby("season"):
        g = g.sort_values("week").reset_index(drop=True)
        rates = []
        for _, r in g.iterrows():
            prior = rates.copy()
            rows.append({
                "season": int(season), "week": int(r.week),
                "league_prior4_20": float(np.mean(prior[-4:])) if prior else np.nan,
                "league_std_20": float(np.mean(prior)) if prior else np.nan,
                "league_week_lead20": float(r.league_week_lead20),
            })
            rates.append(float(r.league_week_lead20))
    return pd.DataFrame(rows)


def apply_candidate(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (season, week), g0 in panel.groupby(["season", "week"], sort=True):
        g = g0.copy()
        n = len(g)
        g["base_rank"] = g["p20_base"].rank(method="average", pct=True)
        r1 = num(g["feed20_rate"]).rank(method="average", pct=True)
        r2 = num(g["carry_ceiling95"]).rank(method="average", pct=True)
        g["feed_rank"] = pd.concat([r1, r2], axis=1).mean(axis=1)
        high_base = g["base_rank"].ge(0.50)
        high_feed = g["feed_rank"].ge(0.50)
        g["aligned"] = high_base.eq(high_feed).astype(int)
        g["rank_delta_raw"] = np.where(
            g["aligned"].eq(1),
            RANK_BLEND * (g["feed_rank"] - g["base_rank"]),
            0.0,
        )
        g["rank_delta"] = num(g["rank_delta_raw"]).clip(-RANK_DELTA_CAP, RANK_DELTA_CAP)
        p = g["p20_base"].to_numpy(float)
        lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
        raw_rank = 1 / (1 + np.exp(-np.clip(lp + g["rank_delta"].to_numpy(float), -35, 35)))
        base_mean = float(g["p20_base"].mean())
        rank_only = mean_anchor(raw_rank, base_mean)
        g["p20_rank_only"] = rank_only
        g["rank_mass_error"] = float(np.mean(rank_only) - base_mean)

        prior4 = float(num(g["league_prior4_20"]).dropna().iloc[0]) if num(g["league_prior4_20"]).notna().any() else np.nan
        std = float(num(g["league_std_20"]).dropna().iloc[0]) if num(g["league_std_20"]).notna().any() else np.nan
        if not np.isfinite(prior4) or not np.isfinite(std) or std <= 0:
            ratio = 1.0
        else:
            ratio = float(np.clip(prior4 / std, RATIO_LO, RATIO_HI))
        mass_factor = float(1.0 + MASS_SHRINK * (ratio - 1.0))
        target20 = float(np.clip(base_mean * mass_factor, 0.05, 0.70))
        g["regime_ratio"] = ratio
        g["mass_factor"] = mass_factor
        g["weekly_target20"] = target20
        g["p20_m95t"] = mean_anchor(rank_only, target20)

        if "p25_base" in g.columns and num(g["p25_base"]).notna().any():
            rel = np.clip(g["p20_m95t"].to_numpy(float) / np.clip(p, EPS, None), 0.70, 1.30)
            raw25 = np.minimum(num(g["p25_base"]).to_numpy(float) * rel, g["p20_m95t"].to_numpy(float))
            target25 = float(np.clip(num(g["p25_base"]).mean() * mass_factor, 0.001, target20))
            p25 = mean_anchor(raw25, target25)
            g["p25_m95t"] = np.minimum(p25, g["p20_m95t"].to_numpy(float))
        out.append(g)
    return pd.concat(out, ignore_index=True, sort=False).sort_values(KEYS).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95r-dir", type=Path, required=True)
    ap.add_argument("--m95q-dir", type=Path, required=True)
    ap.add_argument("--m95k-dir", type=Path, required=True)
    ap.add_argument("--m95l-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_exact_panel(args.m95r_dir)
    p25 = load_p25(args.m95q_dir, args.m95k_dir, args.m95l_dir)
    panel = panel.merge(p25, on=KEYS, how="left", validate="one_to_one")

    history = load_cached_rb_history(args.cache_dir)
    feed = add_composites(build_feed_features(history[KEYS + ["actual_carries"]], FEED_SHRINK_K))
    panel = panel.merge(feed, on=KEYS, how="left", validate="one_to_one")
    yards = history[KEYS + ["actual_rush_yards"]].drop_duplicates(KEYS)
    panel = panel.merge(yards, on=KEYS, how="left", validate="one_to_one")
    anchors = build_league_anchors(history)
    panel = panel.merge(anchors, on=["season", "week"], how="left", validate="many_to_one")

    for season, g in panel.groupby("season"):
        feed_cov = float(g[["feed20_rate", "carry_ceiling95"]].notna().all(axis=1).mean())
        yard_cov = float(g["actual_rush_yards"].notna().mean())
        if feed_cov < 0.98:
            raise RuntimeError(f"M95T feed join coverage below 98% season={season}: {feed_cov:.4%}")
        if yard_cov < 0.98:
            raise RuntimeError(f"M95T rush-yard truth coverage below 98% season={season}: {yard_cov:.4%}")

    primary = panel.loc[num(panel["week"]).between(13, 18)].copy()
    cand = apply_candidate(primary)
    cand["actual_75plus_rush_yards"] = num(cand["actual_rush_yards"]).ge(75).astype(int)
    cand["actual_100plus_rush_yards"] = num(cand["actual_rush_yards"]).ge(100).astype(int)
    cand.to_csv(args.out_dir / "m95t_trace.csv", index=False)

    rows = []
    for season, g in cand.groupby("season"):
        b = metrics(g["actual_20plus"], g["p20_base"])
        c = metrics(g["actual_20plus"], g["p20_m95t"])
        row = {"season": int(season), **{f"base_{k}": v for k, v in b.items()}, **{f"cand_{k}": v for k, v in c.items()}}
        row["brier_gain"] = b["brier"] - c["brier"]
        row["logloss_gain"] = b["logloss"] - c["logloss"]
        row["auc_gain"] = c["auc"] - b["auc"] if np.isfinite(b["auc"]) and np.isfinite(c["auc"]) else np.nan
        row["abs_gap_gain"] = b["abs_calibration_gap"] - c["abs_calibration_gap"]
        rows.append(row)
    b = metrics(cand["actual_20plus"], cand["p20_base"])
    c = metrics(cand["actual_20plus"], cand["p20_m95t"])
    pooled = {"season": "POOLED", **{f"base_{k}": v for k, v in b.items()}, **{f"cand_{k}": v for k, v in c.items()}}
    pooled["brier_gain"] = b["brier"] - c["brier"]
    pooled["logloss_gain"] = b["logloss"] - c["logloss"]
    pooled["auc_gain"] = c["auc"] - b["auc"]
    pooled["abs_gap_gain"] = b["abs_calibration_gap"] - c["abs_calibration_gap"]
    rows.append(pooled)
    mdf = pd.DataFrame(rows)
    mdf.to_csv(args.out_dir / "m95t_20plus_metrics.csv", index=False)

    p25rows = []
    if cand["p25_base"].notna().any():
        cand["actual_25plus"] = num(cand["actual_carries"]).ge(25).astype(int)
        for season, g in cand.groupby("season"):
            b25 = metrics(g["actual_25plus"], g["p25_base"])
            c25 = metrics(g["actual_25plus"], g["p25_m95t"])
            p25rows.append({"season": int(season), **{f"base_{k}": v for k, v in b25.items()}, **{f"cand_{k}": v for k, v in c25.items()}})
        b25 = metrics(cand["actual_25plus"], cand["p25_base"])
        c25 = metrics(cand["actual_25plus"], cand["p25_m95t"])
        p25rows.append({"season": "POOLED", **{f"base_{k}": v for k, v in b25.items()}, **{f"cand_{k}": v for k, v in c25.items()}})
    pd.DataFrame(p25rows).to_csv(args.out_dir / "m95t_25plus_diagnostic.csv", index=False)

    yr = []
    for season, g in list(cand.groupby("season")) + [("POOLED", cand)]:
        pear = float(num(g["actual_carries"]).corr(num(g["actual_rush_yards"])))
        ok = num(g["actual_carries"]).notna() & num(g["actual_rush_yards"]).notna()
        spr = float(spearmanr(num(g.loc[ok, "actual_carries"]), num(g.loc[ok, "actual_rush_yards"])).statistic) if ok.sum() >= 5 else np.nan
        row = {
            "season": season, "n": int(len(g)),
            "carry_rushyard_pearson": pear, "carry_rushyard_spearman": spr,
            "events75": int(g["actual_75plus_rush_yards"].sum()),
            "events100": int(g["actual_100plus_rush_yards"].sum()),
            "base_auc75": auc_score(g["actual_75plus_rush_yards"], g["p20_base"]),
            "cand_auc75": auc_score(g["actual_75plus_rush_yards"], g["p20_m95t"]),
            "base_auc100": auc_score(g["actual_100plus_rush_yards"], g["p20_base"]),
            "cand_auc100": auc_score(g["actual_100plus_rush_yards"], g["p20_m95t"]),
        }
        row["auc75_gain"] = row["cand_auc75"] - row["base_auc75"] if np.isfinite(row["base_auc75"]) and np.isfinite(row["cand_auc75"]) else np.nan
        row["auc100_gain"] = row["cand_auc100"] - row["base_auc100"] if np.isfinite(row["base_auc100"]) and np.isfinite(row["cand_auc100"]) else np.nan
        yr.append(row)
    ydf = pd.DataFrame(yr)
    ydf.to_csv(args.out_dir / "m95t_rush_yard_translation_guard.csv", index=False)

    season_rows = mdf.loc[mdf["season"].astype(str).ne("POOLED")].copy()
    prow = mdf.loc[mdf["season"].astype(str).eq("POOLED")].iloc[0]
    ypool = ydf.loc[ydf["season"].astype(str).eq("POOLED")].iloc[0]
    mass_err = float(cand["rank_mass_error"].abs().max())
    brier_nonneg = int((season_rows["brier_gain"] >= -1e-6).sum())
    logloss_nonneg = int((season_rows["logloss_gain"] >= -1e-6).sum())
    max_brier_reg = float(np.maximum(-season_rows["brier_gain"], 0).max())
    max_ll_reg = float(np.maximum(-season_rows["logloss_gain"], 0).max())
    max_gap_reg = float(np.maximum(-season_rows["abs_gap_gain"], 0).max())
    trouble = season_rows.loc[season_rows["season"].astype(int).isin([2023, 2025])]
    trouble_guard = int(
        (trouble["brier_gain"] >= -0.0075).all()
        and (trouble["logloss_gain"] >= -0.020).all()
        and (trouble["abs_gap_gain"] >= -0.025).all()
    )
    yard75_guard = int((not np.isfinite(ypool["auc75_gain"])) or ypool["auc75_gain"] >= -0.01)
    yard100_guard = int((not np.isfinite(ypool["auc100_gain"])) or ypool["auc100_gain"] >= -0.01)
    gates = {
        "pooled_brier_improves": int(prow["brier_gain"] > 0),
        "pooled_logloss_improves": int(prow["logloss_gain"] > 0),
        "pooled_auc_guard": int(prow["auc_gain"] >= -0.01),
        "season_brier_nonnegative_ge4": int(brier_nonneg >= 4),
        "season_logloss_nonnegative_ge4": int(logloss_nonneg >= 4),
        "max_brier_regression_le_0p0075": int(max_brier_reg <= 0.0075),
        "max_logloss_regression_le_0p020": int(max_ll_reg <= 0.020),
        "max_abs_gap_regression_le_0p025": int(max_gap_reg <= 0.025),
        "trouble_year_2023_2025_guard": trouble_guard,
        "rank_layer_mass_preserved": int(mass_err <= 1e-9),
        "rushyard_75_auc_guard": yard75_guard,
        "rushyard_100_auc_guard": yard100_guard,
    }
    passed = all(gates.values())
    disposition = "M95T_PASS_FREEZE_FOR_M96_AND_2026_PROSPECTIVE_CONFIRMATION" if passed else "M95T_FAIL_STOP_NEW_RB_TAIL_CANDIDATES_RETAIN_M94C_M95F_PROCEED_M96"
    disp = {
        "m95t_role": "final_retrospective_stable_workhorse_tail_candidate",
        "primary_target": "stable_workhorse_20plus",
        "seasons": "2020,2021,2022,2023,2024,2025",
        "model_fit": 0, "feature_search": 0, "coefficient_search": 0,
        "hyperparameter_search": 0, "sportsbook_inputs": 0, "production_change": 0,
        "feed_shrink_k": FEED_SHRINK_K, "ratio_lo": RATIO_LO, "ratio_hi": RATIO_HI,
        "mass_shrink": MASS_SHRINK, "rank_blend": RANK_BLEND,
        "rank_delta_cap": RANK_DELTA_CAP, "season_brier_nonnegative": brier_nonneg,
        "season_logloss_nonnegative": logloss_nonneg,
        "max_brier_regression": max_brier_reg, "max_logloss_regression": max_ll_reg,
        "max_abs_gap_regression": max_gap_reg, "rank_mass_error_max": mass_err,
        **gates, "retrospective_pass": int(passed), "disposition": disposition,
        "next_phase": "M96_RB_RUSHING_YARD_SYNTHESIS_MANDATORY",
    }
    pd.DataFrame([disp]).to_csv(args.out_dir / "m95t_disposition.csv", index=False)

    audit = {
        "candidate": "M95F backbone + within-week aligned feed rerank + fast relative league population anchor",
        "population_anchor": "prior4 lead20 / season-to-date lead20; clipped 0.70-1.30; 50% shrink; max +/-15% relative mass",
        "player_rerank": "k=4 M95K feed20_rate + carry_ceiling95 within-week percentile; aligned half only; 0.50*(feed-base rank); +/-0.25 logodds cap; mass-preserved",
        "central_carries_changed": 0, "vacancy_changed": 0,
        "target_week_postgame_inputs": 0, "sportsbook_inputs": 0,
        "yardage_claim": "translation guard only; M96 required for rushing-yard point synthesis",
    }
    pd.DataFrame([audit]).to_csv(args.out_dir / "m95t_method_audit.csv", index=False)

    print("\n[M95T] 20+ metrics")
    print(mdf.to_string(index=False))
    print("\n[M95T] yardage translation guard")
    print(ydf.to_string(index=False))
    print("\n[M95T] disposition")
    print(pd.DataFrame([disp]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
