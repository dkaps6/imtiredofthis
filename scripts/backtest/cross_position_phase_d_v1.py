#!/usr/bin/env python3
"""Phase D: same-game conservation + player environment sensitivity.

Diagnostic-only continuation of the frozen cross-position catastrophic casebook.
No sportsbook inputs or production model changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20260907
BOOTSTRAPS = 2000
THRESHOLDS = {"QB": 100.0, "WR": 50.0, "TE": 40.0, "RB": 40.0}


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def slope_xy(x: pd.Series, y: pd.Series) -> float:
    q = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(q) < 3 or float(q["x"].std(ddof=0)) <= 1e-12:
        return np.nan
    return float(np.polyfit(q["x"].to_numpy(float), q["y"].to_numpy(float), 1)[0])


def pearson_xy(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_xy(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(spearmanr(x, y, nan_policy="omit").statistic)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, rng: np.random.Generator) -> tuple[float, float]:
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    vals = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        v = fn(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def aggregate_team_games(all_rows: pd.DataFrame) -> pd.DataFrame:
    x = all_rows.copy()
    x["position"] = x["position"].astype(str).str.upper()
    for c in ["season", "week", "pred", "actual", "pred_opportunity", "actual_opportunity"]:
        x[c] = num(x[c])

    keys = ["season", "week", "team"]

    qb = x.loc[x["position"].eq("QB")].copy()
    # Frozen rule: primary QB selected by largest predicted opportunity, never actual.
    qb = qb.sort_values(keys + ["pred_opportunity"], ascending=[True, True, True, False])
    qb = qb.drop_duplicates(keys, keep="first")
    qb_out = qb[keys].copy()
    qb_out["qb_attempt_resid"] = qb["actual_opportunity"].to_numpy() - qb["pred_opportunity"].to_numpy()
    qb_out["qb_yard_resid"] = qb["actual"].to_numpy() - qb["pred"].to_numpy()
    qb_out["qb_cat_under"] = (qb["catastrophic"].astype(bool) & qb["direction"].eq("UNDERPROJECTED")).to_numpy()
    qb_out["qb_cat_over"] = (qb["catastrophic"].astype(bool) & qb["direction"].eq("OVERPROJECTED")).to_numpy()

    def pos_agg(pos: str, prefix: str) -> pd.DataFrame:
        p = x.loc[x["position"].eq(pos)].copy()
        p["opp_resid"] = p["actual_opportunity"] - p["pred_opportunity"]
        p["yard_resid"] = p["actual"] - p["pred"]
        p["cat_under_i"] = p["catastrophic"].astype(bool) & p["direction"].eq("UNDERPROJECTED")
        p["cat_over_i"] = p["catastrophic"].astype(bool) & p["direction"].eq("OVERPROJECTED")
        g = p.groupby(keys, as_index=False).agg(
            opp_resid=("opp_resid", "sum"),
            yard_resid=("yard_resid", "sum"),
            cat_under=("cat_under_i", "max"),
            cat_over=("cat_over_i", "max"),
            n_players=("player_clean_key", "nunique"),
        )
        return g.rename(columns={
            "opp_resid": f"{prefix}_opp_resid",
            "yard_resid": f"{prefix}_yard_resid",
            "cat_under": f"{prefix}_cat_under",
            "cat_over": f"{prefix}_cat_over",
            "n_players": f"{prefix}_n_players",
        })

    wr = pos_agg("WR", "wr")
    te = pos_agg("TE", "te")
    rb = pos_agg("RB", "rb")

    out = qb_out.merge(wr, on=keys, how="inner").merge(te, on=keys, how="inner").merge(rb, on=keys, how="inner")
    out["receiver_target_resid"] = out["wr_opp_resid"] + out["te_opp_resid"]
    out["receiver_yard_resid"] = out["wr_yard_resid"] + out["te_yard_resid"]
    out["receiver_cat_under"] = out["wr_cat_under"] | out["te_cat_under"]
    out["receiver_cat_over"] = out["wr_cat_over"] | out["te_cat_over"]

    def signature(r: pd.Series) -> str:
        q = float(r.qb_attempt_resid)
        rc = float(r.receiver_target_resid)
        rbv = float(r.rb_opp_resid)
        if q > 0 and rc > 0 and rbv < 0:
            return "PASS_STATE_HIGH"
        if q < 0 and rc < 0 and rbv > 0:
            return "PASS_STATE_LOW"
        if (q > 0 and rc > 0) or (q < 0 and rc < 0):
            return "PASS_RECEIVER_SHARED_ONLY"
        if rbv != 0 and not (((q > 0) and (rc > 0)) or ((q < 0) and (rc < 0))):
            return "RUSH_ONLY"
        return "MIXED"

    out["game_state_signature"] = out.apply(signature, axis=1)
    return out


def correlation_scorecard(team: pd.DataFrame) -> pd.DataFrame:
    primary = team.loc[team["season"].between(2024, 2025)].copy()
    pairs = [
        ("QB_ATT__WR_TARGET", "qb_attempt_resid", "wr_opp_resid", "positive"),
        ("QB_ATT__TE_TARGET", "qb_attempt_resid", "te_opp_resid", "positive"),
        ("QB_ATT__WRTE_TARGET", "qb_attempt_resid", "receiver_target_resid", "positive"),
        ("QB_ATT__RB_CARRY", "qb_attempt_resid", "rb_opp_resid", "negative"),
        ("QB_YARDS__WR_YARDS", "qb_yard_resid", "wr_yard_resid", "positive"),
        ("QB_YARDS__TE_YARDS", "qb_yard_resid", "te_yard_resid", "positive"),
        ("QB_YARDS__WRTE_YARDS", "qb_yard_resid", "receiver_yard_resid", "positive"),
        ("QB_YARDS__RB_RUSH_YARDS", "qb_yard_resid", "rb_yard_resid", "diagnostic"),
        ("WRTE_TARGET__RB_CARRY", "receiver_target_resid", "rb_opp_resid", "negative"),
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, a, b, expected in pairs:
        q = primary[[a, b]].dropna()
        xv = q[a].to_numpy(float)
        yv = q[b].to_numpy(float)
        p = pearson_xy(xv, yv)
        s = spearman_xy(xv, yv)
        pcl, pch = bootstrap_ci(xv, yv, pearson_xy, rng)
        scl, sch = bootstrap_ci(xv, yv, spearman_xy, rng)
        same = float(np.mean(np.sign(xv) == np.sign(yv)))
        opp = float(np.mean(np.sign(xv) == -np.sign(yv)))
        rows.append({
            "relation": name,
            "x": a,
            "y": b,
            "expected_sign": expected,
            "n": len(q),
            "pearson": p,
            "pearson_ci_low": pcl,
            "pearson_ci_high": pch,
            "spearman": s,
            "spearman_ci_low": scl,
            "spearman_ci_high": sch,
            "same_sign_rate": same,
            "opposite_sign_rate": opp,
            "zero_involved_rate": float(np.mean((np.sign(xv) == 0) | (np.sign(yv) == 0))),
        })
    return pd.DataFrame(rows)


def catastrophic_cooccurrence(team: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "qb_cat_under", "qb_cat_over",
        "wr_cat_under", "wr_cat_over",
        "te_cat_under", "te_cat_over",
        "receiver_cat_under", "receiver_cat_over",
        "rb_cat_under", "rb_cat_over",
    ]
    q = team.loc[team["season"].between(2024, 2025)].copy()
    rows = []
    for a in flags:
        a_mask = q[a].astype(bool)
        a_n = int(a_mask.sum())
        for b in flags:
            if a == b:
                continue
            b_mask = q[b].astype(bool)
            both = int((a_mask & b_mask).sum())
            rows.append({
                "event_a": a,
                "event_b": b,
                "n_team_games": len(q),
                "event_a_n": a_n,
                "event_b_n": int(b_mask.sum()),
                "both_n": both,
                "p_b_given_a": both / a_n if a_n else np.nan,
                "p_b_unconditional": float(b_mask.mean()) if len(q) else np.nan,
                "lift_b_given_a": (both / a_n) / float(b_mask.mean()) if a_n and b_mask.mean() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def player_sensitivity(all_rows: pd.DataFrame) -> pd.DataFrame:
    x = all_rows.copy()
    x["position"] = x["position"].astype(str).str.upper()
    x["signed_resid"] = num(x["actual"]) - num(x["pred"])
    x["abs_error"] = num(x["abs_error"])
    x["game_spot_score"] = num(x["game_spot_score"])
    x["cat_under"] = x["catastrophic"].astype(bool) & x["direction"].eq("UNDERPROJECTED")
    x["cat_over"] = x["catastrophic"].astype(bool) & x["direction"].eq("OVERPROJECTED")
    x["is_high_opp"] = x["opportunity_quartile"].isin(["Q3", "Q4"])
    x["is_q4"] = x["opportunity_quartile"].eq("Q4")
    keys = ["position", "player_clean_key", "player"]
    out = []

    for k, g in x.groupby(keys, dropna=False):
        g = g.sort_values(["season", "week"]).copy()
        n = len(g)
        fav = g.loc[g["game_spot_bucket"].eq("FAVORABLE")]
        adv = g.loc[g["game_spot_bucket"].eq("ADVERSE")]
        neu = g.loc[g["game_spot_bucket"].eq("NEUTRAL")]
        seasons = int(g["season"].nunique())
        available_seasons = int(x.loc[x["position"].eq(k[0]), "season"].nunique())
        eligible = (
            n >= 16
            and len(fav) >= 4
            and len(adv) >= 4
            and float(g["is_high_opp"].mean()) >= 0.40
            and (seasons >= 2 or available_seasons < 2)
        )
        if not eligible:
            continue

        half = n // 2
        g1 = g.iloc[:half]
        g2 = g.iloc[half:]
        full_slope = slope_xy(g["game_spot_score"], g["signed_resid"])
        s1 = slope_xy(g1["game_spot_score"], g1["signed_resid"])
        s2 = slope_xy(g2["game_spot_score"], g2["signed_resid"])
        sp = spearman_xy(g["game_spot_score"].to_numpy(float), g["signed_resid"].to_numpy(float))

        def mean_or_nan(h: pd.DataFrame, col: str) -> float:
            return float(h[col].mean()) if len(h) else np.nan

        out.append({
            "position": k[0],
            "player_clean_key": k[1],
            "player": k[2],
            "n": n,
            "seasons": seasons,
            "season_min": int(g["season"].min()),
            "season_max": int(g["season"].max()),
            "favorable_n": len(fav),
            "neutral_n": len(neu),
            "adverse_n": len(adv),
            "high_opp_share": float(g["is_high_opp"].mean()),
            "q4_share": float(g["is_q4"].mean()),
            "mean_resid_overall": float(g["signed_resid"].mean()),
            "mean_resid_favorable": mean_or_nan(fav, "signed_resid"),
            "mean_resid_neutral": mean_or_nan(neu, "signed_resid"),
            "mean_resid_adverse": mean_or_nan(adv, "signed_resid"),
            "mae_overall": float(g["abs_error"].mean()),
            "mae_favorable": mean_or_nan(fav, "abs_error"),
            "mae_neutral": mean_or_nan(neu, "abs_error"),
            "mae_adverse": mean_or_nan(adv, "abs_error"),
            "cat_under_rate_favorable": float(fav["cat_under"].mean()) if len(fav) else np.nan,
            "cat_under_rate_neutral": float(neu["cat_under"].mean()) if len(neu) else np.nan,
            "cat_under_rate_adverse": float(adv["cat_under"].mean()) if len(adv) else np.nan,
            "cat_over_rate_favorable": float(fav["cat_over"].mean()) if len(fav) else np.nan,
            "cat_over_rate_neutral": float(neu["cat_over"].mean()) if len(neu) else np.nan,
            "cat_over_rate_adverse": float(adv["cat_over"].mean()) if len(adv) else np.nan,
            "fav_minus_adverse_residual": mean_or_nan(fav, "signed_resid") - mean_or_nan(adv, "signed_resid"),
            "spot_slope": full_slope,
            "spot_spearman": sp,
            "first_half_slope": s1,
            "second_half_slope": s2,
            "slope_sign_stable_positive": bool(np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0),
            "slope_sign_stable": bool(np.isfinite(s1) and np.isfinite(s2) and np.sign(s1) == np.sign(s2)),
            "star_q4": bool(float(g["is_q4"].mean()) >= 0.50),
        })

    d = pd.DataFrame(out)
    if d.empty:
        return d

    d["abs_spot_slope"] = d["spot_slope"].abs()
    d["position_median_abs_slope"] = d.groupby("position")["abs_spot_slope"].transform("median")
    d["position_median_adverse_cat_under"] = d.groupby("position")["cat_under_rate_adverse"].transform("median")

    labels = []
    for _, r in d.iterrows():
        th = THRESHOLDS[str(r.position)]
        tags = []
        if r.fav_minus_adverse_residual >= 0.25 * th and bool(r.slope_sign_stable_positive):
            tags.append("ENVIRONMENT_SENSITIVE_CANDIDATE")
        if abs(r.fav_minus_adverse_residual) <= 0.10 * th and r.abs_spot_slope <= r.position_median_abs_slope:
            tags.append("ENVIRONMENT_RESISTANT_CANDIDATE")
        if r.mean_resid_adverse >= 0 and r.cat_under_rate_adverse >= r.position_median_adverse_cat_under:
            tags.append("ADVERSE_SPOT_CEILING_CANDIDATE")
        if (
            (r.cat_under_rate_favorable - r.cat_under_rate_adverse) >= 0.10
            and r.fav_minus_adverse_residual >= 0.15 * th
        ):
            tags.append("FAVORABLE_SPOT_AMPLIFIER_CANDIDATE")
        labels.append("|".join(tags) if tags else "UNSTABLE_OR_UNRESOLVED")
    d["candidate_labels"] = labels
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-c-all-rows", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(args.phase_c_all_rows)

    required = {
        "position", "season", "week", "team", "player", "player_clean_key",
        "pred", "actual", "abs_error", "direction", "threshold", "catastrophic",
        "pred_opportunity", "actual_opportunity", "opportunity_quartile",
        "game_spot_score", "game_spot_bucket",
    }
    missing = sorted(required - set(all_rows.columns))
    if missing:
        raise RuntimeError(f"missing required Phase-C columns: {missing}")

    observed = {
        p: sorted(pd.to_numeric(g["threshold"], errors="coerce").dropna().unique().tolist())
        for p, g in all_rows.groupby(all_rows["position"].astype(str).str.upper())
    }
    for p, th in THRESHOLDS.items():
        vals = observed.get(p, [])
        if vals != [th]:
            raise RuntimeError(f"threshold integrity failure {p}: {vals} != {[th]}")

    team = aggregate_team_games(all_rows)
    corr = correlation_scorecard(team)
    co = catastrophic_cooccurrence(team)
    sens = player_sensitivity(all_rows)
    star = sens.loc[sens["star_q4"]].copy() if not sens.empty else sens.copy()
    labels = sens.loc[sens["candidate_labels"].ne("UNSTABLE_OR_UNRESOLVED")].copy() if not sens.empty else sens.copy()

    sig = (
        team.groupby(["season", "game_state_signature"], as_index=False)
        .agg(
            n_team_games=("team", "size"),
            mean_qb_attempt_resid=("qb_attempt_resid", "mean"),
            mean_receiver_target_resid=("receiver_target_resid", "mean"),
            mean_rb_carry_resid=("rb_opp_resid", "mean"),
            mean_qb_yard_resid=("qb_yard_resid", "mean"),
            mean_receiver_yard_resid=("receiver_yard_resid", "mean"),
            mean_rb_rush_yard_resid=("rb_yard_resid", "mean"),
        )
    )

    team.to_csv(out_dir / "phase_d_same_game_team_casebook.csv", index=False)
    corr.to_csv(out_dir / "phase_d_same_game_correlations.csv", index=False)
    co.to_csv(out_dir / "phase_d_catastrophic_cooccurrence.csv", index=False)
    sig.to_csv(out_dir / "phase_d_game_state_signatures.csv", index=False)
    sens.to_csv(out_dir / "phase_d_player_environment_sensitivity.csv", index=False)
    star.to_csv(out_dir / "phase_d_star_q4_environment_sensitivity.csv", index=False)
    labels.to_csv(out_dir / "phase_d_candidate_labels.csv", index=False)

    result = {
        "disposition": "PHASE_D_GAME_STATE_AND_PLAYER_SENSITIVITY_COMPLETE",
        "input_rows": int(len(all_rows)),
        "common_team_games": int(len(team)),
        "common_team_games_by_season": {str(int(k)): int(v) for k, v in team.groupby("season").size().items()},
        "primary_correlation_rows": int(len(corr)),
        "eligible_player_rows": int(len(sens)),
        "star_q4_player_rows": int(len(star)),
        "labeled_candidate_rows": int(len(labels)),
        "sportsbook_features_used": 0,
        "future_or_same_game_features_added": 0,
        "production_parameters_changed": 0,
        "bootstrap_seed": SEED,
        "bootstraps": BOOTSTRAPS,
        "thresholds": THRESHOLDS,
    }
    (out_dir / "phase_d_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("\n[phase_d] correlations")
    print(corr.to_string(index=False))
    print("\n[phase_d] signatures")
    print(sig.to_string(index=False))
    print("\n[phase_d] candidate labels")
    show = [
        "position", "player", "n", "q4_share", "fav_minus_adverse_residual",
        "spot_slope", "mean_resid_adverse", "cat_under_rate_adverse", "candidate_labels",
    ]
    print(labels[show].to_string(index=False))


if __name__ == "__main__":
    main()
