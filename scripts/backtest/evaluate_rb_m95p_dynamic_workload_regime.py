"""M95P: dynamic workload-regime / population-prior audit.

Research-only diagnostic. M95O showed that a fixed 2024 agreement gate does not
transfer cleanly across 2023/2025 because the stable-workhorse workload prior and
probability calibration move materially by season/window.

M95P asks two narrower questions without fitting a new tail model:

1) Is 2023 actually an unusual RB workload year when viewed against a broader
   modern-NFL history rather than only 2023-2025?
2) Can pregame-only rolling league/team workload state help explain the changing
   stable-workhorse 20+ carry prevalence/calibration seen in 2023-2025?

The broad census uses nflverse weekly player stats for 2018-2025 and evaluates
team lead-RB workload. The exact model-trace layer remains limited to the
comparable M95K/M95L stable-workhorse population available in 2023-2025. Broad
lead-RB census statistics are therefore context, not a substitute label for the
M95K stable-workhorse cohort.

No sportsbook input. No production change. No feature/coefficient search. No
new prediction model is fit. Every rolling regime feature is shifted so it uses
only completed games from prior weeks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.player_form_v2 import _normalize_weekly, _to_pandas

SEASONS = tuple(range(2018, 2026))
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]
RB_POS = {"RB", "FB", "HB"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return hits[0]


def load_weekly(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    x = _to_pandas(raw)
    if x.empty:
        raise RuntimeError(f"nflreadpy returned zero weekly player rows for {season}")
    x = _normalize_weekly(x, season)
    x = x.loc[num(x["week"]).between(1, 18)].copy()
    x["position"] = x["position"].astype("string").fillna("").str.upper().str.strip()
    x["rushes"] = num(x["rushes"]).fillna(0.0)
    return x


def build_team_week_census() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_weeks = []
    for season in SEASONS:
        x = load_weekly(season)
        all_weeks.append(x)
        print(f"[m95p] loaded season={season} normalized_rows={len(x)}")
    logs = pd.concat(all_weeks, ignore_index=True, sort=False)

    team = (
        logs.groupby(TEAM_KEYS, as_index=False)
        .agg(team_rushes=("rushes", "sum"))
    )
    qb = (
        logs.loc[logs["position"].eq("QB")]
        .groupby(TEAM_KEYS, as_index=False)
        .agg(qb_rushes=("rushes", "sum"))
    )
    rb = logs.loc[logs["position"].isin(RB_POS)].copy()
    rb = rb.loc[rb["rushes"].ge(0)].copy()
    if rb.empty:
        raise RuntimeError("M95P broad census found zero RB/FB/HB rows")

    rb_team = (
        rb.groupby(TEAM_KEYS, as_index=False)
        .agg(
            rb_total_carries=("rushes", "sum"),
            lead_rb_carries=("rushes", "max"),
            rb_count=("rushes", lambda s: int((num(s) > 0).sum())),
        )
    )
    out = team.merge(qb, on=TEAM_KEYS, how="left", validate="one_to_one")
    out = out.merge(rb_team, on=TEAM_KEYS, how="inner", validate="one_to_one")
    for c in ["team_rushes", "qb_rushes", "rb_total_carries", "lead_rb_carries", "rb_count"]:
        out[c] = num(out[c]).fillna(0.0)
    out["lead20"] = out["lead_rb_carries"].ge(20).astype(int)
    out["lead25"] = out["lead_rb_carries"].ge(25).astype(int)
    out["lead_rb_share_of_rb"] = np.where(out["rb_total_carries"] > 0, out["lead_rb_carries"] / out["rb_total_carries"], np.nan)
    out["lead_rb_share_of_team"] = np.where(out["team_rushes"] > 0, out["lead_rb_carries"] / out["team_rushes"], np.nan)
    out["qb_rush_share"] = np.where(out["team_rushes"] > 0, out["qb_rushes"] / out["team_rushes"], np.nan)
    out = out.sort_values(TEAM_KEYS).reset_index(drop=True)

    # Weekly league state. These are descriptive current-week outcomes; the
    # pregame feature table below shifts/rolls them so a target week never sees
    # its own outcomes.
    weekly = (
        out.groupby(["season", "week"], as_index=False)
        .agg(
            teams=("team", "nunique"),
            lead20_rate=("lead20", "mean"),
            lead25_rate=("lead25", "mean"),
            mean_lead_carries=("lead_rb_carries", "mean"),
            mean_team_rushes=("team_rushes", "mean"),
            mean_lead_rb_share=("lead_rb_share_of_rb", "mean"),
            mean_qb_rush_share=("qb_rush_share", "mean"),
            mean_rb_count=("rb_count", "mean"),
        )
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    return out, weekly


def add_pregame_regime_features(team_week: pd.DataFrame, weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = weekly.copy().sort_values(["season", "week"]).reset_index(drop=True)
    metrics = [
        "lead20_rate", "lead25_rate", "mean_lead_carries", "mean_team_rushes",
        "mean_lead_rb_share", "mean_qb_rush_share", "mean_rb_count",
    ]
    for c in metrics:
        # Season-to-date through the previous week.
        w[f"league_std_{c}"] = (
            w.groupby("season", sort=False)[c]
            .transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
        )
        # Last four completed league weeks through the previous week.
        w[f"league_l4_{c}"] = (
            w.groupby("season", sort=False)[c]
            .transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
        )

    t = team_week.copy().sort_values(TEAM_KEYS).reset_index(drop=True)
    team_metrics = [
        "lead20", "lead25", "lead_rb_carries", "team_rushes",
        "lead_rb_share_of_rb", "qb_rush_share", "rb_count",
    ]
    for c in team_metrics:
        t[f"team_l4_{c}"] = (
            t.groupby(["season", "team"], sort=False)[c]
            .transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
        )
    league_cols = [c for c in w.columns if c.startswith("league_")]
    t = t.merge(w[["season", "week", *league_cols]], on=["season", "week"], how="left", validate="many_to_one")
    return t, w


def broad_season_summary(team_week: pd.DataFrame) -> pd.DataFrame:
    def summarise(g: pd.DataFrame, window: str) -> dict:
        lead = num(g["lead_rb_carries"]).dropna()
        return {
            "window": window,
            "team_weeks": int(len(g)),
            "lead20_rate": float(num(g["lead20"]).mean()),
            "lead25_rate": float(num(g["lead25"]).mean()),
            "mean_lead_carries": float(lead.mean()),
            "p90_lead_carries": float(lead.quantile(.90)),
            "p95_lead_carries": float(lead.quantile(.95)),
            "mean_team_rushes": float(num(g["team_rushes"]).mean()),
            "mean_lead_rb_share": float(num(g["lead_rb_share_of_rb"]).mean()),
            "mean_qb_rush_share": float(num(g["qb_rush_share"]).mean()),
            "mean_rb_count": float(num(g["rb_count"]).mean()),
        }

    rows = []
    for season, g in team_week.groupby("season"):
        rows.append({"season": int(season), **summarise(g, "full_regular_season")})
        late = g.loc[num(g["week"]).between(13, 18)]
        rows.append({"season": int(season), **summarise(late, "weeks_13_18")})
    out = pd.DataFrame(rows)

    # Outlier context is based on the 8-season modern sample, separately by
    # window. This is descriptive only and never enters a prediction.
    for c in ["lead20_rate", "lead25_rate", "mean_lead_carries"]:
        out[f"{c}_z"] = out.groupby("window")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if float(s.std(ddof=0)) > 0 else 0.0
        )
        out[f"{c}_rank_pct"] = out.groupby("window")[c].rank(pct=True, method="average")
    return out


def stable_mask(z: pd.DataFrame) -> pd.Series:
    trend = num(z["rb_rb_share_avg1"]) - num(z["rb_rb_share_avg5"])
    return (
        num(z["role_is_workhorse"]).fillna(0).eq(1)
        & num(z["prior_top1_unavailable"]).fillna(0).eq(0)
        & num(z["target_was_prior_top1"]).fillna(0).eq(1)
        & trend.ge(-0.10)
        & num(z["self_inj_out"]).fillna(0).eq(0)
        & num(z["self_inj_doubtful"]).fillna(0).eq(0)
    )


def standardize_exact(df: pd.DataFrame, season: int, source: str) -> pd.DataFrame:
    x = lower(df)
    x = x.loc[num(x["season"]).eq(season)].copy()
    if "stable_workhorse_m95k" in x.columns:
        x = x.loc[num(x["stable_workhorse_m95k"]).eq(1)].copy()
    else:
        needed = [
            "role_is_workhorse", "prior_top1_unavailable", "target_was_prior_top1",
            "rb_rb_share_avg1", "rb_rb_share_avg5", "self_inj_out", "self_inj_doubtful",
        ]
        missing = [c for c in needed if c not in x.columns]
        if missing:
            raise RuntimeError(f"M95P {source} cannot reconstruct stable cohort; missing {missing}")
        x = x.loc[stable_mask(x)].copy()

    if "actual_carries" in x.columns:
        x["actual_carries"] = num(x["actual_carries"])
    elif "actual_rush_att" in x.columns:
        x["actual_carries"] = num(x["actual_rush_att"])
    else:
        raise RuntimeError(f"M95P {source} missing actual carries")

    if "p20_base" in x.columns:
        x["p20_base"] = num(x["p20_base"])
    elif "cal_prob_20" in x.columns:
        x["p20_base"] = num(x["cal_prob_20"])
    else:
        raise RuntimeError(f"M95P {source} missing M95F 20+ probability")

    x["actual_20plus"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus"] = x["actual_carries"].ge(25).astype(int)
    x["calibration_error20"] = x["actual_20plus"] - x["p20_base"]
    x["source_trace"] = source
    keep = [*PLAYER_KEYS, "actual_carries", "actual_20plus", "actual_25plus", "p20_base", "calibration_error20", "source_trace"]
    return x[[c for c in keep if c in x.columns]].drop_duplicates(PLAYER_KEYS)


def exact_model_trace(m95g_root: Path, m95k_root: Path, m95l_root: Path) -> pd.DataFrame:
    g24 = pd.read_csv(find_one(m95g_root, "m95g_2024_holdout_trace.csv"), low_memory=False)
    k25 = pd.read_csv(find_one(m95k_root, "m95k_2025_trace.csv"), low_memory=False)
    l23 = pd.read_csv(find_one(m95l_root, "m95l_2023_confirmation_trace.csv"), low_memory=False)
    z23 = standardize_exact(l23, 2023, "m95l_2023")
    z24 = standardize_exact(g24, 2024, "m95g_2024")
    z25 = standardize_exact(k25, 2025, "m95k_2025")
    out = pd.concat([z23, z24, z25], ignore_index=True, sort=False)
    return out.sort_values(PLAYER_KEYS).reset_index(drop=True)


def safe_corr(a, b, method="spearman") -> tuple[int, float]:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 8 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return len(z), np.nan
    return len(z), float(z.a.corr(z.b, method=method))


def exact_scope_summary(z: pd.DataFrame) -> pd.DataFrame:
    scopes = []
    for season in (2023, 2024, 2025):
        g = z.loc[num(z["season"]).eq(season)]
        scopes.append((f"{season}_full_available", g))
        late = g.loc[num(g["week"]).between(13, 18)]
        scopes.append((f"{season}_w13_18", late))
    rows = []
    for scope, g in scopes:
        if g.empty:
            continue
        rows.append({
            "scope": scope,
            "n": int(len(g)),
            "unique_players": int(g["player_clean_key"].nunique()),
            "actual20_rate": float(num(g["actual_20plus"]).mean()),
            "actual25_rate": float(num(g["actual_25plus"]).mean()),
            "mean_actual_carries": float(num(g["actual_carries"]).mean()),
            "mean_m95f_p20": float(num(g["p20_base"]).mean()),
            "calibration_gap_actual_minus_pred": float(num(g["actual_20plus"]).mean() - num(g["p20_base"]).mean()),
            "mean_league_l4_lead20_rate": float(num(g["league_l4_lead20_rate"]).mean()),
            "mean_league_std_lead20_rate": float(num(g["league_std_lead20_rate"]).mean()),
            "mean_league_l4_lead25_rate": float(num(g["league_l4_lead25_rate"]).mean()),
            "mean_league_l4_mean_lead_carries": float(num(g["league_l4_mean_lead_carries"]).mean()),
            "mean_team_l4_lead_rb_carries": float(num(g["team_l4_lead_rb_carries"]).mean()),
            "mean_team_l4_lead_rb_share": float(num(g["team_l4_lead_rb_share_of_rb"]).mean()),
            "mean_team_l4_qb_rush_share": float(num(g["team_l4_qb_rush_share"]).mean()),
            "mean_team_l4_rb_count": float(num(g["team_l4_rb_count"]).mean()),
        })
    return pd.DataFrame(rows)


def signal_audit(z: pd.DataFrame) -> pd.DataFrame:
    features = [
        "league_l4_lead20_rate", "league_std_lead20_rate", "league_l4_lead25_rate",
        "league_l4_mean_lead_carries", "league_l4_mean_team_rushes",
        "league_l4_mean_lead_rb_share", "league_l4_mean_qb_rush_share",
        "league_l4_mean_rb_count", "team_l4_lead20", "team_l4_lead25",
        "team_l4_lead_rb_carries", "team_l4_team_rushes",
        "team_l4_lead_rb_share_of_rb", "team_l4_qb_rush_share", "team_l4_rb_count",
    ]
    rows = []
    scopes = [("all_2023_2025", z)]
    for s in (2023, 2024, 2025):
        scopes.append((str(s), z.loc[num(z["season"]).eq(s)]))
    for scope, g in scopes:
        for feat in features:
            if feat not in g.columns:
                continue
            n1, c1 = safe_corr(g[feat], g["actual_20plus"])
            n2, c2 = safe_corr(g[feat], g["calibration_error20"])
            rows.append({
                "scope": scope, "feature": feat,
                "n_actual20": n1, "spearman_vs_actual20": c1,
                "n_calibration_error": n2, "spearman_vs_calibration_error20": c2,
            })
    return pd.DataFrame(rows)


def regime_bins(z: pd.DataFrame) -> pd.DataFrame:
    # Diagnostic population-relative bins. Quantile boundaries are descriptive
    # and are not a proposed production threshold.
    feat = "league_l4_lead20_rate"
    x = z.loc[num(z[feat]).notna()].copy()
    if x.empty:
        return pd.DataFrame()
    x["regime_bin"] = pd.qcut(num(x[feat]), q=4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    rows = []
    for b, g in x.groupby("regime_bin", observed=True):
        rows.append({
            "regime_bin": str(b), "n": int(len(g)),
            "mean_pregame_league_l4_lead20_rate": float(num(g[feat]).mean()),
            "actual20_rate": float(num(g["actual_20plus"]).mean()),
            "mean_m95f_p20": float(num(g["p20_base"]).mean()),
            "calibration_gap_actual_minus_pred": float(num(g["actual_20plus"]).mean() - num(g["p20_base"]).mean()),
            "mean_actual_carries": float(num(g["actual_carries"]).mean()),
        })
    return pd.DataFrame(rows)


def disposition(broad: pd.DataFrame, scope: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    late = broad.loc[broad["window"].eq("weeks_13_18")].copy()
    r23 = late.loc[late["season"].eq(2023)]
    z20 = float(r23["lead20_rate_z"].iloc[0]) if len(r23) else np.nan
    z25 = float(r23["lead25_rate_z"].iloc[0]) if len(r23) else np.nan
    rank20 = float(r23["lead20_rate_rank_pct"].iloc[0]) if len(r23) else np.nan
    rank25 = float(r23["lead25_rate_rank_pct"].iloc[0]) if len(r23) else np.nan
    broad_2023_extreme = int((np.isfinite(z20) and abs(z20) >= 1.5) or (np.isfinite(z25) and abs(z25) >= 1.5))

    allsig = signals.loc[signals["scope"].eq("all_2023_2025")].copy()
    best_actual = allsig.loc[allsig["spearman_vs_actual20"].abs().idxmax()] if len(allsig) else None
    best_cal = allsig.loc[allsig["spearman_vs_calibration_error20"].abs().idxmax()] if len(allsig) else None
    best_actual_corr = float(best_actual["spearman_vs_actual20"]) if best_actual is not None else np.nan
    best_cal_corr = float(best_cal["spearman_vs_calibration_error20"]) if best_cal is not None else np.nan
    best_actual_feat = str(best_actual["feature"]) if best_actual is not None else ""
    best_cal_feat = str(best_cal["feature"]) if best_cal is not None else ""

    exact_late = scope.loc[scope["scope"].isin(["2023_w13_18", "2024_w13_18", "2025_w13_18"])].copy()
    rate_range = float(exact_late["actual20_rate"].max() - exact_late["actual20_rate"].min()) if len(exact_late) else np.nan
    cal_range = float(exact_late["calibration_gap_actual_minus_pred"].max() - exact_late["calibration_gap_actual_minus_pred"].min()) if len(exact_late) else np.nan
    nonstationarity = int((np.isfinite(rate_range) and rate_range >= 0.08) or (np.isfinite(cal_range) and cal_range >= 0.10))
    pregame_regime_structure = int((np.isfinite(best_actual_corr) and abs(best_actual_corr) >= 0.15) or (np.isfinite(best_cal_corr) and abs(best_cal_corr) >= 0.15))

    return pd.DataFrame([{
        "m95p_role": "diagnostic_only_no_candidate_fit",
        "broad_history_start": min(SEASONS), "broad_history_end": max(SEASONS), "broad_history_seasons": len(SEASONS),
        "broad_census_population": "team_week_lead_rb_not_exact_stable_workhorse",
        "exact_model_trace_years": "2023|2024|2025",
        "2023_late_lead20_z": z20, "2023_late_lead25_z": z25,
        "2023_late_lead20_rank_pct": rank20, "2023_late_lead25_rank_pct": rank25,
        "2023_broad_workload_extreme_1p5sd": broad_2023_extreme,
        "exact_late_actual20_rate_range": rate_range,
        "exact_late_calibration_gap_range": cal_range,
        "stable_workhorse_nonstationarity_confirmed": nonstationarity,
        "best_pregame_feature_vs_actual20": best_actual_feat,
        "best_pregame_feature_vs_actual20_spearman": best_actual_corr,
        "best_pregame_feature_vs_calibration_gap": best_cal_feat,
        "best_pregame_feature_vs_calibration_gap_spearman": best_cal_corr,
        "pregame_regime_structure_detected_diagnostic": pregame_regime_structure,
        "feature_search": 0, "coefficient_search": 0, "new_model_fit": 0,
        "sportsbook_inputs": 0, "production_change": 0,
        "recommendation": "expand_temporal_backtesting_and_build_precommitted_dynamic_prior_candidate" if pregame_regime_structure else "expand_temporal_backtesting_before_new_candidate",
        "disposition": "M95P_AUDIT_COMPLETE_NO_PRODUCTION_CHANGE",
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m95g-root", type=Path, required=True)
    ap.add_argument("--m95k-root", type=Path, required=True)
    ap.add_argument("--m95l-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    census, weekly = build_team_week_census()
    census, weekly = add_pregame_regime_features(census, weekly)
    broad = broad_season_summary(census)

    exact = exact_model_trace(args.m95g_root, args.m95k_root, args.m95l_root)
    join_cols = [c for c in census.columns if c not in {"lead20", "lead25"}]
    exact = exact.merge(census[join_cols], on=TEAM_KEYS, how="left", validate="many_to_one")
    regime_cols = [c for c in exact.columns if c.startswith("league_") or c.startswith("team_l4_")]
    join_rate = float(exact[regime_cols].notna().any(axis=1).mean()) if regime_cols else 0.0
    if join_rate < 0.95:
        raise RuntimeError(f"M95P exact trace -> broad regime join coverage too low: {join_rate:.3%}")

    scope = exact_scope_summary(exact)
    signals = signal_audit(exact)
    bins = regime_bins(exact)
    disp = disposition(broad, scope, signals)

    census.to_csv(args.out_dir / "m95p_team_week_broad_census_2018_2025.csv", index=False)
    weekly.to_csv(args.out_dir / "m95p_weekly_pregame_regime_2018_2025.csv", index=False)
    broad.to_csv(args.out_dir / "m95p_broad_season_workload_context.csv", index=False)
    exact.to_csv(args.out_dir / "m95p_exact_stable_workhorse_regime_trace.csv", index=False)
    scope.to_csv(args.out_dir / "m95p_exact_scope_summary.csv", index=False)
    signals.to_csv(args.out_dir / "m95p_pregame_regime_signal_audit.csv", index=False)
    bins.to_csv(args.out_dir / "m95p_regime_quartile_diagnostic.csv", index=False)
    disp.to_csv(args.out_dir / "m95p_disposition.csv", index=False)
    pd.DataFrame([{
        "broad_seasons": "2018|2019|2020|2021|2022|2023|2024|2025",
        "broad_population": "each team-week lead RB from nflverse weekly player stats",
        "exact_population": "M95K stable-workhorse definition from comparable 2023-2025 traces",
        "pregame_rule": "all rolling league/team features shifted by >=1 completed week",
        "target_week_outcomes_in_features": 0,
        "candidate_model_fit": 0,
        "note": "broad census diagnoses era/year workload state but does not recreate exact pre-2023 M95K stable-workhorse labels",
    }]).to_csv(args.out_dir / "m95p_method_audit.csv", index=False)

    print("[m95p] broad season workload context")
    print(broad.to_string(index=False))
    print("\n[m95p] exact stable-workhorse scope summary")
    print(scope.to_string(index=False))
    print("\n[m95p] strongest pregame regime correlations")
    s = signals.loc[signals.scope.eq("all_2023_2025")].copy()
    if len(s):
        show = s.assign(abs_cal=s.spearman_vs_calibration_error20.abs()).sort_values("abs_cal", ascending=False).head(12).drop(columns="abs_cal")
        print(show.to_string(index=False))
    print("\n[m95p] regime quartiles")
    print(bins.to_string(index=False))
    print("\n[m95p] disposition")
    print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
