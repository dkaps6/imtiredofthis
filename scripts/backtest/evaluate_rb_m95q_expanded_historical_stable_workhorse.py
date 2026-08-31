#!/usr/bin/env python3
"""M95Q: expanded historical stable-workhorse backtest reconstruction.

Research-only. Reconstructs the M95A/M95B feature trace and the frozen M95F
calibration protocol on earlier temporal rotations, then applies the frozen
M95K stable-workhorse definition using M95G pregame roster/injury/depth
semantics. The primary new evaluation years are 2020-2022, with 2024 rebuilt
as a mechanical parity control against the authoritative M95G trace.

No sportsbook input. No production change. No model-family/feature/coefficient
search. M95F calibration families are frozen: 20+ Platt, 25+ football.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.backtest.evaluate_rb_matchup_truth as a
import scripts.backtest.evaluate_rb_offense_defense_matchup as b
import scripts.backtest.evaluate_rb_absolute_workload_distribution as e
import scripts.backtest.evaluate_rb_workload_regime_calibration as f
import scripts.backtest.evaluate_rb_role_availability as g
import scripts.backtest.evaluate_rb_feed_tendency_carry_ceiling as k
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

REPORT_SEASONS = (2020, 2021, 2022)
PARITY_SEASON = 2024
TRACE_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
LOG_SEASONS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
PBP_SEASONS = LOG_SEASONS
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]
CAL_FAMILY = {"actual_20plus": "platt", "actual_25plus": "football"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"M95Q expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def load_weekly_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    frames = []
    id_rows = []
    for season in LOG_SEASONS:
        raw = _to_pandas(nfl.load_player_stats(seasons=[int(season)], summary_level="week"))
        if raw.empty:
            raise RuntimeError(f"M95Q weekly stats empty for {season}")
        raw_l = lower(raw)
        id_col = next((c for c in ["player_id", "gsis_id", "player_gsis_id"] if c in raw_l.columns), None)
        name_col = next((c for c in ["player_display_name", "player_name", "display_name", "name"] if c in raw_l.columns), None)
        if id_col and name_col:
            tmp = raw_l[[id_col, name_col]].dropna().drop_duplicates().copy()
            tmp["season"] = season
            tmp["stat_player_id"] = tmp[id_col].astype(str).str.strip()
            tmp["stat_name_key"] = tmp[name_col].map(g.norm_name)
            id_rows.append(tmp[["season", "stat_player_id", "stat_name_key"]])

        z = _normalize_weekly(raw, season)
        z = z.loc[num(z["week"]).between(1, 18)].copy()
        out = pd.DataFrame({
            "season": season,
            "week": num(z["week"]).astype(int),
            "team": z["team"].map(g.canon),
            "player": z["player"].astype(str),
            "position": z["position"].astype(str).str.upper().str.strip(),
            "rushes": num(z["rushes"]).fillna(0.0),
            "rush_yards": num(z["rush_yards"]).fillna(0.0),
            "targets": num(z["targets"]).fillna(0.0),
            "receptions": num(z["receptions"]).fillna(0.0),
            "rec_yards": num(z["rec_yards"]).fillna(0.0),
        })
        out["player_clean_key"] = out["player"].map(g.norm_name)
        out = out.loc[out["team"].astype(str).ne("") & out["player_clean_key"].ne("")].copy()
        frames.append(out)
        print(f"[m95q] weekly season={season} rows={len(out)}")
    logs = pd.concat(frames, ignore_index=True, sort=False)
    logs = logs.sort_values(["season", "week", "team", "player_clean_key"]).drop_duplicates(
        ["season", "week", "team", "player_clean_key"], keep="last"
    )
    ids = pd.concat(id_rows, ignore_index=True).drop_duplicates() if id_rows else pd.DataFrame()
    return logs.reset_index(drop=True), ids


def build_alias_map(stat_ids: pd.DataFrame) -> pd.DataFrame:
    import nflreadpy as nfl

    rows = []
    if stat_ids.empty:
        return pd.DataFrame(columns=["season", "stat_name_key", "provider_name_key"])
    for season in TRACE_SEASONS:
        try:
            r = lower(_to_pandas(nfl.load_rosters_weekly(int(season))))
        except Exception:
            continue
        rid = next((c for c in ["gsis_id", "player_id", "player_gsis_id"] if c in r.columns), None)
        rn = next((c for c in ["full_name", "football_name", "player_name", "player", "name"] if c in r.columns), None)
        if not rid or not rn:
            continue
        rr = r[[rid, rn]].dropna().drop_duplicates().copy()
        rr["stat_player_id"] = rr[rid].astype(str).str.strip()
        rr["provider_name_key"] = rr[rn].map(g.norm_name)
        ss = stat_ids.loc[stat_ids["season"].eq(season), ["season", "stat_player_id", "stat_name_key"]].drop_duplicates()
        m = ss.merge(rr[["stat_player_id", "provider_name_key"]], on="stat_player_id", how="inner")
        rows.append(m[["season", "stat_name_key", "provider_name_key"]])
    if not rows:
        return pd.DataFrame(columns=["season", "stat_name_key", "provider_name_key"])
    out = pd.concat(rows, ignore_index=True).dropna().drop_duplicates()
    # Require a deterministic one-to-one alias per season/stat key.
    counts = out.groupby(["season", "stat_name_key"])["provider_name_key"].nunique()
    good = counts[counts.eq(1)].reset_index()[["season", "stat_name_key"]]
    return out.merge(good, on=["season", "stat_name_key"], how="inner").drop_duplicates(["season", "stat_name_key"])


def apply_aliases(x: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    if aliases.empty:
        return z
    lut = aliases.set_index(["season", "stat_name_key"])["provider_name_key"].to_dict()
    z["player_clean_key"] = [
        lut.get((int(s), str(pk)), str(pk)) for s, pk in zip(num(z["season"]).astype(int), z["player_clean_key"].astype(str))
    ]
    if z.duplicated(PLAYER_KEYS).any():
        dup = z.loc[z.duplicated(PLAYER_KEYS, keep=False), PLAYER_KEYS].head(10).to_dict("records")
        raise RuntimeError(f"M95Q alias reconciliation created duplicate player keys: {dup}")
    return z


def build_matchup_trace(logs: pd.DataFrame, pbp_root: Path, pfr_root: Path, ngs_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a.TARGET_SEASONS = TRACE_SEASONS
    a.PBP_SEASONS = PBP_SEASONS
    b.TARGET_SEASONS = TRACE_SEASONS
    b.PBP_SEASONS = PBP_SEASONS

    apbp = a._read_pbp(pbp_root)
    schedule = a._schedule_from_pbp(apbp)
    team_games = a._team_game_from_logs(logs)
    rb_games = a._player_prior_features(logs, team_games)
    rb_allowed = a._rb_allowed_games(rb_games, schedule)
    pbp_def = a._pbp_defense_games(apbp)
    defense_games = pbp_def.merge(rb_allowed, on=["season", "week", "defense"], how="outer", validate="one_to_one")
    metric_cols = [c for c in defense_games.columns if c not in {"season", "week", "defense"}]
    profiles = a._rolling_defense_profiles(defense_games, schedule, metric_cols)
    profiles = a._add_defense_composite(profiles)
    trace = a._truth_trace(rb_games, schedule, profiles)
    trace["player_clean_key"] = trace["player"].map(g.norm_name)

    bpbp = b.read_pbp(pbp_root)
    pfr = b.read_pfr(pfr_root)
    ngs = b.read_ngs(ngs_file)
    x = b.add_offense(trace, bpbp, pfr, ngs)
    x = b.add_scores(x)
    x = e.add_priors(x)
    x["actual_20plus"] = num(x["actual_carries"]).ge(20).astype(int)
    x["actual_25plus"] = num(x["actual_carries"]).ge(25).astype(int)
    x["team"] = x["team"].map(g.canon)
    return x.reset_index(drop=True), profiles


def temporal_oof_for_rotation(trace: pd.DataFrame, season: int, target: str) -> pd.DataFrame:
    pieces = []
    prior_season = season - 1
    for week in range(5, 13):
        tr = trace.loc[
            trace["season"].eq(prior_season)
            | (trace["season"].eq(season) & num(trace["week"]).lt(week))
        ].copy()
        te = trace.loc[trace["season"].eq(season) & num(trace["week"]).eq(week)].copy()
        if te.empty or tr[target].nunique() < 2:
            continue
        q = te.copy()
        q["raw_score"] = f.raw_tail_score(tr, te, target)
        q["actual_label"] = q[target].astype(int)
        pieces.append(q)
    if not pieces:
        raise RuntimeError(f"M95Q no OOF calibration rows for season={season} target={target}")
    return pd.concat(pieces, ignore_index=True, sort=False)


def score_rotation(trace: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = trace.loc[trace["season"].eq(season) & num(trace["week"]).between(13, 18)].copy()
    if hold.empty:
        raise RuntimeError(f"M95Q empty holdout season {season}")
    feature_rows = []
    for target, outcol in [("actual_20plus", "cal_prob_20"), ("actual_25plus", "cal_prob_25")]:
        prior = season - 1
        train = trace.loc[
            trace["season"].eq(prior)
            | (trace["season"].eq(season) & num(trace["week"]).le(12))
        ].copy()
        feats = e.available(train, e.TAIL_FEATURES)
        if not feats:
            raise RuntimeError(f"M95Q no M95F tail features season={season} target={target}")
        raw = f.raw_tail_score(train, hold, target)
        oof = temporal_oof_for_rotation(trace, season, target)
        cal = f.fit_calibrator(oof, CAL_FAMILY[target])
        zz = hold.copy()
        zz["raw_score"] = raw
        zz["actual_label"] = zz[target].astype(int)
        hold[outcol] = f.apply_calibrator(cal, zz, CAL_FAMILY[target])
        feature_rows.append({
            "season": season, "target": target, "calibration_family": CAL_FAMILY[target],
            "raw_feature_count": len(feats), "raw_features": "|".join(feats),
            "oof_rows": len(oof), "hold_rows": len(hold),
        })
    hold["cal_prob_25"] = np.minimum(num(hold["cal_prob_25"]), num(hold["cal_prob_20"]))
    return hold, pd.DataFrame(feature_rows)


def enrich_stable(trace: pd.DataFrame, hold_scored: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rosters, injuries, depth, provider_audit = g.load_provider_sources(seasons)
    rosters = g.add_roster_transition_features(rosters)
    if hasattr(g, "add_depth_transition_features"):
        depth = g.add_depth_transition_features(depth)

    base = hold_scored.copy()
    role_trace = trace.loc[trace["season"].isin(seasons)].copy()
    # Direct roster coverage before M95G's intentional missing->unavailable logic.
    cov = base[PLAYER_KEYS].merge(
        rosters[PLAYER_KEYS + ["self_roster_present"]].drop_duplicates(PLAYER_KEYS),
        on=PLAYER_KEYS, how="left"
    )
    coverage = (
        cov.groupby("season", as_index=False)["self_roster_present"]
        .agg(rows="size", roster_matches=lambda s: int(num(s).fillna(0).gt(0).sum()))
    )
    coverage["roster_join_rate"] = coverage["roster_matches"] / coverage["rows"]

    z = g.enrich_base(base, role_trace, rosters, injuries, depth)
    z["stable_workhorse_m95k"] = k.stable_workhorse(z).astype(int)
    z["actual_20plus"] = num(z["actual_carries"]).ge(20).astype(int)
    z["actual_25plus"] = num(z["actual_carries"]).ge(25).astype(int)
    return z, provider_audit, coverage


def prob_metrics(y, p) -> dict:
    return k.prob_metrics(y, p)


def season_summary(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, q0 in z.groupby("season"):
        q = q0.loc[num(q0["stable_workhorse_m95k"]).eq(1)].copy()
        if q.empty:
            continue
        m20 = prob_metrics(q["actual_20plus"], q["cal_prob_20"])
        m25 = prob_metrics(q["actual_25plus"], q["cal_prob_25"])
        rows.append({
            "season": int(season), "stable_n": len(q), "unique_players": q["player_clean_key"].nunique(),
            "actual20_rate": float(q["actual_20plus"].mean()), "actual25_rate": float(q["actual_25plus"].mean()),
            "mean_carries": float(num(q["actual_carries"]).mean()),
            "m95f_p20_mean": m20["mean_prob"], "m95f_p20_auc": m20["auc"], "m95f_p20_brier": m20["brier"],
            "m95f_p25_mean": m25["mean_prob"], "m95f_p25_auc": m25["auc"], "m95f_p25_brier": m25["brier"],
            "p20_calibration_gap_actual_minus_pred": float(q["actual_20plus"].mean() - m20["mean_prob"]),
        })
    return pd.DataFrame(rows)


def parity_audit(recon: pd.DataFrame, exact_root: Path, aliases: pd.DataFrame) -> pd.DataFrame:
    exact = lower(pd.read_csv(find_one(exact_root, "m95g_2024_holdout_trace.csv"), low_memory=False))
    exact = exact.loc[num(exact["season"]).eq(PARITY_SEASON) & num(exact["week"]).between(13, 18)].copy()
    exact["team"] = exact["team"].map(g.canon)
    exact["player_clean_key"] = exact["player_clean_key"].astype(str).map(g.norm_name)
    exact = apply_aliases(exact, aliases)
    if "stable_workhorse_m95k" not in exact.columns:
        exact["stable_workhorse_m95k"] = k.stable_workhorse(exact).astype(int)
    rr = recon.loc[num(recon["season"]).eq(PARITY_SEASON)].copy()
    keep = PLAYER_KEYS + ["role_is_workhorse", "stable_workhorse_m95k", "cal_prob_20", "cal_prob_25"]
    ex = exact[[c for c in keep if c in exact.columns]].drop_duplicates(PLAYER_KEYS)
    rc = rr[[c for c in keep if c in rr.columns]].drop_duplicates(PLAYER_KEYS)
    m = ex.merge(rc, on=PLAYER_KEYS, how="outer", suffixes=("_exact", "_recon"), indicator=True)
    both = m.loc[m["_merge"].eq("both")].copy()
    exact_n = len(ex); recon_n = len(rc); overlap = len(both)
    role_agree = float((num(both.get("role_is_workhorse_exact", np.nan)) == num(both.get("role_is_workhorse_recon", np.nan))).mean()) if overlap else np.nan
    stable_agree = float((num(both.get("stable_workhorse_m95k_exact", np.nan)) == num(both.get("stable_workhorse_m95k_recon", np.nan))).mean()) if overlap else np.nan
    p20 = pd.DataFrame({"a": num(both.get("cal_prob_20_exact", np.nan)), "b": num(both.get("cal_prob_20_recon", np.nan))}).dropna()
    p25 = pd.DataFrame({"a": num(both.get("cal_prob_25_exact", np.nan)), "b": num(both.get("cal_prob_25_recon", np.nan))}).dropna()
    p20_corr = float(p20.a.corr(p20.b)) if len(p20) > 2 else np.nan
    p20_mae = float((p20.a - p20.b).abs().mean()) if len(p20) else np.nan
    p25_corr = float(p25.a.corr(p25.b)) if len(p25) > 2 else np.nan
    p25_mae = float((p25.a - p25.b).abs().mean()) if len(p25) else np.nan
    parity_pass = int(
        overlap / max(exact_n, 1) >= 0.95
        and role_agree >= 0.98
        and stable_agree >= 0.95
        and np.isfinite(p20_corr) and p20_corr >= 0.90
        and np.isfinite(p20_mae) and p20_mae <= 0.05
    )
    return pd.DataFrame([{
        "exact_rows": exact_n, "recon_rows": recon_n, "overlap_rows": overlap,
        "overlap_vs_exact": overlap / max(exact_n, 1),
        "role_workhorse_agreement": role_agree, "stable_mask_agreement": stable_agree,
        "p20_corr": p20_corr, "p20_mae": p20_mae, "p25_corr": p25_corr, "p25_mae": p25_mae,
        "parity_pass": parity_pass,
    }])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pbp-root", type=Path, required=True)
    p.add_argument("--pfr-root", type=Path, required=True)
    p.add_argument("--ngs-file", type=Path, required=True)
    p.add_argument("--m95g-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logs, stat_ids = load_weekly_logs()
    aliases = build_alias_map(stat_ids)
    trace, profiles = build_matchup_trace(logs, args.pbp_root, args.pfr_root, args.ngs_file)
    trace = apply_aliases(trace, aliases)

    scored_parts = []
    feat_parts = []
    for season in [*REPORT_SEASONS, PARITY_SEASON]:
        scored, feats = score_rotation(trace, season)
        scored_parts.append(scored); feat_parts.append(feats)
        print(f"[m95q] scored season={season} hold_rows={len(scored)}")
    scored = pd.concat(scored_parts, ignore_index=True, sort=False)
    feature_audit = pd.concat(feat_parts, ignore_index=True, sort=False)

    enriched, provider_audit, roster_coverage = enrich_stable(trace, scored, [*REPORT_SEASONS, PARITY_SEASON])
    summary = season_summary(enriched)
    parity = parity_audit(enriched, args.m95g_root, aliases)

    # Source/comparability gate for each new historical season. 2024 parity is
    # a global mechanical prerequisite; a season also needs strong roster
    # coverage and a nontrivial stable cohort.
    src_rows = []
    parity_pass = int(parity.iloc[0]["parity_pass"])
    f24 = int(feature_audit.loc[feature_audit["season"].eq(PARITY_SEASON) & feature_audit["target"].eq("actual_20plus"), "raw_feature_count"].iloc[0])
    for season in REPORT_SEASONS:
        cov = roster_coverage.loc[roster_coverage["season"].eq(season)]
        roster_rate = float(cov.iloc[0]["roster_join_rate"]) if len(cov) else 0.0
        fs = feature_audit.loc[feature_audit["season"].eq(season) & feature_audit["target"].eq("actual_20plus")]
        feat_count = int(fs.iloc[0]["raw_feature_count"]) if len(fs) else 0
        sm = summary.loc[summary["season"].eq(season)]
        stable_n = int(sm.iloc[0]["stable_n"]) if len(sm) else 0
        comparable = int(parity_pass and roster_rate >= 0.95 and feat_count >= max(8, int(np.floor(0.70 * f24))) and stable_n >= 15)
        src_rows.append({
            "season": season, "roster_join_rate": roster_rate, "raw20_feature_count": feat_count,
            "feature_count_vs_2024": feat_count / max(f24, 1), "stable_n": stable_n,
            "parity_prerequisite_pass": parity_pass, "historical_season_comparable": comparable,
        })
    comparability = pd.DataFrame(src_rows)
    usable = int(comparability["historical_season_comparable"].sum())
    disposition = pd.DataFrame([{
        "m95q_role": "expanded_exact_historical_reconstruction",
        "new_exact_years_attempted": len(REPORT_SEASONS),
        "new_exact_years_comparable": usable,
        "parity_2024_pass": parity_pass,
        "feature_search": 0, "coefficient_search": 0, "sportsbook_inputs": 0, "production_change": 0,
        "disposition": "M95Q_EXPANDED_PANEL_READY" if usable >= 2 and parity_pass else "M95Q_RECONSTRUCTION_NOT_YET_COMPARABLE",
    }])

    trace.to_csv(args.out_dir / "m95q_matchup_trace_2019_2024.csv", index=False)
    scored.to_csv(args.out_dir / "m95q_rotated_m95f_holdouts.csv", index=False)
    enriched.to_csv(args.out_dir / "m95q_enriched_holdouts.csv", index=False)
    summary.to_csv(args.out_dir / "m95q_stable_workhorse_summary.csv", index=False)
    feature_audit.to_csv(args.out_dir / "m95q_feature_audit.csv", index=False)
    provider_audit.to_csv(args.out_dir / "m95q_provider_source_audit.csv", index=False)
    roster_coverage.to_csv(args.out_dir / "m95q_roster_join_coverage.csv", index=False)
    aliases.to_csv(args.out_dir / "m95q_identity_alias_bridge.csv", index=False)
    parity.to_csv(args.out_dir / "m95q_2024_parity_audit.csv", index=False)
    comparability.to_csv(args.out_dir / "m95q_historical_comparability.csv", index=False)
    disposition.to_csv(args.out_dir / "m95q_disposition.csv", index=False)

    print("\n[m95q] 2024 parity")
    print(parity.to_string(index=False))
    print("\n[m95q] historical stable-workhorse summary")
    print(summary.to_string(index=False))
    print("\n[m95q] historical comparability")
    print(comparability.to_string(index=False))
    print("\n[m95q] disposition")
    print(disposition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
