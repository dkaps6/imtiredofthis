#!/usr/bin/env python3
"""M95Q v3: exact M91-universe completion and provider-key reconciliation.

Runs #1-#4 established that the generalized M95A/M95B feature reconstruction
works, but Run #4 failed the 2024 parity prerequisite because it scored the
M95F tail model directly on the M95B-enriched rows.  The canonical M95E-v3 /
M95F pipeline instead starts from the complete frozen M91/M94D RB comparison
universe and left-joins M95B features, retaining low-volume RB/FB rows that do
not receive M95B enrichment.

This wrapper is mechanical only:
* uses exact M91 walk-forward component-prediction RB universes for each year;
* reconciles M91 keys to the M95B feature key by exact player display identity,
  matching M95E-v3 semantics;
* preserves the feature-trace/statistics identity rather than rewriting it to
  provider roster names;
* maps provider roster/injury/depth keys back to deterministic statistics keys
  only when the GSIS-derived alias bridge is one-to-one;
* keeps the frozen M95F 20+ Platt / 25+ football families and M95K stable mask.

No sportsbook input, model-family search, feature search, coefficient search,
or production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Import v2 for its already-vetted mechanical short-name and historical M95G
# compatibility patches, then replace only the pieces superseded below.
import scripts.backtest.evaluate_rb_m95q_expanded_historical_stable_workhorse_v2 as compat

m = compat.m


def _prediction_files(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in root.rglob("component_predictions.csv"):
        try:
            z = pd.read_csv(p, usecols=["season"], nrows=5)
        except Exception:
            continue
        vals = pd.to_numeric(z["season"], errors="coerce").dropna().astype(int).unique()
        if len(vals) != 1:
            continue
        season = int(vals[0])
        if season in out:
            raise RuntimeError(f"M95Q duplicate M91 component_predictions for season {season}: {out[season]} and {p}")
        out[season] = p
    return out


def load_m91_rb_universe(root: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    files = _prediction_files(root)
    rows = []
    for season in seasons:
        if season not in files:
            raise RuntimeError(f"M95Q missing reconstructed M91 component_predictions for {season} under {root}")
        x = m.lower(pd.read_csv(files[season], low_memory=False))
        need = set(m.PLAYER_KEYS + ["player", "position", "market", "actual"])
        miss = sorted(need - set(x.columns))
        if miss:
            raise RuntimeError(f"M95Q M91 {season} predictions missing {miss}")
        x = x.loc[
            x["market"].astype(str).str.lower().eq("rush_att")
            & x["position"].astype(str).str.upper().eq("RB")
            & m.num(x["week"]).between(1, 18)
        ].copy()
        x["season"] = m.num(x["season"]).astype(int)
        x["week"] = m.num(x["week"]).astype(int)
        x["team"] = x["team"].map(m.g.canon)
        x["actual_carries"] = m.num(x["actual"])
        x = x[m.PLAYER_KEYS + ["player", "actual_carries"]].drop_duplicates(m.PLAYER_KEYS)
        rows.append(x)
        print(f"[m95q-v3] M91 universe season={season} rows={len(x)}")
    return pd.concat(rows, ignore_index=True, sort=False)


def complete_feature_trace(feature_trace: pd.DataFrame, m91_rb: pd.DataFrame) -> pd.DataFrame:
    """M95E-v3-style complete workload frame using M91 universe as the base."""
    feat = feature_trace.copy()
    feat["team"] = feat["team"].map(m.g.canon)

    # Exact display-identity reconciliation used by M95E-v3.  This retains the
    # canonical M95B/statistics key whenever an enriched row exists.
    keymap = feat[["season", "week", "team", "player", "player_clean_key"]].drop_duplicates(
        ["season", "week", "team", "player"]
    ).rename(columns={"player_clean_key": "_feature_player_key"})
    rb = m91_rb.copy().merge(
        keymap, on=["season", "week", "team", "player"], how="left", validate="many_to_one"
    )
    rb["player_clean_key"] = rb["_feature_player_key"].fillna(rb["player"].map(m.g.norm_name))
    rb = rb.drop(columns=["_feature_player_key"])
    if rb.duplicated(m.PLAYER_KEYS).any():
        bad = rb.loc[rb.duplicated(m.PLAYER_KEYS, keep=False), m.PLAYER_KEYS + ["player"]].head(20)
        raise RuntimeError(f"M95Q completed M91 universe produced duplicate keys: {bad.to_dict('records')}")

    # The base owns truth and population.  M95B contributes pregame features.
    feat_cols = [c for c in feat.columns if c not in {"actual_carries", "player"}]
    out = rb.merge(feat[feat_cols], on=m.PLAYER_KEYS, how="left", validate="one_to_one")
    out["actual_20plus"] = m.num(out["actual_carries"]).ge(20).astype(int)
    out["actual_25plus"] = m.num(out["actual_carries"]).ge(25).astype(int)
    out = m.e.add_priors(out)
    return out.reset_index(drop=True)


def _reverse_alias_lut(aliases: pd.DataFrame) -> dict[tuple[int, str], str]:
    if aliases.empty:
        return {}
    a = aliases[["season", "stat_name_key", "provider_name_key"]].dropna().drop_duplicates().copy()
    counts = a.groupby(["season", "provider_name_key"])["stat_name_key"].nunique()
    good = set(counts[counts.eq(1)].index.tolist())
    return {
        (int(r.season), str(r.provider_name_key)): str(r.stat_name_key)
        for r in a.itertuples(index=False)
        if (int(r.season), str(r.provider_name_key)) in good
    }


def bridge_provider_to_stats(x: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    if z.empty or "player_clean_key" not in z.columns or "season" not in z.columns:
        return z
    lut = _reverse_alias_lut(aliases)
    if not lut:
        return z
    z["player_clean_key"] = [
        lut.get((int(s), str(pk)), str(pk))
        for s, pk in zip(m.num(z["season"]).astype(int), z["player_clean_key"].astype(str))
    ]
    return z


def enrich_stable_complete(trace: pd.DataFrame, hold_scored: pd.DataFrame, seasons: list[int], aliases: pd.DataFrame):
    g, k = m.g, m.k
    rosters, injuries, depth, provider_audit = g.load_provider_sources(seasons)
    rosters = bridge_provider_to_stats(rosters, aliases)
    injuries = bridge_provider_to_stats(injuries, aliases)
    depth = bridge_provider_to_stats(depth, aliases)
    rosters = g.add_roster_transition_features(rosters)
    if hasattr(g, "add_depth_transition_features"):
        depth = g.add_depth_transition_features(depth)

    base = hold_scored.copy()
    role_trace = trace.loc[trace["season"].isin(seasons)].copy()
    cov = base[m.PLAYER_KEYS].merge(
        rosters[m.PLAYER_KEYS + ["self_roster_present"]].drop_duplicates(m.PLAYER_KEYS),
        on=m.PLAYER_KEYS, how="left",
    )
    coverage = cov.groupby("season")["self_roster_present"].agg(
        rows="size", roster_matches=lambda s: int(m.num(s).fillna(0).gt(0).sum())
    ).reset_index()
    coverage["roster_join_rate"] = coverage["roster_matches"] / coverage["rows"]

    z = g.enrich_base(base, role_trace, rosters, injuries, depth)
    z["stable_workhorse_m95k"] = k.stable_workhorse(z).astype(int)
    z["actual_20plus"] = m.num(z["actual_carries"]).ge(20).astype(int)
    z["actual_25plus"] = m.num(z["actual_carries"]).ge(25).astype(int)
    return z, provider_audit, coverage


def parity_audit_v3(recon: pd.DataFrame, exact_root: Path) -> pd.DataFrame:
    exact = m.lower(pd.read_csv(m.find_one(exact_root, "m95g_2024_holdout_trace.csv"), low_memory=False))
    exact = exact.loc[m.num(exact["season"]).eq(m.PARITY_SEASON) & m.num(exact["week"]).between(13, 18)].copy()
    exact["team"] = exact["team"].map(m.g.canon)
    # Exact M95G keys already represent the M95E/M95F canonical statistics key.
    exact["player_clean_key"] = exact["player_clean_key"].astype(str).map(m.g.norm_name)
    if "stable_workhorse_m95k" not in exact.columns:
        exact["stable_workhorse_m95k"] = m.k.stable_workhorse(exact).astype(int)
    rr = recon.loc[m.num(recon["season"]).eq(m.PARITY_SEASON)].copy()
    keep = m.PLAYER_KEYS + ["role_is_workhorse", "stable_workhorse_m95k", "cal_prob_20", "cal_prob_25"]
    ex = exact[[c for c in keep if c in exact.columns]].drop_duplicates(m.PLAYER_KEYS)
    rc = rr[[c for c in keep if c in rr.columns]].drop_duplicates(m.PLAYER_KEYS)
    z = ex.merge(rc, on=m.PLAYER_KEYS, how="outer", suffixes=("_exact", "_recon"), indicator=True)
    both = z.loc[z["_merge"].eq("both")].copy()
    exact_n, recon_n, overlap = len(ex), len(rc), len(both)
    role = pd.DataFrame({"a": m.num(both.get("role_is_workhorse_exact", np.nan)), "b": m.num(both.get("role_is_workhorse_recon", np.nan))}).dropna()
    stable = pd.DataFrame({"a": m.num(both.get("stable_workhorse_m95k_exact", np.nan)), "b": m.num(both.get("stable_workhorse_m95k_recon", np.nan))}).dropna()
    role_agree = float(role.a.eq(role.b).mean()) if len(role) else np.nan
    stable_agree = float(stable.a.eq(stable.b).mean()) if len(stable) else np.nan
    p20 = pd.DataFrame({"a": m.num(both.get("cal_prob_20_exact", np.nan)), "b": m.num(both.get("cal_prob_20_recon", np.nan))}).dropna()
    p25 = pd.DataFrame({"a": m.num(both.get("cal_prob_25_exact", np.nan)), "b": m.num(both.get("cal_prob_25_recon", np.nan))}).dropna()
    p20_corr = float(p20.a.corr(p20.b)) if len(p20) > 2 else np.nan
    p20_mae = float((p20.a-p20.b).abs().mean()) if len(p20) else np.nan
    p25_corr = float(p25.a.corr(p25.b)) if len(p25) > 2 else np.nan
    p25_mae = float((p25.a-p25.b).abs().mean()) if len(p25) else np.nan
    parity_pass = int(
        overlap / max(exact_n, 1) >= 0.95
        and role_agree >= 0.98
        and stable_agree >= 0.95
        and np.isfinite(p20_corr) and p20_corr >= 0.90
        and np.isfinite(p20_mae) and p20_mae <= 0.05
    )
    return pd.DataFrame([{
        "exact_rows": exact_n, "recon_rows": recon_n, "overlap_rows": overlap,
        "overlap_vs_exact": overlap/max(exact_n,1), "role_workhorse_agreement": role_agree,
        "stable_mask_agreement": stable_agree, "p20_corr": p20_corr, "p20_mae": p20_mae,
        "p25_corr": p25_corr, "p25_mae": p25_mae, "parity_pass": parity_pass,
        "exact_only_rows": int((z["_merge"]=="left_only").sum()),
        "recon_only_rows": int((z["_merge"]=="right_only").sum()),
    }])


def m91_parity_audit(rebuilt_root: Path, exact_root: Path) -> pd.DataFrame:
    rb = load_m91_rb_universe(rebuilt_root, (m.PARITY_SEASON,))
    ex = load_m91_rb_universe(exact_root, (m.PARITY_SEASON,))
    a = ex[m.PLAYER_KEYS + ["actual_carries"]].copy()
    b = rb[m.PLAYER_KEYS + ["actual_carries"]].copy()
    z = a.merge(b, on=m.PLAYER_KEYS, how="outer", suffixes=("_exact", "_rebuild"), indicator=True)
    both = z.loc[z["_merge"].eq("both")].copy()
    truth = pd.DataFrame({"a": m.num(both["actual_carries_exact"]), "b": m.num(both["actual_carries_rebuild"])}).dropna()
    exact_n = len(a); overlap = len(both)
    truth_mae = float((truth.a-truth.b).abs().mean()) if len(truth) else np.nan
    # This is an added reconstruction gate, not a relaxation of the existing Q parity gate.
    passed = int(overlap/max(exact_n,1) >= 0.995 and truth_mae <= 1e-9)
    return pd.DataFrame([{
        "exact_2024_m91_rb_rows": exact_n, "rebuilt_2024_m91_rb_rows": len(b),
        "overlap_rows": overlap, "overlap_vs_exact": overlap/max(exact_n,1),
        "truth_carry_mae": truth_mae, "m91_universe_parity_pass": passed,
        "exact_only_rows": int((z["_merge"]=="left_only").sum()),
        "rebuild_only_rows": int((z["_merge"]=="right_only").sum()),
    }])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pbp-root", type=Path, required=True)
    p.add_argument("--pfr-root", type=Path, required=True)
    p.add_argument("--ngs-file", type=Path, required=True)
    p.add_argument("--m95g-root", type=Path, required=True)
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--m91-exact-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logs, stat_ids = m.load_weekly_logs()
    aliases = m.build_alias_map(stat_ids)
    feature_trace, _ = m.build_matchup_trace(logs, args.pbp_root, args.pfr_root, args.ngs_file)
    # Important: do NOT rewrite the canonical M95B/statistics keys to provider names.
    universe = load_m91_rb_universe(args.m91_root, m.TRACE_SEASONS)
    trace = complete_feature_trace(feature_trace, universe)

    scored_parts, feat_parts = [], []
    for season in [*m.REPORT_SEASONS, m.PARITY_SEASON]:
        scored, feats = m.score_rotation(trace, season)
        scored_parts.append(scored); feat_parts.append(feats)
        print(f"[m95q-v3] scored season={season} hold_rows={len(scored)}")
    scored = pd.concat(scored_parts, ignore_index=True, sort=False)
    feature_audit = pd.concat(feat_parts, ignore_index=True, sort=False)

    enriched, provider_audit, roster_coverage = enrich_stable_complete(
        trace, scored, [*m.REPORT_SEASONS, m.PARITY_SEASON], aliases
    )
    summary = m.season_summary(enriched)
    parity = parity_audit_v3(enriched, args.m95g_root)
    m91_parity = m91_parity_audit(args.m91_root, args.m91_exact_root)

    src_rows = []
    parity_pass = int(parity.iloc[0]["parity_pass"] and m91_parity.iloc[0]["m91_universe_parity_pass"])
    f24 = int(feature_audit.loc[(feature_audit["season"]==m.PARITY_SEASON) & (feature_audit["target"]=="actual_20plus"), "raw_feature_count"].iloc[0])
    for season in m.REPORT_SEASONS:
        cov = roster_coverage.loc[roster_coverage["season"].eq(season)]
        roster_rate = float(cov.iloc[0]["roster_join_rate"]) if len(cov) else 0.0
        fs = feature_audit.loc[(feature_audit["season"]==season) & (feature_audit["target"]=="actual_20plus")]
        feat_count = int(fs.iloc[0]["raw_feature_count"]) if len(fs) else 0
        sm = summary.loc[summary["season"].eq(season)]
        stable_n = int(sm.iloc[0]["stable_n"]) if len(sm) else 0
        comparable = int(parity_pass and roster_rate >= 0.95 and feat_count >= max(8, int(np.floor(0.70*f24))) and stable_n >= 15)
        src_rows.append({"season":season,"roster_join_rate":roster_rate,"raw20_feature_count":feat_count,
                         "feature_count_vs_2024":feat_count/max(f24,1),"stable_n":stable_n,
                         "parity_prerequisite_pass":parity_pass,"historical_season_comparable":comparable})
    comparability = pd.DataFrame(src_rows)
    usable = int(comparability["historical_season_comparable"].sum())
    disposition = pd.DataFrame([{
        "m95q_role":"expanded_exact_historical_reconstruction",
        "new_exact_years_attempted":len(m.REPORT_SEASONS),"new_exact_years_comparable":usable,
        "parity_2024_pass":int(parity.iloc[0]["parity_pass"]),
        "m91_universe_2024_pass":int(m91_parity.iloc[0]["m91_universe_parity_pass"]),
        "feature_search":0,"coefficient_search":0,"sportsbook_inputs":0,"production_change":0,
        "disposition":"M95Q_EXPANDED_PANEL_READY" if usable>=2 and parity_pass else "M95Q_RECONSTRUCTION_NOT_YET_COMPARABLE",
    }])

    feature_trace.to_csv(args.out_dir/"m95q_m95b_feature_trace_2019_2024.csv", index=False)
    trace.to_csv(args.out_dir/"m95q_completed_matchup_trace_2019_2024.csv", index=False)
    scored.to_csv(args.out_dir/"m95q_rotated_m95f_holdouts.csv", index=False)
    enriched.to_csv(args.out_dir/"m95q_enriched_holdouts.csv", index=False)
    summary.to_csv(args.out_dir/"m95q_stable_workhorse_summary.csv", index=False)
    feature_audit.to_csv(args.out_dir/"m95q_feature_audit.csv", index=False)
    provider_audit.to_csv(args.out_dir/"m95q_provider_source_audit.csv", index=False)
    roster_coverage.to_csv(args.out_dir/"m95q_roster_join_coverage.csv", index=False)
    aliases.to_csv(args.out_dir/"m95q_identity_alias_bridge.csv", index=False)
    parity.to_csv(args.out_dir/"m95q_2024_parity_audit.csv", index=False)
    m91_parity.to_csv(args.out_dir/"m95q_2024_m91_universe_parity.csv", index=False)
    comparability.to_csv(args.out_dir/"m95q_historical_comparability.csv", index=False)
    disposition.to_csv(args.out_dir/"m95q_disposition.csv", index=False)

    print("\n[m95q-v3] M91 2024 universe parity\n", m91_parity.to_string(index=False))
    print("\n[m95q-v3] downstream 2024 parity\n", parity.to_string(index=False))
    print("\n[m95q-v3] historical stable summary\n", summary.to_string(index=False))
    print("\n[m95q-v3] comparability\n", comparability.to_string(index=False))
    print("\n[m95q-v3] disposition\n", disposition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
