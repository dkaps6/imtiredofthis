#!/usr/bin/env python3
"""Corrected authoritative M69 forensic entrypoint.

This entrypoint fixes implementation issues found before accepting the first
M69 output, without changing the frozen scientific thresholds:
- use the actual canonical QB attempt-share field for role attribution;
- use verified playcaller prior first-15 DBR only for opening deviation;
- attach the already-pregame M62 defensive matchup fields explicitly;
- make Raw MC error decomposition exact by retaining the point-product remainder;
- compute realized man/zone/coverage rates only over labeled snaps.

M69 remains discovery-only. No output is production-actionable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

import scripts.backtest.audit_qb_gamescript_forensic_atlas as m


DEFENSE_FEATURES = [
    "opp_coverage_man_rate",
    "opp_coverage_zone_rate",
    "opp_pressure_rate_generated",
    "opp_def_pass_epa",
    "opp_success_rate_def",
    "opponent_force_pass",
    "opp_explosive_play_rate_allowed",
]
SCHEME_CANDIDATES = DEFENSE_FEATURES + ["market_abs_spread", "market_total"]


def finite_value(row: pd.Series, *cols: str) -> float:
    for col in cols:
        v = m.num(pd.Series([row.get(col, np.nan)])).iloc[0]
        if np.isfinite(v):
            return float(v)
    return np.nan


def corrected_realized_defense(part: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    if part.empty:
        return pd.DataFrame()
    keys = m.shared_keys(part, pbp)
    if not keys:
        return pd.DataFrame()
    need = keys + [c for c in ["season", "week", "posteam", "defteam", "qb_dropback", "pass_attempt", "rush_attempt"] if c in pbp]
    right = pbp[need].drop_duplicates(keys)
    x = part.merge(right, on=keys, how="inner", suffixes=("", "_pbp"))
    if x.empty:
        return pd.DataFrame()
    team_col = "posteam" if "posteam" in x else "possession_team" if "possession_team" in x else None
    def_col = "defteam" if "defteam" in x else None
    if not team_col or not def_col:
        return pd.DataFrame()
    x["team"] = x[team_col].map(m.canon)
    x["opponent"] = x[def_col].map(m.canon)
    db = m.num(x.get("qb_dropback", 0)).fillna(0).eq(1)
    pa = m.num(x.get("pass_attempt", 0)).fillna(0).eq(1)
    ra = m.num(x.get("rush_attempt", 0)).fillna(0).eq(1)
    x = x[db | pa | ra].copy()

    man = x.get("defense_man_zone_type", pd.Series(pd.NA, index=x.index, dtype="string")).astype("string").str.upper().str.strip()
    mz_valid = man.notna() & man.ne("") & ~man.isin(["NAN", "NONE", "<NA>"])
    x["_man"] = np.nan
    x["_zone"] = np.nan
    x.loc[mz_valid, "_man"] = man.loc[mz_valid].str.contains("MAN", na=False).astype(float)
    x.loc[mz_valid, "_zone"] = man.loc[mz_valid].str.contains("ZONE", na=False).astype(float)

    cov = x.get("defense_coverage_type", pd.Series(pd.NA, index=x.index, dtype="string")).astype("string").str.upper().str.strip()
    cov_valid = cov.notna() & cov.ne("") & ~cov.isin(["NAN", "NONE", "<NA>"])
    for n in [0, 1, 2, 3, 4, 6]:
        col = f"_cover{n}"
        x[col] = np.nan
        x.loc[cov_valid, col] = cov.loc[cov_valid].str.contains(rf"(?:^|\D){n}(?:\D|$)", regex=True, na=False).astype(float)

    x["_box"] = m.num(x.get("defenders_in_box", np.nan))
    x["_rushers"] = m.num(x.get("number_of_pass_rushers", np.nan))
    x["_pressure"] = m.num(x.get("was_pressure", np.nan))

    rows = []
    for (season, week, team, opp), g in x.groupby(["season", "week", "team", "opponent"], sort=True):
        rec = {
            "season": int(season),
            "week": int(week),
            "team": m.canon(team),
            "opponent": m.canon(opp),
            "realized_def_man_rate": float(g._man.mean()) if g._man.notna().any() else np.nan,
            "realized_def_zone_rate": float(g._zone.mean()) if g._zone.notna().any() else np.nan,
            "realized_def_box_mean": float(g._box.mean()) if g._box.notna().any() else np.nan,
            "realized_def_heavy_box_rate": float(g.loc[g._box.notna(), "_box"].ge(8).mean()) if g._box.notna().any() else np.nan,
            "realized_def_light_box_rate": float(g.loc[g._box.notna(), "_box"].le(6).mean()) if g._box.notna().any() else np.nan,
            "realized_def_pass_rushers_mean": float(g._rushers.mean()) if g._rushers.notna().any() else np.nan,
            "realized_def_pressure_rate": float(g._pressure.mean()) if g._pressure.notna().any() else np.nan,
        }
        for n in [0, 1, 2, 3, 4, 6]:
            col = f"_cover{n}"
            rec[f"realized_cover{n}_rate"] = float(g[col].mean()) if g[col].notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def classify(row: pd.Series) -> str:
    att_res = row["attempt_residual_actual_minus_pred"]
    vol = row["volume_yard_contribution"]
    eff = row["efficiency_yard_contribution"]
    inter = row["interaction_yard_contribution"]
    rem = row["mc_point_product_remainder"]

    # If the MC-vs-point-product remainder itself dominates, keep it distinct.
    if np.isfinite(rem) and abs(rem) > max(abs(vol), abs(eff), abs(inter)) + 5:
        return "mc_distribution_center_remainder"
    if np.isfinite(eff) and abs(eff) > abs(vol) + 5 and abs(eff) >= abs(inter) and (not np.isfinite(rem) or abs(eff) >= abs(rem)):
        return "ypa_explosion" if eff > 0 else "ypa_collapse"

    share = finite_value(row, "m64_actual_qb_attempt_share", "actual_qb_attempt_share", "actual_attempt_share")
    if np.isfinite(share) and share < .80:
        return "role_or_participation"

    op = row["opening_deviation_vs_playcaller"]
    if np.isfinite(att_res) and att_res >= 4 and np.isfinite(op) and op >= .12:
        return "planned_pass_heavy_opening"
    if np.isfinite(att_res) and att_res <= -4 and np.isfinite(op) and op <= -.12:
        return "planned_run_heavy_opening"

    tr = row["trailing_share_surprise"]
    ld = row["leading_share_surprise"]
    if np.isfinite(att_res) and att_res >= 4 and np.isfinite(tr) and tr >= .15:
        return "forced_trailing_volume"
    if np.isfinite(att_res) and att_res <= -4 and np.isfinite(ld) and ld >= .15:
        return "leading_suppression"

    dr = row["drive_residual_actual_minus_pred"]
    if np.isfinite(att_res) and att_res >= 4 and np.isfinite(dr) and dr >= 2:
        return "possession_explosion"
    if np.isfinite(att_res) and att_res <= -4 and np.isfinite(dr) and dr <= -2:
        return "possession_collapse"

    ac = row["attempt_conversion_residual"]
    if np.isfinite(att_res) and att_res <= -4 and np.isfinite(ac) and ac <= -.08:
        return "dropback_to_attempt_conversion_loss"

    rush_epa = row["actual_rush_epa"]
    if np.isfinite(att_res) and att_res <= -4 and np.isfinite(rush_epa) and rush_epa >= .08:
        return "run_game_takeover"
    if np.isfinite(att_res) and att_res >= 4 and np.isfinite(rush_epa) and rush_epa <= -.08:
        return "run_game_failure_pass_pivot"
    return "other_volume_or_mixed"


def load_pregame_defense(paths: list[Path], seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    manifest = []
    for path in paths:
        q = m.lower(pd.read_csv(path))
        q["season"] = m.num(q.season).astype(int)
        q["week"] = m.num(q.week).astype(int)
        q["team"] = q.team.map(m.canon)
        cols = [c for c in DEFENSE_FEATURES if c in q.columns]
        manifest.append({"season": ",".join(map(str, sorted(q.season.unique()))), "family": "m62_pregame_defense", "status": f"recovered:{len(cols)}/{len(DEFENSE_FEATURES)}"})
        frames.append(q[["season", "week", "team"] + cols].copy())
    if not frames:
        raise RuntimeError("M69 corrected run requires explicit M62 pregame-defense files")
    d = pd.concat(frames, ignore_index=True, sort=False)
    d = d[d.season.isin(seasons)].copy()
    # These fields are team-week pregame values; deduplicate any QB-row repetition.
    d = d.sort_values(["season", "week", "team"]).drop_duplicates(["season", "week", "team"], keep="first")
    missing = [c for c in DEFENSE_FEATURES if c not in d.columns]
    if missing:
        raise RuntimeError(f"M69 missing frozen pregame defense fields: {missing}")
    return d, pd.DataFrame(manifest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m65-game-level", type=Path, required=True)
    ap.add_argument("--m68-features", type=Path, required=True)
    ap.add_argument("--pregame-defense-file", type=Path, action="append", required=True)
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    seasons = [int(v) for v in a.seasons.split(",") if v.strip()]

    base = m.lower(pd.read_csv(a.m65_game_level))
    new = m.lower(pd.read_csv(a.m68_features))
    for q in [base, new]:
        q["season"] = m.num(q.season).astype(int)
        q["week"] = m.num(q.week).astype(int)
        q["team"] = q.team.map(m.canon)
        if "opponent" in q:
            q["opponent"] = q.opponent.map(m.canon)
    base = base[base.season.isin(seasons)].copy()
    keep = [c for c in new.columns if c != "opponent"]
    base = base.merge(new[keep], on=["season", "week", "team"], how="left", validate="many_to_one")

    defense, defense_manifest = load_pregame_defense(a.pregame_defense_file, seasons)
    base = base.merge(defense, on=["season", "week", "team"], how="left", validate="many_to_one")
    for col in DEFENSE_FEATURES:
        coverage = float(base[col].notna().mean())
        if coverage < .95:
            raise RuntimeError(f"M69 pregame defense coverage too low for {col}: {coverage:.3f}")

    pbp, part, manifest = m.load_sources(seasons)
    manifest = pd.concat([manifest, defense_manifest], ignore_index=True, sort=False)
    pbg = m.pbp_team_games(pbp)
    rdef = corrected_realized_defense(part, pbp)
    x = base.merge(pbg, on=["season", "week", "team", "opponent"], how="left", suffixes=("", "_pbp"), validate="many_to_one")
    if not rdef.empty:
        x = x.merge(rdef, on=["season", "week", "team", "opponent"], how="left", validate="many_to_one")
    if x.actual_dropback_rate.isna().any():
        raise RuntimeError(f"M69 failed to attach PBP script to {int(x.actual_dropback_rate.isna().sum())} rows")

    x["pred_pass_yards"] = m.num(x.get("m64_pass_raw_reference", x.get("raw_pass_yards", np.nan)))
    x["actual_pass_yards"] = m.num(x.get("actual", np.nan))
    x["pred_attempts"] = m.num(x.get("attempts_raw", np.nan))
    x["actual_attempts"] = m.num(x.get("actual_pass_att", np.nan))
    x["pred_ypa"] = m.num(x.get("ypa_contextual", np.nan))
    x["actual_ypa"] = np.where(x.actual_attempts.gt(0), x.actual_pass_yards / x.actual_attempts, np.nan)
    x["pred_point_product"] = x.pred_attempts * x.pred_ypa
    x["pass_residual_actual_minus_pred"] = x.actual_pass_yards - x.pred_pass_yards
    x["attempt_residual_actual_minus_pred"] = x.actual_attempts - x.pred_attempts
    x["ypa_residual_actual_minus_pred"] = x.actual_ypa - x.pred_ypa
    x["volume_yard_contribution"] = (x.actual_attempts - x.pred_attempts) * x.pred_ypa
    x["efficiency_yard_contribution"] = x.pred_attempts * (x.actual_ypa - x.pred_ypa)
    x["interaction_yard_contribution"] = (x.actual_attempts - x.pred_attempts) * (x.actual_ypa - x.pred_ypa)
    x["mc_point_product_remainder"] = x.pred_point_product - x.pred_pass_yards
    x["decomposition_sum"] = x.volume_yard_contribution + x.efficiency_yard_contribution + x.interaction_yard_contribution + x.mc_point_product_remainder
    x["decomposition_roundoff"] = x.pass_residual_actual_minus_pred - x.decomposition_sum
    if float(x.decomposition_roundoff.abs().max()) > 1e-6:
        raise RuntimeError(f"M69 exact decomposition failed: max remainder {x.decomposition_roundoff.abs().max()}")

    x["abs_pass_error"] = x.pass_residual_actual_minus_pred.abs()
    x["cat75"] = x.abs_pass_error.ge(75)
    x["cat100"] = x.abs_pass_error.ge(100)
    x["dbr_residual_actual_minus_pred"] = x.actual_dropback_rate - m.num(x.get("m64_pred_dropback_rate_neutral", np.nan))
    x["drive_residual_actual_minus_pred"] = x.actual_drives - m.num(x.get("m64_pred_drives", np.nan))
    x["plays_per_drive_residual"] = x.actual_plays_per_drive - m.num(x.get("m64_pred_plays_per_drive", np.nan))
    x["attempt_conversion_residual"] = x.actual_attempt_conversion - m.num(x.get("m64_pred_attempt_conversion", np.nan))
    x["trailing_share_surprise"] = x.actual_trailing8_share - m.num(x.get("m65_pred_trailing_share", np.nan))
    x["leading_share_surprise"] = x.actual_leading8_share - m.num(x.get("m65_pred_leading_share", np.nan))
    x["opening_baseline_playcaller"] = x.apply(lambda r: finite_value(r, "playcaller_opening_first15_dbr_mean8"), axis=1)
    x["opening_deviation_vs_playcaller"] = x.actual_first15_dbr - x.opening_baseline_playcaller
    x["mechanism"] = x.apply(classify, axis=1)

    recover_map = {
        "planned_pass_heavy_opening": "pregame_candidate",
        "planned_run_heavy_opening": "pregame_candidate",
        "role_or_participation": "pregame_candidate",
        "run_game_takeover": "partially_in_game",
        "run_game_failure_pass_pivot": "partially_in_game",
        "forced_trailing_volume": "partially_in_game",
        "leading_suppression": "partially_in_game",
        "dropback_to_attempt_conversion_loss": "partially_in_game",
        "possession_explosion": "mostly_in_game",
        "possession_collapse": "mostly_in_game",
        "ypa_explosion": "separate_efficiency_problem",
        "ypa_collapse": "separate_efficiency_problem",
        "mc_distribution_center_remainder": "model_aggregation",
        "other_volume_or_mixed": "unresolved",
    }
    x["recoverability"] = x.mechanism.map(recover_map).fillna("unresolved")

    sums = []
    for label, q in [("all", x), ("75plus", x[x.cat75]), ("100plus", x[x.cat100])]:
        for (season, mech), g in q.groupby(["season", "mechanism"], dropna=False):
            den = len(q[q.season.eq(season)])
            sums.append({"slice": label, "season": int(season), "mechanism": mech, "n": len(g), "share": len(g)/den if den else np.nan,
                         "mean_abs_pass_error": float(g.abs_pass_error.mean()), "mean_attempt_residual": float(g.attempt_residual_actual_minus_pred.mean()), "mean_ypa_residual": float(g.ypa_residual_actual_minus_pred.mean())})
        for mech, g in q.groupby("mechanism", dropna=False):
            sums.append({"slice": label, "season": "combined", "mechanism": mech, "n": len(g), "share": len(g)/len(q) if len(q) else np.nan,
                         "mean_abs_pass_error": float(g.abs_pass_error.mean()), "mean_attempt_residual": float(g.attempt_residual_actual_minus_pred.mean()), "mean_ypa_residual": float(g.ypa_residual_actual_minus_pred.mean())})
    summary = pd.DataFrame(sums)

    screen = []
    for feature in SCHEME_CANDIDATES:
        c24 = m.safe_corr(x.loc[x.season.eq(2024), feature], x.loc[x.season.eq(2024), "opening_deviation_vs_playcaller"])
        c25 = m.safe_corr(x.loc[x.season.eq(2025), feature], x.loc[x.season.eq(2025), "opening_deviation_vs_playcaller"])
        cc = m.safe_corr(x[feature], x.opening_deviation_vs_playcaller)
        strong = bool(np.isfinite(c24) and np.isfinite(c25) and np.sign(c24) == np.sign(c25) and abs(c24) >= .10 and abs(c25) >= .10 and abs(cc) >= .15)
        screen.append({"pregame_defense_feature": feature, "target": "opening_deviation_vs_playcaller", "corr_2024": c24, "corr_2025": c25, "corr_combined": cc, "strong_replicated_descriptive": strong})
    scheme_screen = pd.DataFrame(screen)

    stability = []
    pairs = [("opp_coverage_man_rate", "realized_def_man_rate"), ("opp_coverage_zone_rate", "realized_def_zone_rate"), ("opp_pressure_rate_generated", "realized_def_pressure_rate")]
    for pre, actual in pairs:
        if actual not in x:
            continue
        for sl, q in [("2024", x[x.season.eq(2024)]), ("2025", x[x.season.eq(2025)]), ("combined", x)]:
            z = pd.DataFrame({"p": m.num(q[pre]), "a": m.num(q[actual])}).dropna()
            stability.append({"season": sl, "pregame_feature": pre, "realized_feature": actual, "n": len(z), "corr": m.safe_corr(z.p, z.a),
                              "mae": float((z.p-z.a).abs().mean()) if len(z) else np.nan, "bias": float((z.p-z.a).mean()) if len(z) else np.nan})
    stability = pd.DataFrame(stability)

    x["opening_regime"] = pd.cut(x.opening_deviation_vs_playcaller, [-np.inf, -.12, .12, np.inf], labels=["run_heavier_than_caller", "near_caller_baseline", "pass_heavier_than_caller"])
    realized_cols = [c for c in x if c.startswith("realized_def_") or c.startswith("realized_cover")]
    cohort = []
    for (season, regime), g in x.groupby(["season", "opening_regime"], observed=True):
        rec = {"season": int(season), "opening_regime": str(regime), "n": len(g), "mean_opening_deviation": float(g.opening_deviation_vs_playcaller.mean())}
        for col in realized_cols:
            rec[col] = float(m.num(g[col]).mean()) if m.num(g[col]).notna().any() else np.nan
        cohort.append(rec)
    cohort = pd.DataFrame(cohort)

    recovery = []
    for sl, q in [("75plus", x[x.cat75]), ("100plus", x[x.cat100])]:
        total = float(q.abs_pass_error.sum())
        for recov, g in q.groupby("recoverability"):
            recovery.append({"slice": sl, "recoverability": recov, "n": len(g), "game_share": len(g)/len(q) if len(q) else np.nan,
                             "error_share": float(g.abs_pass_error.sum()/total) if total else np.nan, "mean_abs_error": float(g.abs_pass_error.mean())})
    recovery = pd.DataFrame(recovery)

    strong_scheme = int(scheme_screen.strong_replicated_descriptive.sum())
    planned100 = int(x[x.cat100 & x.mechanism.isin(["planned_pass_heavy_opening", "planned_run_heavy_opening"])].shape[0])
    interp = "m69_matchup_conditioning_supported_for_m70_hypothesis" if strong_scheme > 0 else "m69_opening_signal_not_explained_by_current_pregame_defense_profiles"
    interpretation = pd.DataFrame([{
        "target_games": len(x),
        "cat75_games": int(x.cat75.sum()),
        "cat100_games": int(x.cat100.sum()),
        "planned_opening_cat100_games": planned100,
        "strong_replicated_pregame_scheme_pairs": strong_scheme,
        "role_below_80_rows": int((m.num(x.get("m64_actual_qb_attempt_share", np.nan)) < .80).sum()),
        "mean_abs_mc_point_product_remainder": float(x.mc_point_product_remainder.abs().mean()),
        "max_abs_decomposition_roundoff": float(x.decomposition_roundoff.abs().max()),
        "m69_interpretation": interp,
        "production_actionable": False,
    }])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "m69_game_forensic_atlas.csv", index=False)
    summary.to_csv(a.out_dir / "m69_mechanism_summary.csv", index=False)
    scheme_screen.to_csv(a.out_dir / "m69_pregame_defense_opening_deviation_screen.csv", index=False)
    stability.to_csv(a.out_dir / "m69_defensive_scheme_stability.csv", index=False)
    cohort.to_csv(a.out_dir / "m69_realized_defense_by_opening_regime.csv", index=False)
    recovery.to_csv(a.out_dir / "m69_recoverability_summary.csv", index=False)
    interpretation.to_csv(a.out_dir / "m69_precommitted_interpretation.csv", index=False)
    manifest.to_csv(a.out_dir / "m69_source_manifest.csv", index=False)

    print("=== M69 CORRECTED INTERPRETATION ===")
    print(interpretation.to_string(index=False))
    print("=== M69 CORRECTED 100+ MECHANISMS ===")
    print(summary[(summary.slice.eq("100plus")) & summary.season.astype(str).eq("combined")].sort_values("n", ascending=False).to_string(index=False))
    print("=== M69 CORRECTED RECOVERABILITY ===")
    print(recovery.to_string(index=False))
    print("=== M69 CORRECTED PREGAME SCHEME -> OPENING DEVIATION ===")
    print(scheme_screen.to_string(index=False))
    print("=== M69 CORRECTED SCHEME STABILITY ===")
    print(stability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
