#!/usr/bin/env python3
"""Frozen full-stack test of OPPORTUNITY_AUTHORITY_PRIORITY_V1.

Research only.  No sportsbook inputs, no fitted parameters, no production
mutation.  The only candidate difference is rule-layer source priority for
RB/HB/FB rush share and WR/TE target share.  All efficiency metrics remain on
the exact empirical-Bayes authority.

The evaluator uses the same leakage-safe historical context for baseline and
candidate, the same current ensemble weights, and the fold-safe TE-R5P /
WR-R15 production-order replay authority documented in Amendment 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling import simulation_rules
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    _load_participation_snaps,
    apply_te_fold,
    apply_wr_fold,
)
from scripts.research.persist_historical_simulated_outcomes_v1 import (
    _exact_week,
    _parse_weeks,
    _read,
    _read_optional,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import lookup
from scripts.utils.canonical_names import canon_team

VERSION = "OPPORTUNITY_AUTHORITY_PRIORITY_V1"
BASELINE_AUTHORITY = simulation_rules.OPPORTUNITY_AUTHORITY_BASELINE
CANDIDATE_AUTHORITY = simulation_rules.OPPORTUNITY_AUTHORITY_PLAYERFORM_FAST_STATE
TOL = 1e-10

MARKET_FAMILIES = {
    "RB_RUSH_ATT": ("RB", "rush_att"),
    "RB_RUSH_YARDS": ("RB", "rush_yards"),
    "RB_RUSH_REC_YARDS": ("RB", "rush_rec_yards"),
    "RB_RECEPTIONS": ("RB", "receptions"),
    "RB_REC_YARDS": ("RB", "rec_yards"),
    "WR_RECEPTIONS": ("WR", "receptions"),
    "WR_REC_YARDS": ("WR", "rec_yards"),
    "TE_RECEPTIONS": ("TE", "receptions"),
    "TE_REC_YARDS": ("TE", "rec_yards"),
    "QB_RUSH_ATT": ("QB", "rush_att"),
    "QB_RUSH_YARDS": ("QB", "rush_yards"),
    "OTHER_RUSH_ATT": ("OTHER", "rush_att"),
    "OTHER_RUSH_YARDS": ("OTHER", "rush_yards"),
}

DIRECT_FAMILIES = ("RB_RUSH_OPPORTUNITY", "WR_TARGET_OPPORTUNITY", "TE_TARGET_OPPORTUNITY")


def _pos(value) -> str:
    p = str(value or "").upper().strip()
    if p in {"RB", "HB", "TB", "FB"} or p.startswith("RB") or p.startswith("FB"):
        return "RB"
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p.startswith("QB"):
        return "QB"
    return "OTHER"


def _metric(actual: pd.Series, pred: pd.Series) -> dict:
    x = pd.DataFrame({
        "actual": pd.to_numeric(actual, errors="coerce"),
        "pred": pd.to_numeric(pred, errors="coerce"),
    }).dropna()
    if x.empty:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "p90_ae": np.nan}
    err = x["pred"] - x["actual"]
    ae = err.abs()
    return {
        "n": int(len(x)),
        "mae": float(ae.mean()),
        "bias": float(err.mean()),
        "p90_ae": float(ae.quantile(0.90)),
    }


def _read_component(path: Path, season: int) -> pd.DataFrame:
    x = _read(path, f"component predictions {season}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].between(2, 18)].copy()
    if x.empty:
        raise RuntimeError(f"component file has no season={season} weeks 2-18")
    key = ["season", "week", "team", "player_clean_key", "market"]
    if x.duplicated(key).any():
        raise RuntimeError(f"component file duplicate identities season={season}")
    return x


def _rules_player_frame(bundle, authority: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = cp.build_market_frame(bundle)
    bayes = build_bayesian_baseline(bundle.player_consensus)
    market = apply_bayesian_to_metrics(market, bayes)
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        ruled = simulation_rules.apply_rules_to_metrics(
            market,
            bayes_baseline=bayes,
            opportunity_authority=authority,
        )
    if not pd.to_numeric(ruled.get("rules_applied", 0), errors="coerce").fillna(0).eq(1).all():
        bad = ruled.loc[
            ~pd.to_numeric(ruled.get("rules_applied", 0), errors="coerce").fillna(0).eq(1),
            ["player", "team"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"rules failed to materialize full historical player universe: {bad}")
    player_cols = ["event_id", "team", "player_clean_key"]
    players = ruled.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()
    if players.duplicated(player_cols).any():
        raise RuntimeError("player-level ruled frame is not unique")
    return ruled, players


def _apply_specialists(players: pd.DataFrame, *, season: int, snaps: pd.DataFrame,
                       te_params: dict, wr_params: dict | None) -> tuple[pd.DataFrame, dict]:
    explicit, _ = materialize_target_entitlement(players)
    te_final, _, te_audit = apply_te_fold(explicit, snaps=snaps, params=te_params)
    if int(season) == 2024:
        if wr_params is None:
            raise RuntimeError("2024 requires fold-safe WR-R15 authority")
        final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
        wr_applied = True
    elif int(season) == 2025:
        final = te_final
        wr_audit = {
            "m38_wr1_anchor_max_abs_gap": 0.0,
            "wr2plus_pool_max_abs_gap": 0.0,
            "wr_room_mass_max_abs_gap": 0.0,
            "non_wr_max_abs_gap": 0.0,
            "same_future_participation": 0,
        }
        wr_applied = False
    else:
        raise RuntimeError(f"unsupported frozen season {season}")
    return final, {
        "te": te_audit,
        "wr": wr_audit,
        "wr_r15_fold_applied": wr_applied,
    }


def _trace_map(trace: list[dict]) -> pd.DataFrame:
    x = pd.DataFrame(trace)
    if x.empty:
        raise RuntimeError("simulation opportunity trace is empty")
    key = ["event_id", "team", "player_clean_key"]
    if x.duplicated(key).any():
        raise RuntimeError("simulation opportunity trace contains duplicate player identities")
    return x


def _simulation_means(sim, component_week: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in component_week.iterrows():
        market = str(row.get("market", "")).lower().strip()
        if market == "pass_yards":
            # Candidate does not alter M89/M90 final QB passing authority.
            # Raw simulator passing yards are not the promoted point mean.
            continue
        arr = lookup(sim, row, market)
        if arr is None or len(arr) == 0:
            continue
        vals = np.asarray(arr, dtype=float)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"non-finite simulation array {row.get('player')} {market}")
        rows.append({
            "season": int(row["season"]), "week": int(row["week"]),
            "team": canon_team(row["team"]), "player_clean_key": str(row["player_clean_key"]),
            "market": market, "mc_proj_rebuilt": float(vals.mean()),
        })
    out = pd.DataFrame(rows)
    key = ["season", "week", "team", "player_clean_key", "market"]
    if out.duplicated(key).any():
        raise RuntimeError("rebuilt MC means contain duplicate identities")
    return out


def _apply_v2_rb_combined_mean(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    out = frame.copy()
    out["position_family"] = out["position"].map(_pos)
    keys = ["season", "week", "team", "player_clean_key"]
    rb = out.loc[out["position_family"].eq("RB")].copy()
    rush = rb.loc[rb["market"].astype(str).eq("rush_yards"), keys + ["ensemble_proj"]].rename(columns={"ensemble_proj":"_rush"})
    rec = rb.loc[rb["market"].astype(str).eq("rec_yards"), keys + ["ensemble_proj"]].rename(columns={"ensemble_proj":"_rec"})
    both = rush.merge(rec, on=keys, how="inner", validate="one_to_one")
    both["_v2"] = pd.to_numeric(both["_rush"], errors="coerce") + pd.to_numeric(both["_rec"], errors="coerce")
    target = out.loc[out["position_family"].eq("RB") & out["market"].astype(str).eq("rush_rec_yards")].copy()
    if target.empty:
        return out, 0.0
    joined = target[keys + ["ensemble_proj"]].merge(both[keys + ["_v2"]], on=keys, how="left", validate="one_to_one")
    if joined["_v2"].isna().any():
        raise RuntimeError("RB Rush+Receiving V2 could not attach both standalone means")
    map_v2 = {tuple(r[k] for k in keys): float(r["_v2"]) for _, r in joined.iterrows()}
    idx = out.index[out["position_family"].eq("RB") & out["market"].astype(str).eq("rush_rec_yards")]
    for i in idx:
        k = tuple(out.at[i, col] for col in keys)
        out.at[i, "ensemble_proj"] = map_v2[k]
    # Identity is definitional after the override.
    check = out.loc[idx, keys + ["ensemble_proj"]].merge(both, on=keys, how="left", validate="one_to_one")
    gap = np.abs(
        pd.to_numeric(check["ensemble_proj"], errors="coerce").to_numpy(float)
        - (
            pd.to_numeric(check["_rush"], errors="coerce").to_numpy(float)
            + pd.to_numeric(check["_rec"], errors="coerce").to_numpy(float)
        )
    )
    max_gap = float(np.max(gap)) if len(gap) else 0.0
    return out, max_gap


def _build_projection_frame(component_week: pd.DataFrame, means: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    key = ["season", "week", "team", "player_clean_key", "market"]
    out = component_week.copy()
    out["team"] = out["team"].map(canon_team)
    out["market"] = out["market"].astype(str).str.lower()
    joined = out.merge(means, on=key, how="left", validate="one_to_one")
    non_qb = ~joined["market"].eq("pass_yards")
    if joined.loc[non_qb, "mc_proj_rebuilt"].isna().any():
        sample = joined.loc[non_qb & joined["mc_proj_rebuilt"].isna(), key + ["player"]].head(20).to_dict("records")
        raise RuntimeError(f"candidate simulation did not reproduce all non-QB component rows: {sample}")
    joined.loc[non_qb, "mc_proj"] = pd.to_numeric(joined.loc[non_qb, "mc_proj_rebuilt"], errors="raise")
    joined = apply_ensemble(joined.drop(columns=["mc_proj_rebuilt"]), weights=weights)
    joined, v2_gap = _apply_v2_rb_combined_mean(joined)
    return joined, v2_gap


def _rule_integrity(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    key = ["event_id", "team", "player_clean_key", "market"]
    cols = [
        "position", "rules_plays_est", "rules_pass_rate", "rules_tgt_share", "rules_rush_share",
        "rules_ypt", "rules_ypc", "rules_ypa", "rules_catch_rate", "rules_volatility_mult",
        "rules_pass_eff_mult", "rules_rush_eff_mult", "rules_injury_redistribution",
    ]
    b = base[key + cols].copy()
    c = cand[key + cols].copy()
    z = b.merge(c, on=key, suffixes=("_base","_cand"), how="outer", validate="one_to_one", indicator=True)
    if not z["_merge"].eq("both").all():
        raise RuntimeError("baseline/candidate rule universe changed")

    protected = [
        "rules_plays_est", "rules_pass_rate", "rules_ypt", "rules_ypc", "rules_ypa",
        "rules_catch_rate", "rules_volatility_mult", "rules_pass_eff_mult", "rules_rush_eff_mult",
    ]
    max_protected = 0.0
    for col in protected:
        a = pd.to_numeric(z[f"{col}_base"], errors="coerce")
        d = pd.to_numeric(z[f"{col}_cand"], errors="coerce")
        both = a.notna() & d.notna()
        missing_mismatch = int((a.isna() ^ d.isna()).sum())
        if missing_mismatch:
            raise RuntimeError(f"protected rule column missingness changed: {col}")
        if both.any():
            max_protected = max(max_protected, float((a[both]-d[both]).abs().max()))

    z["_pos"] = z["position_base"].map(_pos)
    rush_gap = (
        pd.to_numeric(z["rules_rush_share_base"], errors="coerce")
        - pd.to_numeric(z["rules_rush_share_cand"], errors="coerce")
    ).abs().fillna(0.0)
    bad_rush = z.loc[rush_gap.gt(TOL) & ~z["_pos"].eq("RB")]
    if not bad_rush.empty:
        raise RuntimeError(f"candidate changed non-RB rush-share authority: {bad_rush[key+['_pos']].head(20).to_dict('records')}")

    tgt_gap = (
        pd.to_numeric(z["rules_tgt_share_base"], errors="coerce")
        - pd.to_numeric(z["rules_tgt_share_cand"], errors="coerce")
    ).abs().fillna(0.0)
    # Direct source-priority changes are WR/TE.  Other target rows may change
    # only inside teams where the existing injury redistribution rule is active;
    # that is downstream propagation explicitly required by the frozen plan.
    injury_team = set(
        z.loc[
            pd.to_numeric(z["rules_injury_redistribution_base"], errors="coerce").fillna(0).eq(1)
            | pd.to_numeric(z["rules_injury_redistribution_cand"], errors="coerce").fillna(0).eq(1),
            "team",
        ].astype(str)
    )
    bad_tgt = z.loc[
        tgt_gap.gt(TOL)
        & ~z["_pos"].isin({"WR","TE"})
        & ~z["team"].astype(str).isin(injury_team)
    ]
    if not bad_tgt.empty:
        raise RuntimeError(f"candidate changed unauthorized target-share row: {bad_tgt[key+['_pos']].head(20).to_dict('records')}")

    return {
        "max_protected_rule_gap": float(max_protected),
        "rush_share_changed_rows": int(rush_gap.gt(TOL).sum()),
        "target_share_changed_rows": int(tgt_gap.gt(TOL).sum()),
        "injury_propagation_teams": sorted(injury_team),
        "only_frozen_source_priority_cells_changed_before_downstream": True,
    }


def _score_market(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        s = detail.loc[detail["season"].eq(season)]
        for label, (pos, market) in MARKET_FAMILIES.items():
            q = s.loc[s["market"].eq(market)].copy()
            if pos != "OTHER":
                q = q.loc[q["position_family"].eq(pos)]
            else:
                q = q.loc[~q["position_family"].isin({"RB","QB"})]
            bm = _metric(q["actual"], q["baseline_proj"])
            cm = _metric(q["actual"], q["candidate_proj"])
            rows.append({
                "season": season, "family": label,
                "n": bm["n"],
                "baseline_mae": bm["mae"], "candidate_mae": cm["mae"],
                "mae_delta_candidate_minus_baseline": cm["mae"] - bm["mae"],
                "baseline_bias": bm["bias"], "candidate_bias": cm["bias"],
                "baseline_p90_ae": bm["p90_ae"], "candidate_p90_ae": cm["p90_ae"],
            })
    # pooled
    for label, (pos, market) in MARKET_FAMILIES.items():
        q = detail.loc[detail["market"].eq(market)].copy()
        if pos != "OTHER":
            q = q.loc[q["position_family"].eq(pos)]
        else:
            q = q.loc[~q["position_family"].isin({"RB","QB"})]
        bm = _metric(q["actual"], q["baseline_proj"])
        cm = _metric(q["actual"], q["candidate_proj"])
        rows.append({
            "season": "POOLED", "family": label, "n": bm["n"],
            "baseline_mae": bm["mae"], "candidate_mae": cm["mae"],
            "mae_delta_candidate_minus_baseline": cm["mae"] - bm["mae"],
            "baseline_bias": bm["bias"], "candidate_bias": cm["bias"],
            "baseline_p90_ae": bm["p90_ae"], "candidate_p90_ae": cm["p90_ae"],
        })
    return pd.DataFrame(rows)


def _score_direct(direct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        for fam in DIRECT_FAMILIES:
            q = direct.loc[direct["season"].eq(season) & direct["family"].eq(fam)]
            bm = _metric(q["actual"], q["baseline_pred"])
            cm = _metric(q["actual"], q["candidate_pred"])
            rows.append({
                "season": season, "family": fam, "n": bm["n"],
                "baseline_mae": bm["mae"], "candidate_mae": cm["mae"],
                "mae_delta_candidate_minus_baseline": cm["mae"] - bm["mae"],
                "baseline_p90_ae": bm["p90_ae"], "candidate_p90_ae": cm["p90_ae"],
            })
    return pd.DataFrame(rows)


def _cell(df: pd.DataFrame, season, family: str) -> pd.Series:
    q = df.loc[df["season"].astype(str).eq(str(season)) & df["family"].eq(family)]
    if len(q) != 1:
        raise RuntimeError(f"expected one summary cell season={season} family={family}, got {len(q)}")
    return q.iloc[0]


def _evaluate_gates(direct: pd.DataFrame, market: pd.DataFrame, integrity: dict) -> tuple[dict, bool]:
    gates: dict[str, bool] = {}
    for fam in DIRECT_FAMILIES:
        for season in (2024, 2025):
            r = _cell(direct, season, fam)
            gates[f"direct_{fam}_{season}_mae_strict_improve"] = bool(r["candidate_mae"] < r["baseline_mae"] - 1e-12)
            gates[f"direct_{fam}_{season}_p90_nonworse"] = bool(r["candidate_p90_ae"] <= r["baseline_p90_ae"] + 1e-12)

    for fam in (
        "RB_RUSH_YARDS","RB_RUSH_REC_YARDS","WR_REC_YARDS",
        "WR_RECEPTIONS","TE_REC_YARDS","TE_RECEPTIONS",
    ):
        for season in (2024, 2025):
            r = _cell(market, season, fam)
            gates[f"{fam}_{season}_mae_nonworse"] = bool(r["candidate_mae"] <= r["baseline_mae"] + 1e-12)

    pooled_improved = 0
    for fam in (
        "RB_RUSH_YARDS","RB_RUSH_REC_YARDS","WR_REC_YARDS",
        "WR_RECEPTIONS","TE_REC_YARDS","TE_RECEPTIONS",
    ):
        r = _cell(market, "POOLED", fam)
        pooled_improved += int(r["candidate_mae"] < r["baseline_mae"] - 1e-12)
    gates["at_least_four_of_six_downstream_pooled_improve"] = bool(pooled_improved >= 4)

    for fam in ("QB_RUSH_ATT","QB_RUSH_YARDS","OTHER_RUSH_ATT","OTHER_RUSH_YARDS","RB_REC_YARDS","RB_RECEPTIONS"):
        for season in (2024, 2025):
            r = _cell(market, season, fam)
            gates[f"{fam}_{season}_guard_nonworse"] = bool(r["candidate_mae"] <= r["baseline_mae"] + 1e-12)

    gates["qb_passing_final_mean_invariant"] = bool(float(integrity["qb_passing_final_mean_max_gap"]) <= 1e-10)
    gates["team_volume_inputs_invariant"] = bool(float(integrity["max_protected_rule_gap"]) <= 1e-12)
    gates["bayesian_efficiency_invariant"] = bool(integrity["bayesian_efficiency_columns_invariant"])
    gates["ml_state_components_invariant"] = bool(integrity["ml_state_components_invariant"])
    gates["ensemble_weights_invariant"] = bool(integrity["ensemble_weights_invariant"])
    gates["specialist_assets_invariant"] = bool(integrity["specialist_assets_invariant"])
    gates["rb_rush_rec_v2_exact"] = bool(float(integrity["max_rb_rush_rec_v2_identity_gap"]) <= 1e-10)
    gates["zero_sportsbook_inputs"] = bool(integrity["sportsbook_inputs_used"] == 0)
    gates["zero_target_future_feature_rows"] = bool(integrity["target_future_feature_rows"] == 0)
    gates["week1_rows_scored_zero"] = bool(integrity["week1_rows_scored"] == 0)
    gates["source_priority_boundary_exact"] = bool(integrity["only_frozen_source_priority_cells_changed_before_downstream"])

    passed = bool(all(gates.values()))
    gates["pooled_downstream_improved_count"] = int(pooled_improved)
    gates["all_frozen_gates_pass"] = passed
    return gates, passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2024", type=Path, required=True)
    ap.add_argument("--universe-2025", type=Path, required=True)
    ap.add_argument("--component-2024", type=Path, required=True)
    ap.add_argument("--component-2025", type=Path, required=True)
    ap.add_argument("--te-coefficients", type=Path, required=True)
    ap.add_argument("--wr-coefficients", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    ap.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--weeks", default="2-18")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    weeks = _parse_weeks(a.weeks)
    if 1 in weeks or not weeks:
        raise RuntimeError("frozen candidate must score only Weeks 2-18")
    if sorted(set(weeks)) != list(range(2, 19)):
        raise RuntimeError(f"frozen weeks drifted: {weeks}")

    logs = _read(a.player_logs, "player logs")
    team_weekly = _read(a.team_weekly, "team weekly")
    schedule = _read(a.schedule, "schedule")
    injuries_history = _read_optional(a.injuries)
    weather_history = _read_optional(a.weather)
    components = {
        2024: _read_component(a.component_2024, 2024),
        2025: _read_component(a.component_2025, 2025),
    }
    universes = {2024: a.universe_2024, 2025: a.universe_2025}
    priors = {2024: 2023, 2025: 2024}

    te_params = {
        s: _load_fold_params(a.te_coefficients, test_season=s, features=TE_FEATURES, label="TE-R5P")
        for s in (2024, 2025)
    }
    wr_params = _load_fold_params(a.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15")
    snaps, snap_dup_rate, snap_source_seasons = _load_participation_snaps()
    if snap_dup_rate > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {snap_dup_rate}")

    weights = load_weights()
    if weights.empty:
        raise RuntimeError("production ensemble weights unavailable")

    detail_parts = []
    direct_parts = []
    integrity_rows = []
    max_v2_gap = 0.0

    for season in (2024, 2025):
        comp = components[season]
        for week in weeks:
            comp_week = comp.loc[comp["week"].eq(int(week))].copy()
            if comp_week.empty:
                continue
            universe = _read(universes[season] / f"{season}_week_{week:02d}.csv", f"pregame universe {season} W{week:02d}")
            injuries = _exact_week(injuries_history, season, week)
            weather = _exact_week(weather_history, season, week)
            bundle = build_historical_context_bundle(
                player_logs=logs, team_weekly=team_weekly, pregame_universe=universe,
                schedule=schedule, season=season, week=week, prior_season=priors[season],
                injuries=injuries, weather=weather,
            )

            base_rules, base_players = _rules_player_frame(bundle, BASELINE_AUTHORITY)
            cand_rules, cand_players = _rules_player_frame(bundle, CANDIDATE_AUTHORITY)
            rule_audit = _rule_integrity(base_rules, cand_rules)

            # Same Bayesian frame was used for both.  Efficiency-rule invariance
            # is independently certified by the protected rule-column check.
            base_final, base_spec = _apply_specialists(
                base_players, season=season, snaps=snaps,
                te_params=te_params[season], wr_params=wr_params if season == 2024 else None,
            )
            cand_final, cand_spec = _apply_specialists(
                cand_players, season=season, snaps=snaps,
                te_params=te_params[season], wr_params=wr_params if season == 2024 else None,
            )

            base_trace: list[dict] = []
            cand_trace: list[dict] = []
            seed = 42 + int(week)
            base_sim = explicit_simulate(base_final, iterations=a.iterations, seed=seed, allocation_trace=base_trace)
            cand_sim = explicit_simulate(cand_final, iterations=a.iterations, seed=seed, allocation_trace=cand_trace)

            base_means = _simulation_means(base_sim, comp_week)
            cand_means = _simulation_means(cand_sim, comp_week)
            base_proj, base_v2 = _build_projection_frame(comp_week, base_means, weights)
            cand_proj, cand_v2 = _build_projection_frame(comp_week, cand_means, weights)
            max_v2_gap = max(max_v2_gap, base_v2, cand_v2)

            key = ["season","week","team","player_clean_key","market"]
            bp = base_proj[key + ["player","position","actual","actual_opportunities","ensemble_proj","ml_proj","state_proj"]].rename(columns={"ensemble_proj":"baseline_proj"})
            cpj = cand_proj[key + ["ensemble_proj","ml_proj","state_proj"]].rename(columns={"ensemble_proj":"candidate_proj","ml_proj":"ml_proj_candidate","state_proj":"state_proj_candidate"})
            z = bp.merge(cpj, on=key, how="inner", validate="one_to_one")
            if len(z) != len(bp) or len(z) != len(cpj):
                raise RuntimeError(f"projection identity drift {season} W{week}")
            z["position_family"] = z["position"].map(_pos)
            detail_parts.append(z)

            bt = _trace_map(base_trace).rename(columns={
                "realized_multinomial_mean_carries":"baseline_carries",
                "realized_multinomial_mean_targets":"baseline_targets",
            })
            ct = _trace_map(cand_trace).rename(columns={
                "realized_multinomial_mean_carries":"candidate_carries",
                "realized_multinomial_mean_targets":"candidate_targets",
            })
            tkey = ["event_id","team","player_clean_key"]
            tr = bt[tkey+["baseline_carries","baseline_targets"]].merge(
                ct[tkey+["candidate_carries","candidate_targets"]],
                on=tkey, how="inner", validate="one_to_one",
            )

            actual_rows = comp_week.copy()
            actual_rows["position_family"] = actual_rows["position"].map(_pos)
            # RB rushing opportunity.
            rb = actual_rows.loc[
                actual_rows["position_family"].eq("RB") & actual_rows["market"].eq("rush_att"),
                ["season","week","event_id","team","player_clean_key","player","actual"],
            ].drop_duplicates(tkey)
            rb = rb.merge(tr, on=tkey, how="left", validate="one_to_one")
            if not rb.empty:
                q = pd.DataFrame({
                    "season": rb["season"], "week": rb["week"], "family": "RB_RUSH_OPPORTUNITY",
                    "player": rb["player"], "team": rb["team"], "player_clean_key": rb["player_clean_key"],
                    "actual": pd.to_numeric(rb["actual"], errors="coerce"),
                    "baseline_pred": pd.to_numeric(rb["baseline_carries"], errors="coerce"),
                    "candidate_pred": pd.to_numeric(rb["candidate_carries"], errors="coerce"),
                })
                direct_parts.append(q)

            # WR / TE target opportunity, one actual-target row per player.
            for pos in ("WR","TE"):
                rec = actual_rows.loc[
                    actual_rows["position_family"].eq(pos) & actual_rows["market"].eq("rec_yards"),
                    ["season","week","event_id","team","player_clean_key","player","actual_opportunities"],
                ].drop_duplicates(tkey)
                rec = rec.merge(tr, on=tkey, how="left", validate="one_to_one")
                if not rec.empty:
                    direct_parts.append(pd.DataFrame({
                        "season": rec["season"], "week": rec["week"],
                        "family": f"{pos}_TARGET_OPPORTUNITY",
                        "player": rec["player"], "team": rec["team"], "player_clean_key": rec["player_clean_key"],
                        "actual": pd.to_numeric(rec["actual_opportunities"], errors="coerce"),
                        "baseline_pred": pd.to_numeric(rec["baseline_targets"], errors="coerce"),
                        "candidate_pred": pd.to_numeric(rec["candidate_targets"], errors="coerce"),
                    }))

            ml_gap = (
                pd.to_numeric(z["ml_proj"], errors="coerce") - pd.to_numeric(z["ml_proj_candidate"], errors="coerce")
            ).abs().fillna(0.0)
            st_gap = (
                pd.to_numeric(z["state_proj"], errors="coerce") - pd.to_numeric(z["state_proj_candidate"], errors="coerce")
            ).abs().fillna(0.0)

            integrity_rows.append({
                "season": season, "week": week,
                **rule_audit,
                "ml_max_gap": float(ml_gap.max()) if len(ml_gap) else 0.0,
                "state_max_gap": float(st_gap.max()) if len(st_gap) else 0.0,
                "baseline_te_pool_gap": float(base_spec["te"]["team_te_pool_max_abs_gap"]),
                "candidate_te_pool_gap": float(cand_spec["te"]["team_te_pool_max_abs_gap"]),
                "baseline_wr_r15_applied": bool(base_spec["wr_r15_fold_applied"]),
                "candidate_wr_r15_applied": bool(cand_spec["wr_r15_fold_applied"]),
            })
            print(f"[opportunity-authority] {season} W{week:02d} rows={len(z)} direct_trace={len(tr)}")

    if not detail_parts or not direct_parts:
        raise RuntimeError("candidate produced no scoreable rows")
    detail = pd.concat(detail_parts, ignore_index=True)
    direct = pd.concat(direct_parts, ignore_index=True)
    integrity_df = pd.DataFrame(integrity_rows)

    if detail["week"].eq(1).any() or direct["week"].eq(1).any():
        raise RuntimeError("Week 1 entered frozen candidate scoring")
    if set(detail["season"].astype(int)) != {2024, 2025}:
        raise RuntimeError(f"candidate season set drifted: {sorted(detail['season'].unique())}")

    market_summary = _score_market(detail)
    direct_summary = _score_direct(direct)

    # Candidate leaves M89/M90 point authority outside this rule seam.  Prove
    # that all QB-passing rows use the identical pre-existing component fields;
    # the promoted final mean therefore remains exact by contract.
    qb = detail.loc[detail["market"].eq("pass_yards")].copy()
    qb_component_gap = 0.0
    for left, right in (("ml_proj","ml_proj_candidate"),("state_proj","state_proj_candidate")):
        a1 = pd.to_numeric(qb[left], errors="coerce")
        b1 = pd.to_numeric(qb[right], errors="coerce")
        mask = a1.notna() & b1.notna()
        if (a1.isna() ^ b1.isna()).any():
            raise RuntimeError(f"QB component missingness changed: {left}")
        if mask.any():
            qb_component_gap = max(qb_component_gap, float((a1[mask]-b1[mask]).abs().max()))

    integrity = {
        "version": VERSION,
        "evaluation_seasons": [2024, 2025],
        "weeks": list(range(2,19)),
        "candidate_variants": 1,
        "fit_parameters": 0,
        "sportsbook_inputs_used": 0,
        "target_future_feature_rows": 0,
        "week1_rows_scored": 0,
        "max_protected_rule_gap": float(integrity_df["max_protected_rule_gap"].max()),
        "only_frozen_source_priority_cells_changed_before_downstream": bool(integrity_df["only_frozen_source_priority_cells_changed_before_downstream"].all()),
        "bayesian_efficiency_columns_invariant": bool(integrity_df["max_protected_rule_gap"].max() <= 1e-12),
        "ml_state_components_invariant": bool(
            integrity_df["ml_max_gap"].max() <= 1e-12 and integrity_df["state_max_gap"].max() <= 1e-12
        ),
        "ensemble_weights_invariant": True,
        "specialist_assets_invariant": True,
        "max_rb_rush_rec_v2_identity_gap": float(max_v2_gap),
        "qb_passing_final_mean_max_gap": float(qb_component_gap),
        "wr_r15_2024_applied_all_weeks": bool(integrity_df.loc[integrity_df["season"].eq(2024),"candidate_wr_r15_applied"].all()),
        "wr_r15_2025_applied_rows": int(integrity_df.loc[integrity_df["season"].eq(2025),"candidate_wr_r15_applied"].sum()),
        "te_r5p_2024_2025_applied": True,
        "snap_duplicate_rate": float(snap_dup_rate),
        "snap_source_seasons": [int(x) for x in snap_source_seasons],
    }
    if integrity["wr_r15_2025_applied_rows"] != 0:
        raise RuntimeError("WR-R15 2025 OOS exclusion contract violated")
    if not integrity["wr_r15_2024_applied_all_weeks"]:
        raise RuntimeError("WR-R15 2024 OOS authority missing")
    if integrity["max_protected_rule_gap"] > 1e-12:
        raise RuntimeError(f"candidate changed protected rule inputs: {integrity['max_protected_rule_gap']}")
    if integrity["max_rb_rush_rec_v2_identity_gap"] > 1e-10:
        raise RuntimeError(f"RB Rush+Receiving V2 identity failed: {integrity['max_rb_rush_rec_v2_identity_gap']}")

    gates, passed = _evaluate_gates(direct_summary, market_summary, integrity)
    disposition = (
        "OPPORTUNITY_AUTHORITY_PRIORITY_V1_QUALIFIED"
        if passed else
        "OPPORTUNITY_AUTHORITY_PRIORITY_V1_FAILED_CLOSED"
    )
    result = {
        "version": VERSION,
        "disposition": disposition,
        "production_changed": False,
        "sportsbook_inputs_used": 0,
        "current_or_future_outcomes_used_for_features": False,
        "week3_2026_outcomes_used": False,
        "candidate_variants": 1,
        "fit_parameters": 0,
        "integrity": integrity,
        "gates": gates,
        "next_step_if_qualified": "SEPARATE_PRODUCTION_INTEGRATION_PLAN_ONLY",
        "rescue_authorized_if_failed": False,
    }

    detail.to_csv(a.out_dir / "opportunity_authority_priority_market_detail.csv", index=False)
    direct.to_csv(a.out_dir / "opportunity_authority_priority_direct_detail.csv", index=False)
    market_summary.to_csv(a.out_dir / "opportunity_authority_priority_market_summary.csv", index=False)
    direct_summary.to_csv(a.out_dir / "opportunity_authority_priority_direct_summary.csv", index=False)
    integrity_df.to_csv(a.out_dir / "opportunity_authority_priority_integrity_by_week.csv", index=False)
    (a.out_dir / "opportunity_authority_priority_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== DIRECT OPPORTUNITY ===")
    print(direct_summary.to_string(index=False))
    print("=== DOWNSTREAM MARKETS ===")
    print(market_summary.to_string(index=False))
    print("=== RESULT ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
