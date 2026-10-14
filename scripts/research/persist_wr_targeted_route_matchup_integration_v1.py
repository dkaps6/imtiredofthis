#!/usr/bin/env python3
"""Generate 2024 promoted-WR baseline and targeted-route efficiency integration arms.

Research only. Frozen by WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1_FREEZE.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.research.persist_historical_simulated_outcomes_v1 import (
    KEYS, _canon_keys, _exact_week, _historical_outcomes, _parse_weeks, _read, _read_optional,
)
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES, WR_FEATURES, _compare_results_exact, _load_fold_params,
    apply_te_fold, apply_wr_fold, _pos_family,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

SEASON = 2024
PRIOR_SEASON = 2023
RECEIVING_MARKETS = {"rec_yards", "receptions", "rush_rec_yards"}
WR_POS = {"WR", "LWR", "RWR", "SWR"}


def _first_finite(row: pd.Series, names: list[str], default: float) -> float:
    for name in names:
        if name in row.index:
            try:
                v = float(row.get(name))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return float(default)


def _ols_slope_clipped(x, y) -> tuple[float, float, int]:
    z = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(z) < 100:
        raise RuntimeError(f"development slope has insufficient rows: {len(z)}")
    a = np.column_stack([np.ones(len(z)), z["x"].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(a, z["y"].to_numpy(float), rcond=None)
    raw = float(coef[1])
    clipped = float(np.clip(raw, 0.0, 1.0))
    return raw, clipped, int(len(z))


def _fit_betas(casebook: pd.DataFrame) -> dict:
    q = casebook.loc[
        casebook["season"].eq(2023)
        & casebook["position"].astype(str).str.upper().eq("WR")
    ].copy()
    y_raw, y_beta, y_n = _ols_slope_clipped(q["matchup_delta_ypt"], q["ypt_residual"])
    c_raw, c_beta, c_n = _ols_slope_clipped(q["matchup_delta_catch"], q["catch_residual"])
    return {
        "fit_season": 2023,
        "beta_ypt_raw": y_raw,
        "beta_ypt": y_beta,
        "beta_ypt_n": y_n,
        "beta_catch_raw": c_raw,
        "beta_catch": c_beta,
        "beta_catch_n": c_n,
    }


def _bridge_casebook(casebook: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    cb = casebook.loc[
        casebook["season"].eq(SEASON)
        & casebook["position"].astype(str).str.upper().eq("WR")
    ].copy()
    cb["team"] = cb["team"].map(canon_team)
    cb["player_id"] = cb["player_id"].astype(str).str.strip()

    logs = player_logs.copy()
    logs["team"] = logs["team"].map(canon_team)
    if "player_id" not in logs.columns or "player_clean_key" not in logs.columns:
        raise RuntimeError("historical player logs missing player_id/player_clean_key bridge")
    bridge = logs.loc[
        logs["season"].eq(SEASON),
        ["season", "week", "team", "player_id", "player_clean_key"],
    ].copy()
    bridge["player_id"] = bridge["player_id"].astype(str).str.strip()
    bridge = bridge.drop_duplicates(["season", "week", "team", "player_id"], keep="last")
    if bridge.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("player identity bridge is not unique")

    out = cb.merge(
        bridge,
        on=["season", "week", "team", "player_id"],
        how="left",
        validate="many_to_one",
    )
    out["identity_bridge_method"] = np.where(
        out["player_clean_key"].notna() & out["player_clean_key"].astype(str).ne(""),
        "EXACT_WEEK_TEAM_GSIS",
        "",
    )

    # Mechanical identity fallback only: GSIS is the stable player authority.
    # Some weekly-stat rows can be absent or carry a different team-side
    # representation even when the targeted-route source has an official target.
    # A missing exact bridge may therefore use the historical logs' unique
    # GSIS -> canonical player key mapping, but only when that mapping is
    # globally one-to-one across the loaded 2023-2024 history.
    global_map = (
        logs.loc[
            logs["player_id"].astype(str).str.strip().ne(""),
            ["player_id", "player_clean_key"],
        ]
        .drop_duplicates()
        .copy()
    )
    global_map["player_id"] = global_map["player_id"].astype(str).str.strip()
    counts = global_map.groupby("player_id")["player_clean_key"].nunique()
    ambiguous = set(counts.loc[counts.ne(1)].index.astype(str))
    unique_map = (
        global_map.loc[~global_map["player_id"].isin(ambiguous)]
        .drop_duplicates("player_id")
        .set_index("player_id")["player_clean_key"]
        .to_dict()
    )
    miss = out["player_clean_key"].isna() | out["player_clean_key"].astype(str).eq("")
    if miss.any():
        fallback = out.loc[miss, "player_id"].astype(str).map(unique_map)
        resolved = fallback.notna() & fallback.astype(str).ne("")
        idx = fallback.index[resolved]
        out.loc[idx, "player_clean_key"] = fallback.loc[idx]
        out.loc[idx, "identity_bridge_method"] = "UNIQUE_GSIS_HISTORY_FALLBACK"

    miss = out["player_clean_key"].isna() | out["player_clean_key"].astype(str).eq("")
    if miss.any():
        sample = out.loc[miss, ["season","week","team","player_id"]].head(20).to_dict("records")
        raise RuntimeError(f"route casebook identity bridge failed rows={int(miss.sum())} sample={sample}")
    if out.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("route casebook maps to duplicate canonical player-week identities")
    print(
        "[wr_route_matchup] identity_bridge "
        + out["identity_bridge_method"].value_counts(dropna=False).to_dict().__repr__()
    )
    return out


def _feature_map(cb2024: pd.DataFrame, week: int) -> dict:
    q = cb2024.loc[cb2024["week"].eq(int(week))].copy()
    return {
        (canon_team(r.team), str(r.player_clean_key)): {
            "delta_ypt": float(r.matchup_delta_ypt),
            "delta_catch": float(r.matchup_delta_catch),
            "player_id": str(r.player_id),
        }
        for r in q.itertuples(index=False)
    }


def _apply_arm(base: pd.DataFrame, fmap: dict, betas: dict, *, ypt: bool, catch: bool) -> tuple[pd.DataFrame, dict]:
    out = base.copy()
    base_ent = pd.to_numeric(base["entitlement_tgt_share"], errors="raise").to_numpy(float)
    matched = 0
    changed_ypt = changed_catch = 0
    for i, row in out.iterrows():
        if _pos_family(row.get("position")) != "WR":
            continue
        key = (canon_team(row.get("team")), str(row.get("player_clean_key")))
        f = fmap.get(key)
        if f is None:
            continue
        matched += 1
        if ypt:
            b = _first_finite(row, ["rules_ypt","bayes_ypt","ypt"], 7.5)
            v = float(np.clip(b + betas["beta_ypt"] * f["delta_ypt"], 2.0, 20.0))
            out.at[i, "rules_ypt"] = v
            changed_ypt += int(abs(v - b) > 1e-12)
        if catch:
            b = _first_finite(row, ["rules_catch_rate","bayes_receptions_per_target","receptions_per_target","catch_rate"], 0.64)
            v = float(np.clip(b + betas["beta_catch"] * f["delta_catch"], 0.001, 0.999))
            out.at[i, "rules_catch_rate"] = v
            changed_catch += int(abs(v - b) > 1e-12)

    ent = pd.to_numeric(out["entitlement_tgt_share"], errors="raise").to_numpy(float)
    ent_gap = float(np.max(np.abs(ent - base_ent))) if len(ent) else 0.0
    pos = base["position"].map(_pos_family)
    non_wr = ~pos.eq("WR")
    def _col_gap(col: str) -> float:
        a = pd.to_numeric(base.loc[non_wr, col], errors="coerce") if col in base.columns else pd.Series(np.nan, index=base.index[non_wr])
        b = pd.to_numeric(out.loc[non_wr, col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=base.index[non_wr])
        aa, bb = a.to_numpy(float), b.to_numpy(float)
        m = np.isfinite(aa) & np.isfinite(bb)
        return float(np.max(np.abs(aa[m]-bb[m]))) if m.any() else 0.0
    audit = {
        "matched_wr_players": int(matched),
        "changed_ypt_players": int(changed_ypt),
        "changed_catch_players": int(changed_catch),
        "target_share_max_abs_delta": ent_gap,
        "non_wr_rules_ypt_max_abs_delta": _col_gap("rules_ypt"),
        "non_wr_rules_catch_rate_max_abs_delta": _col_gap("rules_catch_rate"),
    }
    if max(audit["target_share_max_abs_delta"], audit["non_wr_rules_ypt_max_abs_delta"], audit["non_wr_rules_catch_rate_max_abs_delta"]) > 1e-12:
        raise RuntimeError(f"candidate seam integrity failure: {audit}")
    return out, audit


def _empirical_crps(draws: np.ndarray, actual: float) -> float:
    x = np.sort(np.asarray(draws, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    first = float(np.mean(np.abs(x - float(actual))))
    idx = np.arange(1, n + 1, dtype=float)
    half_pair = float(np.sum((2.0 * idx - n - 1.0) * x) / (n * n))
    return first - half_pair


def _actual_map(player_logs: pd.DataFrame, week: int) -> dict:
    q = player_logs.loc[
        player_logs["season"].eq(SEASON) & player_logs["week"].eq(int(week))
    ].copy()
    return {
        (canon_team(r.team), str(r.player_clean_key)): {
            "rec_yards": float(r.rec_yards),
            "receptions": float(r.receptions),
        }
        for r in q.itertuples(index=False)
    }


def _row_metrics(draws: np.ndarray, actual: float, prefix: str) -> dict:
    a = np.asarray(draws, dtype=float)
    lo80, hi80 = np.quantile(a, [0.10, 0.90])
    lo90, hi90 = np.quantile(a, [0.05, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_sd": float(np.std(a, ddof=1)),
        f"{prefix}_crps": _empirical_crps(a, actual),
        f"{prefix}_lo80": float(lo80), f"{prefix}_hi80": float(hi80),
        f"{prefix}_lo90": float(lo90), f"{prefix}_hi90": float(hi90),
    }


def _update_component(component: pd.DataFrame, updates: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = component.copy()
    if updates.empty:
        return out
    u = _canon_keys(updates[KEYS + [value_col]].copy())
    if u.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate update identities for {value_col}")
    z = out.merge(u, on=KEYS, how="left", validate="one_to_one")
    mask = z[value_col].notna()
    z.loc[mask, "mc_proj"] = pd.to_numeric(z.loc[mask, value_col], errors="raise")
    return z.drop(columns=[value_col])


def run(a) -> dict:
    if a.season != SEASON or a.prior_season != PRIOR_SEASON:
        raise RuntimeError("V1 is frozen to 2024 integration with 2023 prior season")
    player_logs = _read(a.player_logs, "player logs")
    team_weekly = _read(a.team_weekly, "historical team weekly")
    schedule = _read(a.schedule, "historical schedule")
    component = _canon_keys(_read(a.component_file, "canonical component baseline"))
    casebook = _read(a.casebook, "targeted-route Stage-B casebook")
    injuries_history, weather_history = _read_optional(a.injuries), _read_optional(a.weather)

    betas = _fit_betas(casebook)
    cb2024 = _bridge_casebook(casebook, player_logs)

    te_params = _load_fold_params(a.te_coefficients, test_season=2024, features=TE_FEATURES, label="TE-R5P")
    wr_params = _load_fold_params(a.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15")

    from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps
    snaps, snap_dup_rate, snap_source_seasons = _load_snaps()
    if snap_dup_rate > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {snap_dup_rate}")

    prod_updates = []
    ypt_updates = []
    catch_updates = []
    combined_updates = []
    direct_rows = []
    audits = []

    for week in _parse_weeks(a.weeks):
        universe = _read(a.universe_dir / f"2024_week_{week:02d}.csv", f"pregame universe 2024 W{week:02d}")
        bundle = build_historical_context_bundle(
            player_logs=player_logs, team_weekly=team_weekly, pregame_universe=universe,
            schedule=schedule, season=SEASON, week=int(week), prior_season=PRIOR_SEASON,
            injuries=_exact_week(injuries_history, SEASON, week),
            weather=_exact_week(weather_history, SEASON, week),
        )
        metrics = build_mc_predictions(bundle, iterations=int(a.iterations), seed=42+int(week))
        player_cols = ["event_id","team","player_clean_key"]
        players = metrics.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()
        explicit_base, _ = materialize_target_entitlement(players)
        te_final, _, te_audit = apply_te_fold(explicit_base, snaps=snaps, params=te_params)
        promoted, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)

        fmap = _feature_map(cb2024, week)
        ypt_state, ypt_audit = _apply_arm(promoted, fmap, betas, ypt=True, catch=False)
        catch_state, catch_audit = _apply_arm(promoted, fmap, betas, ypt=False, catch=True)
        combined_state, combined_audit = _apply_arm(promoted, fmap, betas, ypt=True, catch=True)

        seed = 42 + int(week)
        prod_sim = explicit_simulate(promoted, iterations=int(a.iterations), seed=seed)
        ypt_sim = explicit_simulate(ypt_state, iterations=int(a.iterations), seed=seed)
        catch_sim = explicit_simulate(catch_state, iterations=int(a.iterations), seed=seed)
        combined_sim = explicit_simulate(combined_state, iterations=int(a.iterations), seed=seed)

        base_team = explicit_base.groupby(["event_id","team"])["entitlement_tgt_share"].sum()
        prod_team = promoted.groupby(["event_id","team"])["entitlement_tgt_share"].sum()
        team_gap = float((prod_team-base_team).abs().max()) if len(base_team) else 0.0
        if team_gap > 1e-10:
            raise RuntimeError(f"promoted entitlement team mass drift W{week}: {team_gap}")

        actuals = _actual_map(player_logs, week)
        route_keys = set(fmap.keys())

        # Production baseline updates all authorized 2024 WR/TE receiving markets.
        for _, row in metrics.iterrows():
            market = str(row.get("market","")).lower()
            pos = _pos_family(row.get("position"))
            if market not in RECEIVING_MARKETS or pos not in {"WR","TE"}:
                continue
            arr = _historical_outcomes(prod_sim, row)
            if arr is None:
                continue
            keyrow = {
                "season": SEASON, "week": int(week), "team": canon_team(row.get("team")),
                "opponent": canon_team(row.get("opponent")), "player_clean_key": str(row.get("player_clean_key")),
                "market": market,
            }
            prod_updates.append({**keyrow, "prod_mc_proj": float(np.mean(arr))})

            route_key = (keyrow["team"], keyrow["player_clean_key"])
            if pos != "WR" or route_key not in route_keys:
                continue

            if market in {"rec_yards","rush_rec_yards"}:
                ya = _historical_outcomes(ypt_sim, row)
                ca = _historical_outcomes(combined_sim, row)
                if ya is not None:
                    ypt_updates.append({**keyrow, "ypt_mc_proj": float(np.mean(ya))})
                if ca is not None:
                    combined_updates.append({**keyrow, "combined_mc_proj": float(np.mean(ca))})
            elif market == "receptions":
                caa = _historical_outcomes(catch_sim, row)
                coa = _historical_outcomes(combined_sim, row)
                if caa is not None:
                    catch_updates.append({**keyrow, "catch_mc_proj": float(np.mean(caa))})
                if coa is not None:
                    combined_updates.append({**keyrow, "combined_mc_proj": float(np.mean(coa))})

        # Direct player-game diagnostics for primary markets.
        for _, prow in promoted.iterrows():
            if _pos_family(prow.get("position")) != "WR":
                continue
            tkey=(canon_team(prow.get("team")), str(prow.get("player_clean_key")))
            feat=fmap.get(tkey)
            act=actuals.get(tkey)
            if feat is None or act is None:
                continue
            event=str(prow.get("event_id"))
            pkey=str(prow.get("player_clean_key"))
            anchor=bool(prow.get("wr_r15_oos_anchor", False))
            for market, cand_sim, actual in (
                ("rec_yards", ypt_sim, act["rec_yards"]),
                ("receptions", catch_sim, act["receptions"]),
            ):
                b=prod_sim.values.get((event,pkey,market))
                c=cand_sim.values.get((event,pkey,market))
                if b is None or c is None:
                    continue
                rowd={
                    "season":SEASON,"week":int(week),"event_id":event,"team":tkey[0],
                    "player_clean_key":pkey,"player_id":feat["player_id"],"market":market,
                    "wr1_anchor":anchor,"actual":float(actual),
                    "matchup_delta_ypt":feat["delta_ypt"],"matchup_delta_catch":feat["delta_catch"],
                }
                rowd.update(_row_metrics(b,float(actual),"baseline"))
                rowd.update(_row_metrics(c,float(actual),"candidate"))
                direct_rows.append(rowd)

        audits.append({
            "season":SEASON,"week":int(week),"beta_ypt":betas["beta_ypt"],"beta_catch":betas["beta_catch"],
            "route_feature_players":int(len(fmap)),
            "te_pool_max_abs_gap":te_audit["team_te_pool_max_abs_gap"],
            "wr1_anchor_max_abs_gap":wr_audit["m38_wr1_anchor_max_abs_gap"],
            "wr2plus_pool_max_abs_gap":wr_audit["wr2plus_pool_max_abs_gap"],
            "wr_room_mass_max_abs_gap":wr_audit["wr_room_mass_max_abs_gap"],
            "team_modeled_mass_max_abs_gap":team_gap,
            "ypt_target_share_max_abs_delta":ypt_audit["target_share_max_abs_delta"],
            "catch_target_share_max_abs_delta":catch_audit["target_share_max_abs_delta"],
            "combined_target_share_max_abs_delta":combined_audit["target_share_max_abs_delta"],
            "ypt_non_wr_max_abs_delta":max(ypt_audit["non_wr_rules_ypt_max_abs_delta"],ypt_audit["non_wr_rules_catch_rate_max_abs_delta"]),
            "catch_non_wr_max_abs_delta":max(catch_audit["non_wr_rules_ypt_max_abs_delta"],catch_audit["non_wr_rules_catch_rate_max_abs_delta"]),
            "status":"PASS",
        })

    prod_u=pd.DataFrame(prod_updates); ypt_u=pd.DataFrame(ypt_updates); catch_u=pd.DataFrame(catch_updates); comb_u=pd.DataFrame(combined_updates)
    prod_component=_update_component(component,prod_u,"prod_mc_proj")
    ypt_component=_update_component(prod_component,ypt_u,"ypt_mc_proj")
    catch_component=_update_component(prod_component,catch_u,"catch_mc_proj")
    combined_component=_update_component(_update_component(prod_component,ypt_u,"ypt_mc_proj"),catch_u,"catch_mc_proj")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    prod_component.to_csv(a.out_dir/"component_2024_promoted_baseline.csv",index=False)
    ypt_component.to_csv(a.out_dir/"component_2024_route_ypt.csv",index=False)
    catch_component.to_csv(a.out_dir/"component_2024_route_catch.csv",index=False)
    combined_component.to_csv(a.out_dir/"component_2024_route_combined.csv",index=False)
    pd.DataFrame(direct_rows).to_csv(a.out_dir/"wr_route_matchup_direct_mc_rows.csv",index=False)
    pd.DataFrame(audits).to_csv(a.out_dir/"wr_route_matchup_integration_audit.csv",index=False)
    cb2024.to_csv(a.out_dir/"wr_route_matchup_2024_features.csv",index=False)
    (a.out_dir/"wr_route_matchup_betas.json").write_text(json.dumps(betas,indent=2,sort_keys=True)+"\n")

    result={
        "version":"WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1_GENERATOR",
        "betas":betas,
        "direct_rows":int(len(direct_rows)),
        "feature_player_games":int(len(cb2024)),
        "feature_players":int(cb2024["player_clean_key"].nunique()),
        "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_read":0,"production_changed":False,
    }
    (a.out_dir/"wr_route_matchup_generator_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    return result


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,default=2024)
    ap.add_argument("--prior-season",type=int,default=2023)
    ap.add_argument("--weeks",default="1-18")
    ap.add_argument("--iterations",type=int,default=2000)
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-dir",type=Path,required=True)
    ap.add_argument("--component-file",type=Path,required=True)
    ap.add_argument("--casebook",type=Path,required=True)
    ap.add_argument("--te-coefficients",type=Path,required=True)
    ap.add_argument("--wr-coefficients",type=Path,required=True)
    ap.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv"))
    ap.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv"))
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    run(a)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
