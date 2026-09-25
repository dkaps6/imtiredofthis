#!/usr/bin/env python3
"""Current-stack receiver compensation audit.

Diagnostic only. This study attributes where the already-qualified team-level
targetable-dropback signal is lost in the current player stack. It fits no
parameter and authorizes no candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps as _load_participation_snaps
from scripts.research.evaluate_receiver_targetable_dropback_v1_team_calibration import (
    build_team_actual_history,
    strict_prior_rate,
)
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    TE_FEATURES,
    WR_FEATURES,
    _load_fold_params,
    apply_te_fold,
    apply_wr_fold,
)
from scripts.simulation_c2_qb_candidate import simulate_with_states
from scripts.simulation_v2 import _clip_prob, _num

VERSION = "CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1"
ROOMS = ("WR", "TE", "RB_FB")
PLAYER_POSITIONS = ("WR", "TE", "RB")
TOL = 1e-9


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def pos(value: object) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR"):
        return "WR"
    if p == "TE" or p.startswith("TE"):
        return "TE"
    if p in {"RB", "HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p == "FB" or p.startswith("FB"):
        return "FB"
    return "OTHER"


def room(value: object) -> str:
    p = pos(value)
    if p == "WR":
        return "WR"
    if p == "TE":
        return "TE"
    if p in {"RB", "FB"}:
        return "RB_FB"
    return "OTHER"


def score(actual: pd.Series, pred: pd.Series) -> dict:
    z = pd.DataFrame({
        "actual": pd.to_numeric(actual, errors="coerce"),
        "pred": pd.to_numeric(pred, errors="coerce"),
    }).dropna()
    if z.empty:
        return {"n": 0}
    e = z["pred"].to_numpy(float) - z["actual"].to_numpy(float)
    ae = np.abs(e)
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "abs_bias": float(abs(e.mean())),
        "median_ae": float(np.quantile(ae, 0.50)),
        "p75_ae": float(np.quantile(ae, 0.75)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "corr": (
            float(np.corrcoef(z["actual"], z["pred"])[0, 1])
            if len(z) > 1 and z["actual"].std() > 0 and z["pred"].std() > 0
            else None
        ),
    }


def paired(actual: pd.Series, baseline: pd.Series, targetable: pd.Series) -> dict:
    z = pd.DataFrame({
        "actual": pd.to_numeric(actual, errors="coerce"),
        "baseline": pd.to_numeric(baseline, errors="coerce"),
        "targetable": pd.to_numeric(targetable, errors="coerce"),
    }).dropna()
    if z.empty:
        return {"baseline": {"n": 0}, "targetable": {"n": 0}}
    ba = np.abs(z["baseline"].to_numpy(float) - z["actual"].to_numpy(float))
    ca = np.abs(z["targetable"].to_numpy(float) - z["actual"].to_numpy(float))
    changed = np.abs(z["baseline"].to_numpy(float) - z["targetable"].to_numpy(float)) > 1e-12
    cw = changed & (ca < ba - 1e-12)
    bw = changed & (ba < ca - 1e-12)
    decided = cw | bw
    return {
        "baseline": score(z["actual"], z["baseline"]),
        "targetable": score(z["actual"], z["targetable"]),
        "changed_rows": int(changed.sum()),
        "targetable_closer": int(cw.sum()),
        "baseline_closer": int(bw.sum()),
        "targetable_closer_rate": float(cw.sum() / decided.sum()) if int(decided.sum()) else None,
    }


def actual_player_frame(player_logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    rec = build_actual_rows(player_logs, int(season), int(week))
    if rec.empty:
        return pd.DataFrame(columns=["team", "player_clean_key", "actual_targets", "actual_receptions", "actual_rec_yards"])
    y = rec.loc[rec["market"].eq("rec_yards"), ["team", "player_clean_key", "actual", "actual_opportunities"]].rename(
        columns={"actual": "actual_rec_yards", "actual_opportunities": "actual_targets"}
    )
    r = rec.loc[rec["market"].eq("receptions"), ["team", "player_clean_key", "actual"]].rename(
        columns={"actual": "actual_receptions"}
    )
    out = y.merge(r, on=["team", "player_clean_key"], how="outer", validate="one_to_one")
    for c in ("actual_targets", "actual_receptions", "actual_rec_yards"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["team"] = out["team"].map(canon_team)
    return out


def symmetric_decomp(t_actual, y_actual, t_pred, e_pred):
    ta = np.asarray(t_actual, dtype=float)
    ya = np.asarray(y_actual, dtype=float)
    tp = np.asarray(t_pred, dtype=float)
    ep0 = np.asarray(e_pred, dtype=float)
    if not (np.isfinite(ta).all() and np.isfinite(ya).all() and np.isfinite(tp).all() and np.isfinite(ep0).all()):
        raise RuntimeError("non-finite symmetric decomposition inputs")

    pred_y = tp * ep0
    ea = np.divide(ya, ta, out=ep0.copy(), where=ta > 0)
    opp = np.zeros_like(ya)
    eff = np.zeros_like(ya)

    # Frozen plan semantics:
    # - when both target counts are positive, use the symmetric product identity;
    # - when either side has zero targets, assign the entire yard residual to
    #   opportunity rather than inventing an infinite/undefined efficiency.
    regular = (ta > 0) & (tp > 0)
    ep = ep0.copy()
    opp[regular] = (ta[regular] - tp[regular]) * (ea[regular] + ep[regular]) / 2.0
    eff[regular] = (ea[regular] - ep[regular]) * (ta[regular] + tp[regular]) / 2.0

    edge = ~regular
    opp[edge] = ya[edge] - pred_y[edge]
    eff[edge] = 0.0

    gap = ya - pred_y - (opp + eff)
    max_gap = float(np.max(np.abs(gap))) if len(gap) else 0.0
    if max_gap > TOL:
        raise RuntimeError(f"symmetric decomposition identity failed max_gap={max_gap}")
    return ea, opp, eff, pred_y, max_gap


def evaluate_week(
    *,
    season: int,
    prior_season: int,
    week: int,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe: pd.DataFrame,
    injuries: pd.DataFrame,
    weather: pd.DataFrame,
    rate_history: pd.DataFrame,
    te_params: dict,
    wr_params: dict | None,
    snaps: pd.DataFrame,
    iterations: int,
):
    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=int(season),
        week=int(week),
        prior_season=int(prior_season),
        injuries=_exact_week(injuries, int(season), int(week)),
        weather=_exact_week(weather, int(season), int(week)),
    )
    seed = 42 + int(week)
    metrics = build_mc_predictions(bundle, iterations=20, seed=seed)
    players = (
        metrics.sort_values(["event_id", "team", "player_clean_key"])
        .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        .copy()
    )
    explicit, _ = materialize_target_entitlement(players)
    te_final, _, te_audit = apply_te_fold(explicit, snaps=snaps, params=te_params)
    if int(season) == 2024:
        final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
    else:
        final = te_final
        wr_audit = {
            "m38_wr1_anchor_max_abs_gap": 0.0,
            "wr2plus_pool_max_abs_gap": 0.0,
            "wr_room_mass_max_abs_gap": 0.0,
            "non_wr_max_abs_gap": 0.0,
            "same_future_participation": 0,
        }

    final = final.copy()
    final["team"] = final["team"].map(canon_team)
    sim_metrics = final.copy()
    final["position_family"] = final["position"].map(pos)
    final["room"] = final["position"].map(room)
    final = final.loc[final["room"].isin(ROOMS)].copy()
    if final.empty:
        raise RuntimeError(f"{season} W{week:02d} no receiver players")

    entitlement = pd.to_numeric(final["entitlement_tgt_share"], errors="coerce")
    if entitlement.isna().any() or not np.isfinite(entitlement.to_numpy(float)).all() or entitlement.lt(0).any():
        raise RuntimeError(f"{season} W{week:02d} invalid entitlement")
    final["entitlement_tgt_share"] = entitlement

    state = simulate_with_states(sim_metrics, iterations=int(iterations), seed=int(seed))
    actual = actual_player_frame(player_logs, int(season), int(week))
    final = final.merge(actual, on=["team", "player_clean_key"], how="left", validate="one_to_one")
    for c in ("actual_targets", "actual_receptions", "actual_rec_yards"):
        final[c] = pd.to_numeric(final[c], errors="coerce").fillna(0.0)

    # Current pregame efficiency/conversion inputs.
    pred_ypt = []
    pred_catch = []
    for _, row0 in final.iterrows():
        ypt = _num(row0, "rules_ypt", "bayes_ypt", "ypt")
        if not np.isfinite(ypt) or ypt <= 0:
            ypt = 7.5
        catch = _clip_prob(
            _num(row0, "rules_catch_rate", "bayes_receptions_per_target", "receptions_per_target", "catch_rate", default=0.64),
            0.64,
        )
        pred_ypt.append(float(ypt))
        pred_catch.append(float(catch))
    final["pred_ypt"] = pred_ypt
    final["pred_catch_rate"] = pred_catch

    player_rows = []
    room_rows = []
    team_rows = []
    max_decomp_gap = 0.0

    actual_team_lookup = rate_history.loc[
        rate_history["season"].eq(int(season)) & rate_history["week"].eq(int(week)),
        ["team", "actual_team_targets", "actual_dropbacks"],
    ].copy()
    actual_team_lookup["team"] = actual_team_lookup["team"].map(canon_team)
    actual_team_lookup = actual_team_lookup.set_index("team")

    for (game, team), tdf0 in final.groupby(["event_id", "team"], sort=True):
        game = str(game)
        team = canon_team(team)
        tdf = tdf0.copy().reset_index(drop=True)
        if team not in actual_team_lookup.index:
            raise RuntimeError(f"{season} W{week:02d} missing actual team history team={team}")

        dropbacks = np.asarray(state.team_states[(game, team, "pass_att")], dtype=float)
        db_mean = float(dropbacks.mean())
        rate_info = strict_prior_rate(rate_history, int(season), int(week), team, int(prior_season))
        rt = float(rate_info["targetable_dropback_rate"])
        if not np.isfinite(rt) or not 0 <= rt <= 1:
            raise RuntimeError(f"invalid targetable rate team={team} rate={rt}")
        candidate_pool = db_mean * rt

        p = pd.to_numeric(tdf["entitlement_tgt_share"], errors="raise").to_numpy(float)
        t0 = db_mean * p
        t1 = candidate_pool * p
        tdf["baseline_pred_targets"] = t0
        tdf["targetable_pred_targets"] = t1

        ea0, opp0, eff0, y0, gap0 = symmetric_decomp(
            tdf["actual_targets"], tdf["actual_rec_yards"], t0, tdf["pred_ypt"]
        )
        ea1, opp1, eff1, y1, gap1 = symmetric_decomp(
            tdf["actual_targets"], tdf["actual_rec_yards"], t1, tdf["pred_ypt"]
        )
        max_decomp_gap = max(max_decomp_gap, gap0, gap1)
        tdf["actual_ypt"] = ea0
        tdf["baseline_det_rec_yards"] = y0
        tdf["targetable_det_rec_yards"] = y1
        tdf["baseline_opp_contrib"] = opp0
        tdf["baseline_eff_contrib"] = eff0
        tdf["targetable_opp_contrib"] = opp1
        tdf["targetable_eff_contrib"] = eff1

        actual_catch = np.divide(
            tdf["actual_receptions"].to_numpy(float),
            tdf["actual_targets"].to_numpy(float),
            out=np.full(len(tdf), np.nan),
            where=tdf["actual_targets"].to_numpy(float) > 0,
        )
        tdf["actual_catch_rate"] = actual_catch

        # Current promoted WR hierarchy identity, descriptive only.
        tdf["wr_rank"] = np.nan
        wr_mask = tdf["room"].eq("WR")
        if wr_mask.any():
            ranks = tdf.loc[wr_mask, "entitlement_tgt_share"].rank(method="first", ascending=False)
            tdf.loc[wr_mask, "wr_rank"] = ranks.to_numpy(float)
        tdf["wr_role"] = np.where(
            tdf["room"].eq("WR"),
            np.where(tdf["wr_rank"].eq(1), "WR1", "WR2+"),
            "",
        )

        modeled_actual_targets = float(tdf["actual_targets"].sum())
        complete_actual_targets = float(actual_team_lookup.loc[team, "actual_team_targets"])
        team_rows.append({
            "season": int(season),
            "week": int(week),
            "event_id": game,
            "team": team,
            "baseline_team_dropbacks": db_mean,
            "targetable_rate": rt,
            "targetable_team_pool": candidate_pool,
            "actual_team_targets_complete": complete_actual_targets,
            "actual_modeled_player_targets": modeled_actual_targets,
            "modeled_actual_target_coverage": (
                modeled_actual_targets / complete_actual_targets if complete_actual_targets > 0 else 1.0
            ),
            "conversion_source": str(rate_info["conversion_source"]),
            "prior_history_games": int(rate_info["prior_history_games"]),
        })

        modeled_entitlement = float(tdf["entitlement_tgt_share"].sum())
        modeled_actual = float(tdf["actual_targets"].sum())

        for room_name in ROOMS:
            r = tdf.loc[tdf["room"].eq(room_name)].copy()
            if r.empty:
                continue
            room_ent = float(r["entitlement_tgt_share"].sum())
            actual_room = float(r["actual_targets"].sum())
            base_room = db_mean * room_ent
            target_room = candidate_pool * room_ent
            pred_share = room_ent / modeled_entitlement if modeled_entitlement > 0 else np.nan
            actual_share = actual_room / modeled_actual if modeled_actual > 0 else np.nan
            room_rows.append({
                "season": int(season),
                "week": int(week),
                "event_id": game,
                "team": team,
                "room": room_name,
                "actual_room_targets": actual_room,
                "baseline_room_targets": base_room,
                "targetable_room_targets": target_room,
                "pred_room_share": pred_share,
                "actual_room_share": actual_share,
                "room_share_error_pred_minus_actual": pred_share - actual_share if np.isfinite(pred_share) and np.isfinite(actual_share) else np.nan,
            })
            if actual_room > 0 and room_ent > 0:
                idx = r.index
                tdf.loc[idx, "pred_within_room_share"] = (
                    r["entitlement_tgt_share"].to_numpy(float) / room_ent
                )
                tdf.loc[idx, "actual_within_room_share"] = (
                    r["actual_targets"].to_numpy(float) / actual_room
                )
            else:
                idx = r.index
                tdf.loc[idx, "pred_within_room_share"] = np.nan
                tdf.loc[idx, "actual_within_room_share"] = np.nan

        keep_cols = [
            "season", "week", "event_id", "team", "player", "player_clean_key",
            "position", "position_family", "room", "entitlement_tgt_share",
            "wr_rank", "wr_role", "actual_targets", "actual_receptions",
            "actual_rec_yards", "baseline_pred_targets", "targetable_pred_targets",
            "pred_catch_rate", "actual_catch_rate", "pred_ypt", "actual_ypt",
            "baseline_det_rec_yards", "targetable_det_rec_yards",
            "baseline_opp_contrib", "baseline_eff_contrib",
            "targetable_opp_contrib", "targetable_eff_contrib",
            "pred_within_room_share", "actual_within_room_share",
        ]
        player_rows.append(tdf[keep_cols].copy())

    scope = {
        "season": int(season),
        "week": int(week),
        "te_pool_gap": float(te_audit["team_te_pool_max_abs_gap"]),
        "te_non_te_gap": float(te_audit["non_te_max_abs_gap"]),
        "wr1_anchor_gap": float(wr_audit.get("m38_wr1_anchor_max_abs_gap", 0.0)),
        "wr2plus_pool_gap": float(wr_audit.get("wr2plus_pool_max_abs_gap", 0.0)),
        "wr_room_gap": float(wr_audit.get("wr_room_mass_max_abs_gap", 0.0)),
        "wr_non_wr_gap": float(wr_audit.get("non_wr_max_abs_gap", 0.0)),
        "wr_same_future_participation": int(wr_audit.get("same_future_participation", 0)),
        "max_decomposition_identity_gap": float(max_decomp_gap),
    }
    return pd.concat(player_rows, ignore_index=True), pd.DataFrame(room_rows), pd.DataFrame(team_rows), scope


def summarize(player: pd.DataFrame, rooms: pd.DataFrame, teams: pd.DataFrame) -> dict:
    # Descriptive entitlement quartiles.
    player = player.copy()
    player["entitlement_quartile"] = ""
    for season in sorted(player["season"].unique()):
        q = player.loc[player["season"].eq(season), ["event_id", "team", "player_clean_key", "entitlement_tgt_share"]].drop_duplicates()
        q["quartile"] = pd.qcut(
            q["entitlement_tgt_share"].rank(method="first"),
            4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        ).astype(str)
        qmap = q.set_index(["event_id", "team", "player_clean_key"])["quartile"].to_dict()
        mask = player["season"].eq(season)
        player.loc[mask, "entitlement_quartile"] = [
            qmap.get((str(r.event_id), str(r.team), str(r.player_clean_key)), "")
            for r in player.loc[mask, ["event_id", "team", "player_clean_key"]].itertuples(index=False)
        ]

    out = {"team_volume": {}, "rooms": {}, "within_room": {}, "player_targets": {}, "efficiency": {}, "yard_decomposition": {}}

    for label, frame in [("pooled", teams)] + [(str(s), teams.loc[teams["season"].eq(s)]) for s in (2024, 2025)]:
        out["team_volume"][label] = paired(
            frame["actual_team_targets_complete"],
            frame["baseline_team_dropbacks"],
            frame["targetable_team_pool"],
        )
        out["team_volume"][label]["modeled_actual_target_coverage_median"] = float(frame["modeled_actual_target_coverage"].median())

    for room_name in ROOMS:
        r0 = rooms.loc[rooms["room"].eq(room_name)]
        out["rooms"][room_name] = {}
        for label, frame in [("pooled", r0)] + [(str(s), r0.loc[r0["season"].eq(s)]) for s in (2024, 2025)]:
            out["rooms"][room_name][label] = paired(
                frame["actual_room_targets"],
                frame["baseline_room_targets"],
                frame["targetable_room_targets"],
            )
            out["rooms"][room_name][label]["room_share"] = score(
                frame["actual_room_share"], frame["pred_room_share"]
            )

    for room_name in ROOMS:
        p0 = player.loc[player["room"].eq(room_name) & player["actual_within_room_share"].notna()]
        out["within_room"][room_name] = score(
            p0["actual_within_room_share"], p0["pred_within_room_share"]
        )

    for p in PLAYER_POSITIONS:
        p0 = player.loc[player["position_family"].eq(p)]
        out["player_targets"][p] = paired(
            p0["actual_targets"], p0["baseline_pred_targets"], p0["targetable_pred_targets"]
        )
        nz = p0.loc[p0["actual_targets"].gt(0)]
        out["efficiency"][p] = {
            "ypt": score(nz["actual_ypt"], nz["pred_ypt"]),
            "catch_rate": score(nz["actual_catch_rate"], nz["pred_catch_rate"]),
        }
        out["yard_decomposition"][p] = {
            "baseline_det_yards": score(p0["actual_rec_yards"], p0["baseline_det_rec_yards"]),
            "targetable_det_yards": score(p0["actual_rec_yards"], p0["targetable_det_rec_yards"]),
            "baseline_mean_signed_opportunity": float(p0["baseline_opp_contrib"].mean()),
            "baseline_mean_signed_efficiency": float(p0["baseline_eff_contrib"].mean()),
            "baseline_mean_abs_opportunity": float(p0["baseline_opp_contrib"].abs().mean()),
            "baseline_mean_abs_efficiency": float(p0["baseline_eff_contrib"].abs().mean()),
            "targetable_mean_signed_opportunity": float(p0["targetable_opp_contrib"].mean()),
            "targetable_mean_signed_efficiency": float(p0["targetable_eff_contrib"].mean()),
            "targetable_mean_abs_opportunity": float(p0["targetable_opp_contrib"].abs().mean()),
            "targetable_mean_abs_efficiency": float(p0["targetable_eff_contrib"].abs().mean()),
        }

    q4 = player.loc[player["entitlement_quartile"].eq("Q4_high") & player["position_family"].isin(PLAYER_POSITIONS)]
    out["q4"] = {
        "targets": paired(q4["actual_targets"], q4["baseline_pred_targets"], q4["targetable_pred_targets"]),
        "det_yards": paired(q4["actual_rec_yards"], q4["baseline_det_rec_yards"], q4["targetable_det_rec_yards"]),
        "targetable_mean_signed_opportunity": float(q4["targetable_opp_contrib"].mean()),
        "targetable_mean_signed_efficiency": float(q4["targetable_eff_contrib"].mean()),
        "targetable_mean_abs_opportunity": float(q4["targetable_opp_contrib"].abs().mean()),
        "targetable_mean_abs_efficiency": float(q4["targetable_eff_contrib"].abs().mean()),
    }

    wr = player.loc[player["room"].eq("WR")]
    out["wr_roles"] = {}
    for role in ("WR1", "WR2+"):
        w = wr.loc[wr["wr_role"].eq(role)]
        out["wr_roles"][role] = {
            "targets": paired(w["actual_targets"], w["baseline_pred_targets"], w["targetable_pred_targets"]),
            "within_room_share": score(w["actual_within_room_share"], w["pred_within_room_share"]),
            "det_yards": paired(w["actual_rec_yards"], w["baseline_det_rec_yards"], w["targetable_det_rec_yards"]),
        }

    return out, player


def compensation_answers(summary: dict) -> dict:
    def mae(pair_obj, arm):
        return float(pair_obj[arm]["mae"])
    answers = {}
    # Team volume improves?
    tv = summary["team_volume"]["pooled"]
    answers["team_target_volume_improves"] = mae(tv, "targetable") < mae(tv, "baseline")

    for room_name in ROOMS:
        r = summary["rooms"][room_name]["pooled"]
        answers[f"{room_name.lower()}_room_target_error_improves"] = mae(r, "targetable") < mae(r, "baseline")

    for p in PLAYER_POSITIONS:
        t = summary["player_targets"][p]
        y = summary["yard_decomposition"][p]
        answers[f"{p.lower()}_player_target_mae_improves"] = mae(t, "targetable") < mae(t, "baseline")
        answers[f"{p.lower()}_det_yard_mae_improves"] = (
            float(y["targetable_det_yards"]["mae"]) < float(y["baseline_det_yards"]["mae"])
        )
        answers[f"{p.lower()}_pred_ypt_negative_bias"] = float(summary["efficiency"][p]["ypt"]["bias"]) < 0

    q = summary["q4"]
    answers["q4_target_mae_improves"] = mae(q["targets"], "targetable") < mae(q["targets"], "baseline")
    answers["q4_det_yard_mae_improves"] = mae(q["det_yards"], "targetable") < mae(q["det_yards"], "baseline")
    return answers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2024", type=Path, required=True)
    ap.add_argument("--universe-2025", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--te-coefficients", type=Path, required=True)
    ap.add_argument("--wr-coefficients", type=Path, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    logs = read(args.player_logs, "player logs")
    team = read(args.team_weekly, "team weekly")
    sched = read(args.schedule, "schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)
    rate_history = build_team_actual_history(logs, team)
    snaps, dup, _ = _load_participation_snaps()
    if dup > 0.01:
        raise RuntimeError(f"participation duplicate rate too high: {dup}")

    te24 = _load_fold_params(args.te_coefficients, test_season=2024, features=TE_FEATURES, label="TE-R5P")
    te25 = _load_fold_params(args.te_coefficients, test_season=2025, features=TE_FEATURES, label="TE-R5P")
    wr24 = _load_fold_params(args.wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15")
    weeks = _parse_weeks(args.weeks)

    player_parts = []
    room_parts = []
    team_parts = []
    scopes = []

    for season, prior, universe_dir, te_params, wr_params in [
        (2024, 2023, args.universe_2024, te24, wr24),
        (2025, 2024, args.universe_2025, te25, None),
    ]:
        for week in weeks:
            universe = read(universe_dir / f"{season}_week_{int(week):02d}.csv", f"{season} W{week} universe")
            p, r, t, s = evaluate_week(
                season=season,
                prior_season=prior,
                week=int(week),
                player_logs=logs,
                team_weekly=team,
                schedule=sched,
                universe=universe,
                injuries=injuries,
                weather=weather,
                rate_history=rate_history,
                te_params=te_params,
                wr_params=wr_params,
                snaps=snaps,
                iterations=int(args.iterations),
            )
            player_parts.append(p)
            room_parts.append(r)
            team_parts.append(t)
            scopes.append(s)

    player = pd.concat(player_parts, ignore_index=True)
    rooms = pd.concat(room_parts, ignore_index=True)
    teams = pd.concat(team_parts, ignore_index=True)
    scope = pd.DataFrame(scopes)

    if float(scope["max_decomposition_identity_gap"].max()) > TOL:
        raise RuntimeError("decomposition identity gate failed")
    if int(scope["wr_same_future_participation"].sum()) != 0:
        raise RuntimeError("WR future participation leakage")
    if float(scope[["te_pool_gap", "te_non_te_gap", "wr1_anchor_gap", "wr2plus_pool_gap", "wr_room_gap", "wr_non_wr_gap"]].max().max()) > 1e-9:
        raise RuntimeError("specialist conservation gate failed")

    summary, player = summarize(player, rooms, teams)
    answers = compensation_answers(summary)

    payload = {
        "version": VERSION,
        "disposition": "CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1_COMPLETE",
        "production_changed": False,
        "parameters_fit": 0,
        "candidate_variants_scored": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_upstream": 0,
        "iterations": int(args.iterations),
        "player_rows": int(len(player)),
        "room_rows": int(len(rooms)),
        "team_games": int(len(teams)),
        "max_decomposition_identity_gap": float(scope["max_decomposition_identity_gap"].max()),
        "wr_same_future_participation": int(scope["wr_same_future_participation"].sum()),
        "summary": summary,
        "descriptive_answers": answers,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    player.to_csv(args.out_dir / "player_compensation_detail.csv", index=False)
    rooms.to_csv(args.out_dir / "room_mass_detail.csv", index=False)
    teams.to_csv(args.out_dir / "team_volume_detail.csv", index=False)
    scope.to_csv(args.out_dir / "integrity_scope.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Current-Stack Receiver Compensation Audit V1",
        "",
        "Disposition: **CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1_COMPLETE**",
        "",
        "## Descriptive answers",
        "",
    ]
    lines += [f"- {k}: **{v}**" for k, v in answers.items()]
    lines += ["", "## Team volume", ""]
    tv = summary["team_volume"]["pooled"]
    lines += [
        f"- MAE: {tv['baseline']['mae']:.6f} -> {tv['targetable']['mae']:.6f}",
        f"- p90: {tv['baseline']['p90_ae']:.6f} -> {tv['targetable']['p90_ae']:.6f}",
        f"- targetable closer rate: {tv['targetable_closer_rate']:.6f}",
        "",
        "## Room target MAE",
        "",
    ]
    for rr in ROOMS:
        x = summary["rooms"][rr]["pooled"]
        lines.append(f"- {rr}: {x['baseline']['mae']:.6f} -> {x['targetable']['mae']:.6f}; share bias={x['room_share']['bias']:.6f}")
    lines += ["", "## Player target / deterministic yard MAE", ""]
    for p in PLAYER_POSITIONS:
        t = summary["player_targets"][p]
        y = summary["yard_decomposition"][p]
        lines.append(
            f"- {p}: targets {t['baseline']['mae']:.6f} -> {t['targetable']['mae']:.6f}; "
            f"det yards {y['baseline_det_yards']['mae']:.6f} -> {y['targetable_det_yards']['mae']:.6f}; "
            f"YPT bias={summary['efficiency'][p]['ypt']['bias']:.6f}"
        )
    lines += [
        "",
        "## Q4",
        "",
        f"- targets MAE: {summary['q4']['targets']['baseline']['mae']:.6f} -> {summary['q4']['targets']['targetable']['mae']:.6f}",
        f"- deterministic yards MAE: {summary['q4']['det_yards']['baseline']['mae']:.6f} -> {summary['q4']['det_yards']['targetable']['mae']:.6f}",
        "",
        f"Maximum symmetric decomposition identity gap: {payload['max_decomposition_identity_gap']:.12g}",
        "",
    ]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
