#!/usr/bin/env python3
"""R26K Week-1 allocation mechanism atlas.

No refit and no prediction generation. Joins immutable R26 predictions to immutable
R26J pregame room state and grades only the football states frozen in the R26K plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = tuple(range(2020, 2026))
TOL = 1e-12

PRIMARY_STATES = (
    "K1_LARGE_COMPLEX_ROOM",
    "K2_VETERAN_PRESSURE_FLAT_HIERARCHY",
    "K3_HIGH_VACATED_LOAD_FLAT_HIERARCHY",
    "K4_MANY_CLAIMANTS_HIGH_VACATED_LOAD",
    "K5_VETERAN_PRESSURE_HIGH_VACATED_LOAD",
)

SECONDARY_STATES = (
    "A_CURRENT_ROOM_GE5",
    "A_CONTINUING_GE3",
    "A_ENTRANTS_GE3",
    "B_VETERAN_ENTRY_GE1",
    "D_TOP_SHARE_LE_THIRD",
    "D_HHI_LE_027",
    "C_SUM_EXIT_LAST8_GE3",
    "C_EXIT_HISTORY_COVERAGE_GE095",
)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_many(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if len(paths) != 6:
        raise RuntimeError(f"expected exactly 6 {name} files, found {len(paths)}")
    return pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True, sort=False)


def read_one_csv(root: Path, name: str) -> pd.DataFrame:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(hits)}")
    return pd.read_csv(hits[0], low_memory=False)


def read_one_json(root: Path, name: str) -> dict:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(hits)}")
    return json.loads(hits[0].read_text())


def metrics(g: pd.DataFrame, market: str) -> dict:
    a = num(g[f"actual_{market}"]).to_numpy(float)
    b = num(g[f"baseline_{market}"]).to_numpy(float)
    c = num(g[f"candidate_{market}"]).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[ok], b[ok], c[ok]
    if len(a) == 0:
        return {
            "n": 0, "baseline_mae": np.nan, "candidate_mae": np.nan,
            "relative_mae_change": np.nan, "baseline_rmse": np.nan,
            "candidate_rmse": np.nan, "baseline_bias": np.nan,
            "candidate_bias": np.nan, "baseline_p90": np.nan,
            "candidate_p90": np.nan, "mean_effect_delta": np.nan,
            "sum_effect_delta": np.nan,
        }
    be = b - a
    ce = c - a
    bae = np.abs(be)
    cae = np.abs(ce)
    bmae = float(bae.mean())
    cmae = float(cae.mean())
    return {
        "n": int(len(a)),
        "baseline_mae": bmae,
        "candidate_mae": cmae,
        "relative_mae_change": float(cmae / bmae - 1.0) if bmae > TOL else np.nan,
        "baseline_rmse": float(np.sqrt(np.mean(be * be))),
        "candidate_rmse": float(np.sqrt(np.mean(ce * ce))),
        "baseline_bias": float(be.mean()),
        "candidate_bias": float(ce.mean()),
        "baseline_p90": float(np.quantile(bae, .90)),
        "candidate_p90": float(np.quantile(cae, .90)),
        "mean_effect_delta": float((cae - bae).mean()),
        "sum_effect_delta": float((cae - bae).sum()),
    }


def attach_states(room: pd.DataFrame) -> pd.DataFrame:
    x = room.copy()
    numeric = [
        "current_room_n", "continuing_n", "entrants_n", "veteran_entry_n",
        "baseline_top_room_share", "baseline_room_hhi", "sum_exit_last8_targets_pg",
        "exit_history_coverage",
    ]
    for c in numeric:
        if c not in x.columns:
            raise RuntimeError(f"R26K missing R26J room field {c}")
        x[c] = num(x[c])

    x["K1_LARGE_COMPLEX_ROOM"] = (
        x.current_room_n.ge(5) & x.continuing_n.ge(2) & x.entrants_n.ge(2)
    )
    x["K2_VETERAN_PRESSURE_FLAT_HIERARCHY"] = (
        x.veteran_entry_n.ge(1) & x.baseline_top_room_share.le(1.0 / 3.0)
    )
    x["K3_HIGH_VACATED_LOAD_FLAT_HIERARCHY"] = (
        x.sum_exit_last8_targets_pg.ge(3.0) & x.baseline_top_room_share.le(1.0 / 3.0)
    )
    x["K4_MANY_CLAIMANTS_HIGH_VACATED_LOAD"] = (
        x.continuing_n.ge(2) & x.entrants_n.ge(2) & x.sum_exit_last8_targets_pg.ge(3.0)
    )
    x["K5_VETERAN_PRESSURE_HIGH_VACATED_LOAD"] = (
        x.veteran_entry_n.ge(1) & x.sum_exit_last8_targets_pg.ge(3.0)
    )

    x["A_CURRENT_ROOM_GE5"] = x.current_room_n.ge(5)
    x["A_CONTINUING_GE3"] = x.continuing_n.ge(3)
    x["A_ENTRANTS_GE3"] = x.entrants_n.ge(3)
    x["B_VETERAN_ENTRY_GE1"] = x.veteran_entry_n.ge(1)
    x["D_TOP_SHARE_LE_THIRD"] = x.baseline_top_room_share.le(1.0 / 3.0)
    x["D_HHI_LE_027"] = x.baseline_room_hhi.le(0.27)
    x["C_SUM_EXIT_LAST8_GE3"] = x.sum_exit_last8_targets_pg.ge(3.0)
    x["C_EXIT_HISTORY_COVERAGE_GE095"] = x.exit_history_coverage.ge(0.95)
    return x


def room_diagnostics(full_rows: pd.DataFrame, state: str, period: str) -> tuple[pd.DataFrame, dict]:
    g = full_rows.loc[full_rows[state]].copy()
    if period == "2020":
        g = g.loc[g.season.eq(2020)]
    elif period == "2021_2025":
        g = g.loc[g.season.ge(2021)]
    elif period == "ALL":
        pass
    else:
        raise ValueError(period)

    rows = []
    for (season, team), r in g.groupby(["season", "team"], sort=True):
        z = r.loc[
            r.actual_receptions.notna()
            & r.baseline_receptions.notna()
            & r.candidate_receptions.notna()
        ].copy()
        if z.empty:
            continue
        actual = float(num(z.actual_receptions).sum())
        base = float(num(z.baseline_receptions).sum())
        cand = float(num(z.candidate_receptions).sum())
        bplayer = float((num(z.baseline_receptions) - num(z.actual_receptions)).abs().sum())
        cplayer = float((num(z.candidate_receptions) - num(z.actual_receptions)).abs().sum())
        rows.append({
            "state": state,
            "period": period,
            "season": int(season),
            "team": str(team),
            "n_players": int(len(z)),
            "baseline_room_total_ae": abs(base - actual),
            "candidate_room_total_ae": abs(cand - actual),
            "room_total_effect_delta": abs(cand - actual) - abs(base - actual),
            "baseline_summed_player_ae": bplayer,
            "candidate_summed_player_ae": cplayer,
            "summed_player_effect_delta": cplayer - bplayer,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {
            "room_n": 0,
            "room_total_effect_delta_sum": np.nan,
            "summed_player_effect_delta_sum": np.nan,
            "classification": "NO_DATA",
        }
    room_delta = float(out.room_total_effect_delta.sum())
    player_delta = float(out.summed_player_effect_delta.sum())
    if player_delta > 0 and (room_delta <= 0 or room_delta <= 0.25 * player_delta):
        cls = "ALLOCATION_DOMINANT"
    elif player_delta > 0 and room_delta > 0.25 * player_delta:
        cls = "ROOM_TOTAL_AND_ALLOCATION_BOTH_HARMFUL"
    elif player_delta <= 0:
        cls = "PLAYER_ALLOCATION_NONHARMFUL"
    else:
        cls = "MIXED"
    return out, {
        "room_n": int(out[["season", "team"]].drop_duplicates().shape[0]),
        "room_total_effect_delta_sum": room_delta,
        "summed_player_effect_delta_sum": player_delta,
        "classification": cls,
    }


def ordering_diagnostics(inc: pd.DataFrame, state: str, period: str) -> dict:
    g = inc.loc[inc[state]].copy()
    if period == "2020":
        g = g.loc[g.season.eq(2020)]
    elif period == "2021_2025":
        g = g.loc[g.season.ge(2021)]
    elif period != "ALL":
        raise ValueError(period)

    rows = []
    for (season, team), r in g.groupby(["season", "team"], sort=True):
        z = r.loc[
            r.actual_receptions.notna()
            & r.baseline_receptions.notna()
            & r.candidate_receptions.notna()
        ].copy()
        if len(z) < 2:
            continue
        z["player_clean_key"] = z.player_clean_key.astype(str)
        actual = z.sort_values(["actual_receptions", "player_clean_key"], ascending=[False, True]).iloc[0].player_clean_key
        base = z.sort_values(["baseline_receptions", "player_clean_key"], ascending=[False, True]).iloc[0].player_clean_key
        cand = z.sort_values(["candidate_receptions", "player_clean_key"], ascending=[False, True]).iloc[0].player_clean_key
        rows.append({
            "season": int(season), "team": str(team),
            "baseline_correct": int(base == actual),
            "candidate_correct": int(cand == actual),
        })
    if not rows:
        return {"room_n": 0, "baseline_top_accuracy": np.nan, "candidate_top_accuracy": np.nan}
    d = pd.DataFrame(rows)
    return {
        "room_n": int(len(d)),
        "baseline_top_accuracy": float(d.baseline_correct.mean()),
        "candidate_top_accuracy": float(d.candidate_correct.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--r26j-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    if not a.protected_clean_marker.exists() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26K protected-production clean marker missing")

    jdisp = read_one_json(a.r26j_root, "r26j_source_disposition.json")
    if jdisp.get("disposition") != "2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP":
        raise RuntimeError("R26K R26J parent disposition mismatch")
    if jdisp.get("all_integrity_gates_pass") is not True:
        raise RuntimeError("R26K R26J parent integrity mismatch")
    if jdisp.get("selected_actual_fields") != []:
        raise RuntimeError("R26K R26J selected actual fields")
    if int(jdisp.get("sportsbook_inputs_used", 1)) != 0:
        raise RuntimeError("R26K R26J sportsbook contract violated")

    room = read_one_csv(a.r26j_root, "r26j_week1_vacancy_room_source_state.csv")
    room["season"] = num(room.season).astype(int)
    room["week"] = num(room.week).astype(int)
    room = room.loc[room.season.isin(SEASONS) & room.week.eq(1)].copy()
    room = attach_states(room)
    if room.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("R26K duplicate R26J room keys")

    pred = read_many(a.r26_root, "r26_predictions.csv")
    pred["season"] = num(pred.season).astype(int)
    pred["week"] = num(pred.week).astype(int)
    needed_num = [
        "vacancy_active", "continuing_same_team", "rb_rank",
        "actual_receptions", "baseline_receptions", "candidate_receptions",
        "actual_targets", "baseline_targets", "candidate_targets",
        "sportsbook_inputs_used", "future_outcomes_used",
    ]
    missing = [c for c in needed_num + ["team", "player_clean_key"] if c not in pred.columns]
    if missing:
        raise RuntimeError(f"R26K missing R26 columns: {missing}")
    for c in needed_num:
        pred[c] = num(pred[c])
    pred = pred.loc[pred.season.isin(SEASONS) & pred.week.eq(1) & pred.vacancy_active.eq(1)].copy()

    state_cols = ["season", "week", "team", *PRIMARY_STATES, *SECONDARY_STATES]
    pred = pred.merge(room[state_cols], on=["season", "week", "team"], how="left", validate="many_to_one")
    if pred[list(PRIMARY_STATES)].isna().any().any():
        raise RuntimeError("R26K missing R26J state after join")
    for c in PRIMARY_STATES + SECONDARY_STATES:
        pred[c] = pred[c].astype(bool)

    sportsbook = int(num(pred.sportsbook_inputs_used).fillna(0).sum())
    future = int(num(pred.future_outcomes_used).fillna(0).sum())
    inc = pred.loc[pred.continuing_same_team.eq(1)].copy()
    inc = inc.loc[
        inc.actual_receptions.notna()
        & inc.baseline_receptions.notna()
        & inc.candidate_receptions.notna()
    ].copy()
    if inc.empty:
        raise RuntimeError("R26K zero labeled Week-1 vacancy incumbents")

    inc["baseline_reception_ae"] = (inc.baseline_receptions - inc.actual_receptions).abs()
    inc["candidate_reception_ae"] = (inc.candidate_receptions - inc.actual_receptions).abs()
    inc["reception_effect_delta"] = inc.candidate_reception_ae - inc.baseline_reception_ae
    total20 = float(inc.loc[inc.season.eq(2020), "reception_effect_delta"].sum())
    if total20 <= 0:
        raise RuntimeError(f"R26K expected positive 2020 net R26 worsening, got {total20}")

    by_season_rows = []
    primary_rows = []
    room_rows_all = []
    ordering_rows = []
    qualified = []
    material20 = []

    integrity = {
        "parent_digests_verified_by_workflow": True,
        "protected_production_files_clean": True,
        "r26_predictions_regenerated": False,
        "r9_refit": False,
        "sportsbook_inputs_added": 0,
        "sportsbook_inputs_in_parent_rows": sportsbook,
        "future_outcome_features_in_parent_rows": future,
        "same_week_historical_depth_added": False,
        "production_parameters_changed": False,
        "receiving_yard_means_changed": False,
        "r22_changed": False,
        "r26j_integrity_inherited": True,
        "required_seasons_present": sorted(int(x) for x in inc.season.unique()) == list(SEASONS),
        "room_state_join_coverage": float(pred[list(PRIMARY_STATES)].notna().all(axis=1).mean()),
    }
    integrity_ok = bool(
        integrity["parent_digests_verified_by_workflow"]
        and integrity["protected_production_files_clean"]
        and integrity["r26_predictions_regenerated"] is False
        and integrity["r9_refit"] is False
        and sportsbook == 0
        and future == 0
        and integrity["same_week_historical_depth_added"] is False
        and integrity["production_parameters_changed"] is False
        and integrity["receiving_yard_means_changed"] is False
        and integrity["r22_changed"] is False
        and integrity["required_seasons_present"]
        and integrity["room_state_join_coverage"] >= .999999
    )

    for state in PRIMARY_STATES:
        for season in SEASONS:
            g = inc.loc[inc.season.eq(season) & inc[state]].copy()
            rec = metrics(g, "receptions")
            tgt = metrics(g, "targets")
            rb1 = metrics(g.loc[g.rb_rank.eq(1)], "receptions")
            rb2 = metrics(g.loc[~g.rb_rank.eq(1)], "receptions")
            by_season_rows.append({
                "state": state, "season": season,
                **{f"rec_{k}": v for k, v in rec.items()},
                **{f"tgt_{k}": v for k, v in tgt.items()},
                "rb1_rec_n": rb1["n"], "rb1_rec_baseline_mae": rb1["baseline_mae"], "rb1_rec_candidate_mae": rb1["candidate_mae"],
                "rb2plus_rec_n": rb2["n"], "rb2plus_rec_baseline_mae": rb2["baseline_mae"], "rb2plus_rec_candidate_mae": rb2["candidate_mae"],
            })

        g20 = inc.loc[inc.season.eq(2020) & inc[state]].copy()
        gout = inc.loc[inc.season.ge(2021) & inc[state]].copy()
        gall = inc.loc[inc[state]].copy()
        rec20 = metrics(g20, "receptions")
        recout = metrics(gout, "receptions")
        recall = metrics(gall, "receptions")
        tgtall = metrics(gall, "targets")
        rb1all = metrics(gall.loc[gall.rb_rank.eq(1)], "receptions")
        rb2all = metrics(gall.loc[~gall.rb_rank.eq(1)], "receptions")

        rr20, room20 = room_diagnostics(pred, state, "2020")
        rrout, roomout = room_diagnostics(pred, state, "2021_2025")
        if not rr20.empty:
            room_rows_all.append(rr20)
        if not rrout.empty:
            room_rows_all.append(rrout)

        ord20 = ordering_diagnostics(inc, state, "2020")
        ordout = ordering_diagnostics(inc, state, "2021_2025")
        ordering_rows.extend([
            {"state": state, "period": "2020", **ord20},
            {"state": state, "period": "2021_2025", **ordout},
        ])

        harm_seasons = []
        harm_n = 0
        for season in range(2021, 2026):
            gs = gout.loc[gout.season.eq(season)]
            ms = metrics(gs, "receptions")
            if ms["n"] > 0 and np.isfinite(ms["mean_effect_delta"]) and ms["mean_effect_delta"] > 0:
                harm_seasons.append(season)
                harm_n += int(ms["n"])

        contribution = float(rec20["sum_effect_delta"] / total20) if np.isfinite(rec20["sum_effect_delta"]) else np.nan
        gates = {
            "01_2020_support_n_ge8": rec20["n"] >= 8,
            "02_2020_rec_mae_worse_gt2pct": bool(np.isfinite(rec20["relative_mae_change"]) and rec20["relative_mae_change"] > .02),
            "03_2020_net_worsening_contribution_ge20pct": bool(np.isfinite(contribution) and contribution >= .20),
            "04_outside_2020_support_n_ge20": recout["n"] >= 20,
            "05_harm_replicates_ge2_seasons": len(harm_seasons) >= 2,
            "06_outside_2020_pooled_rec_mae_worse": bool(np.isfinite(recout["relative_mae_change"]) and recout["relative_mae_change"] > 0),
            "07_allocation_dominant_2020_or_outside": room20["classification"] == "ALLOCATION_DOMINANT" or roomout["classification"] == "ALLOCATION_DOMINANT",
            "08_integrity": integrity_ok,
        }
        qualifies = bool(all(gates.values()))
        material = bool(gates["01_2020_support_n_ge8"] and gates["02_2020_rec_mae_worse_gt2pct"] and gates["03_2020_net_worsening_contribution_ge20pct"])
        if qualifies:
            qualified.append(state)
        if material:
            material20.append(state)

        primary_rows.append({
            "state": state,
            "n_all": recall["n"],
            "baseline_rec_mae_all": recall["baseline_mae"],
            "candidate_rec_mae_all": recall["candidate_mae"],
            "relative_rec_mae_change_all": recall["relative_mae_change"],
            "baseline_rec_rmse_all": recall["baseline_rmse"],
            "candidate_rec_rmse_all": recall["candidate_rmse"],
            "baseline_rec_bias_all": recall["baseline_bias"],
            "candidate_rec_bias_all": recall["candidate_bias"],
            "baseline_rec_p90_all": recall["baseline_p90"],
            "candidate_rec_p90_all": recall["candidate_p90"],
            "baseline_target_mae_all": tgtall["baseline_mae"],
            "candidate_target_mae_all": tgtall["candidate_mae"],
            "rb1_rec_n": rb1all["n"],
            "rb1_baseline_rec_mae": rb1all["baseline_mae"],
            "rb1_candidate_rec_mae": rb1all["candidate_mae"],
            "rb2plus_rec_n": rb2all["n"],
            "rb2plus_baseline_rec_mae": rb2all["baseline_mae"],
            "rb2plus_candidate_rec_mae": rb2all["candidate_mae"],
            "n_2020": rec20["n"],
            "relative_rec_mae_change_2020": rec20["relative_mae_change"],
            "sum_rec_ae_delta_2020": rec20["sum_effect_delta"],
            "share_total_2020_net_worsening": contribution,
            "n_2021_2025": recout["n"],
            "relative_rec_mae_change_2021_2025": recout["relative_mae_change"],
            "harmful_seasons_2021_2025": ",".join(map(str, harm_seasons)),
            "harmful_season_count_2021_2025": len(harm_seasons),
            "harmful_n_2021_2025": harm_n,
            "room_diag_2020": room20["classification"],
            "room_total_delta_2020": room20["room_total_effect_delta_sum"],
            "summed_player_delta_2020": room20["summed_player_effect_delta_sum"],
            "room_diag_2021_2025": roomout["classification"],
            "room_total_delta_2021_2025": roomout["room_total_effect_delta_sum"],
            "summed_player_delta_2021_2025": roomout["summed_player_effect_delta_sum"],
            **{f"gate_{k}": v for k, v in gates.items()},
            "materially_harmful_2020": material,
            "replicated_harmful_state": qualifies,
        })

    secondary_rows = []
    for state in SECONDARY_STATES:
        for period, mask in (
            ("2020", inc.season.eq(2020)),
            ("2021_2025", inc.season.ge(2021)),
            ("ALL", pd.Series(True, index=inc.index)),
        ):
            g = inc.loc[mask & inc[state]].copy()
            rec = metrics(g, "receptions")
            tgt = metrics(g, "targets")
            secondary_rows.append({
                "state": state, "period": period,
                **{f"rec_{k}": v for k, v in rec.items()},
                **{f"tgt_{k}": v for k, v in tgt.items()},
            })

    if qualified:
        disposition = "REPLICATED_ALLOCATION_FAILURE_MECHANISM_IDENTIFIED"
    elif material20:
        disposition = "2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER"
    else:
        disposition = "NO_COHERENT_MECHANISM_IDENTIFIED"

    result = {
        "candidate": "RB_R26K_WEEK1_ALLOCATION_MECHANISM_ATLAS_V1",
        "scientific_label": "NO_REFIT_REPLICATED_MECHANISM_DIAGNOSTIC",
        "disposition": disposition,
        "integrity": integrity,
        "all_integrity_gates_pass": integrity_ok,
        "total_2020_week1_vacancy_incumbent_signed_rec_ae_worsening": total20,
        "primary_states": list(PRIMARY_STATES),
        "materially_harmful_2020_states": material20,
        "qualified_replicated_states": qualified,
        "qualified_replicated_state_count": len(qualified),
        "child_candidate_design_authorized": bool(qualified),
        "prospective_shadow_authorized": False,
        "production_promotion_authorized": False,
        "exclude_2020_authorized": False,
        "r9_refit": False,
        "predictions_regenerated": False,
        "production_parameters_changed": False,
        "receiving_yard_means_changed": False,
        "r22_changed": False,
        "sportsbook_inputs_added": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    inc.to_csv(a.out_dir / "r26k_player_state_effects.csv", index=False)
    pd.DataFrame(primary_rows).to_csv(a.out_dir / "r26k_primary_state_summary.csv", index=False)
    pd.DataFrame(by_season_rows).to_csv(a.out_dir / "r26k_primary_state_by_season.csv", index=False)
    if room_rows_all:
        pd.concat(room_rows_all, ignore_index=True, sort=False).to_csv(a.out_dir / "r26k_room_allocation_diagnostics.csv", index=False)
    else:
        pd.DataFrame().to_csv(a.out_dir / "r26k_room_allocation_diagnostics.csv", index=False)
    pd.DataFrame(ordering_rows).to_csv(a.out_dir / "r26k_ordering_diagnostics.csv", index=False)
    pd.DataFrame(secondary_rows).to_csv(a.out_dir / "r26k_secondary_atomic_diagnostics.csv", index=False)
    (a.out_dir / "r26k_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(pd.DataFrame(primary_rows).to_csv(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
