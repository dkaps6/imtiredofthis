#!/usr/bin/env python3
"""R26F no-refit forensic atlas for the 2020 Week-1 R26 vacancy failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = tuple(range(2020, 2026))
TOL = 1e-12


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_many(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if len(paths) < 6:
        raise RuntimeError(f"expected >=6 {name} files, found {len(paths)}")
    return pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True, sort=False)


def metrics(g: pd.DataFrame, market: str = "receptions") -> dict:
    a = num(g[f"actual_{market}"]).to_numpy(float)
    b = num(g[f"baseline_{market}"]).to_numpy(float)
    c = num(g[f"candidate_{market}"]).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a = a[ok]; b = b[ok]; c = c[ok]
    if len(a) == 0:
        return {"n": 0, "baseline_mae": np.nan, "candidate_mae": np.nan, "relative_mae_change": np.nan,
                "baseline_rmse": np.nan, "candidate_rmse": np.nan, "baseline_bias": np.nan, "candidate_bias": np.nan,
                "baseline_p90": np.nan, "candidate_p90": np.nan, "mean_effect_delta": np.nan, "sum_effect_delta": np.nan}
    be = b - a; ce = c - a
    bae = np.abs(be); cae = np.abs(ce)
    bmae = float(bae.mean()); cmae = float(cae.mean())
    return {
        "n": int(len(a)),
        "baseline_mae": bmae,
        "candidate_mae": cmae,
        "relative_mae_change": float(cmae / bmae - 1.0) if bmae > 1e-12 else np.nan,
        "baseline_rmse": float(np.sqrt(np.mean(be * be))),
        "candidate_rmse": float(np.sqrt(np.mean(ce * ce))),
        "baseline_bias": float(be.mean()),
        "candidate_bias": float(ce.mean()),
        "baseline_p90": float(np.quantile(bae, .90)),
        "candidate_p90": float(np.quantile(cae, .90)),
        "mean_effect_delta": float((cae - bae).mean()),
        "sum_effect_delta": float((cae - bae).sum()),
    }


def qlabel(s: pd.Series) -> tuple[pd.Series, list[float]]:
    out = pd.Series(index=s.index, dtype="object")
    finite = num(s).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return out, []
    q = pd.qcut(finite, q=4, duplicates="drop")
    codes = q.cat.codes
    out.loc[finite.index] = [f"Q{int(x)+1}" for x in codes]
    edges = [float(x) for x in q.cat.categories.left.tolist()] + [float(q.cat.categories[-1].right)]
    return out, edges


def direction(delta: pd.Series) -> pd.Series:
    d = num(delta)
    return pd.Series(np.select([d.gt(TOL), d.lt(-TOL)], ["INCREASE", "DECREASE"], default="UNCHANGED"), index=d.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    pred = read_many(a.r26_root, "r26_predictions.csv")
    pred["season"] = num(pred.season).astype(int)
    pred["week"] = num(pred.week).astype(int)
    pred = pred.loc[pred.season.isin(SEASONS)].copy()
    for c in ["vacancy_active", "continuing_same_team", "rb_rank", "room_exits_n", "room_entrants_n", "prior_depth_available",
              "actual_targets", "actual_receptions", "baseline_targets", "candidate_targets", "baseline_receptions", "candidate_receptions",
              "r9_calibrated_residual"]:
        pred[c] = num(pred[c])

    w1_vac = pred.loc[pred.week.eq(1) & pred.vacancy_active.eq(1)].copy()
    inc = w1_vac.loc[w1_vac.continuing_same_team.eq(1)].copy()
    if inc.empty:
        raise RuntimeError("R26F found zero Week-1 vacancy incumbents")

    # Prediction/source-only dimensions frozen in the plan/implementation lock.
    inc["role_state"] = np.where(inc.rb_rank.eq(1), "RB1", "RB2+")
    inc["exit_state"] = np.where(inc.room_exits_n.eq(1), "ONE_EXIT", "TWO_PLUS_EXITS")
    inc["entrant_state"] = np.where(inc.room_entrants_n.eq(0), "NO_ENTRANT", "ONE_PLUS_ENTRANT")
    inc["balance_state"] = np.select(
        [inc.room_exits_n.gt(inc.room_entrants_n), inc.room_exits_n.eq(inc.room_entrants_n)],
        ["EXITS_GT_ENTRANTS", "EXITS_EQ_ENTRANTS"], default="EXITS_LT_ENTRANTS")
    inc["depth_state"] = np.where(inc.prior_depth_available.eq(1), "AVAILABLE", "UNAVAILABLE")

    inc["target_move"] = inc.candidate_targets - inc.baseline_targets
    inc["reception_move"] = inc.candidate_receptions - inc.baseline_receptions
    inc["abs_target_move"] = inc.target_move.abs()
    inc["abs_reception_move"] = inc.reception_move.abs()
    inc["abs_r9_residual"] = inc.r9_calibrated_residual.abs()
    inc["target_direction"] = direction(inc.target_move)
    inc["reception_direction"] = direction(inc.reception_move)
    inc["r9_sign"] = direction(inc.r9_calibrated_residual).replace({"INCREASE": "POSITIVE", "DECREASE": "NEGATIVE", "UNCHANGED": "ZERO"})
    inc["target_move_quartile"], target_edges = qlabel(inc.abs_target_move)
    inc["reception_move_quartile"], reception_edges = qlabel(inc.abs_reception_move)
    inc["r9_magnitude_quartile"], r9_edges = qlabel(inc.abs_r9_residual)

    # Team target leader flip is source/prediction-only.
    leader_rows = []
    for keys, g in w1_vac.groupby(["season", "week", "team"], sort=True):
        gb = g.dropna(subset=["baseline_targets"]).sort_values(["baseline_targets", "player_clean_key"], ascending=[False, True])
        gc = g.dropna(subset=["candidate_targets"]).sort_values(["candidate_targets", "player_clean_key"], ascending=[False, True])
        bl = None if gb.empty else str(gb.iloc[0].player_clean_key)
        cl = None if gc.empty else str(gc.iloc[0].player_clean_key)
        leader_rows.append({"season": keys[0], "week": keys[1], "team": keys[2], "leader_state": "FLIP" if bl != cl else "NO_FLIP"})
    inc = inc.merge(pd.DataFrame(leader_rows), on=["season", "week", "team"], how="left", validate="many_to_one")

    for market in ("receptions", "targets"):
        inc[f"baseline_{market}_ae"] = (inc[f"baseline_{market}"] - inc[f"actual_{market}"]).abs()
        inc[f"candidate_{market}_ae"] = (inc[f"candidate_{market}"] - inc[f"actual_{market}"]).abs()
        inc[f"{market}_effect_delta"] = inc[f"candidate_{market}_ae"] - inc[f"baseline_{market}_ae"]

    dimension_map = {
        "role": "role_state",
        "exits": "exit_state",
        "entrants": "entrant_state",
        "exit_entrant_balance": "balance_state",
        "prior_depth": "depth_state",
        "target_move_quartile": "target_move_quartile",
        "reception_move_quartile": "reception_move_quartile",
        "target_direction": "target_direction",
        "reception_direction": "reception_direction",
        "r9_residual_sign": "r9_sign",
        "r9_magnitude_quartile": "r9_magnitude_quartile",
        "room_target_leader_flip": "leader_state",
    }

    state_rows: list[dict] = []
    for dim, col in dimension_map.items():
        for level in sorted(inc[col].dropna().astype(str).unique()):
            for season in SEASONS:
                g = inc.loc[inc.season.eq(season) & inc[col].astype(str).eq(level)].copy()
                rec = metrics(g, "receptions")
                tgt = metrics(g, "targets")
                state_rows.append({"dimension": dim, "level": level, "season": season, **{f"rec_{k}": v for k, v in rec.items()}, **{f"tgt_{k}": v for k, v in tgt.items()}})
    states = pd.DataFrame(state_rows)

    # Room-total vs within-room allocation diagnosis, using the same labeled rows for all sums.
    room_rows = []
    for (season, team), g in w1_vac.groupby(["season", "team"], sort=True):
        for market in ("receptions", "targets"):
            x = g.loc[g[f"actual_{market}"].notna() & g[f"baseline_{market}"].notna() & g[f"candidate_{market}"].notna()].copy()
            if x.empty:
                continue
            actual = float(x[f"actual_{market}"].sum())
            base = float(x[f"baseline_{market}"].sum())
            cand = float(x[f"candidate_{market}"].sum())
            room_rows.append({
                "season": int(season), "team": str(team), "market": market, "n_players": int(len(x)),
                "baseline_room_total_ae": abs(base - actual), "candidate_room_total_ae": abs(cand - actual),
                "room_total_effect_delta": abs(cand - actual) - abs(base - actual),
                "baseline_summed_player_ae": float((x[f"baseline_{market}"] - x[f"actual_{market}"]).abs().sum()),
                "candidate_summed_player_ae": float((x[f"candidate_{market}"] - x[f"actual_{market}"]).abs().sum()),
                "summed_player_effect_delta": float(((x[f"candidate_{market}"] - x[f"actual_{market}"]).abs() - (x[f"baseline_{market}"] - x[f"actual_{market}"]).abs()).sum()),
            })
    rooms = pd.DataFrame(room_rows)
    r20 = rooms.loc[rooms.season.eq(2020) & rooms.market.eq("receptions")]
    room_delta = float(r20.room_total_effect_delta.sum())
    player_delta = float(r20.summed_player_effect_delta.sum())
    if player_delta > 0 and (room_delta <= 0 or room_delta <= 0.25 * player_delta):
        room_diagnosis = "WITHIN_ROOM_ALLOCATION_DOMINANT"
    elif player_delta > 0 and room_delta > 0.25 * player_delta:
        room_diagnosis = "ROOM_TOTAL_AND_ALLOCATION_BOTH_HARMFUL"
    else:
        room_diagnosis = "ROOM_DIAGNOSIS_MIXED"

    labeled20 = inc.loc[inc.season.eq(2020) & inc.receptions_effect_delta.notna()].copy()
    total_2020_signed_delta = float(labeled20.receptions_effect_delta.sum())
    if total_2020_signed_delta <= 0:
        raise RuntimeError(f"R26F expected positive known 2020 net worsening; got {total_2020_signed_delta}")

    forensic_rows = []
    replicated_states = []
    material_states = []
    for (dim, level), gg in states.groupby(["dimension", "level"], sort=True):
        r20s = gg.loc[gg.season.eq(2020)].iloc[0]
        n20 = int(r20s.rec_n)
        contribution = float(r20s.rec_sum_effect_delta / total_2020_signed_delta) if np.isfinite(r20s.rec_sum_effect_delta) else np.nan
        material = bool(
            n20 >= 10
            and float(r20s.rec_mean_effect_delta) > 0
            and float(r20s.rec_relative_mae_change) > 0.02
            and contribution >= 0.20
        )
        support = gg.loc[(gg.season.ge(2021)) & (gg.rec_mean_effect_delta.gt(0)) & (gg.rec_n.gt(0))].copy()
        supporting_seasons = sorted(int(x) for x in support.season.unique())
        supporting_n = int(support.rec_n.sum()) if not support.empty else 0
        replicated = bool(material and len(supporting_seasons) >= 2 and supporting_n >= 15)
        row = {
            "dimension": dim, "level": level,
            "n_2020": n20,
            "relative_mae_change_2020": float(r20s.rec_relative_mae_change) if np.isfinite(r20s.rec_relative_mae_change) else np.nan,
            "mean_effect_delta_2020": float(r20s.rec_mean_effect_delta) if np.isfinite(r20s.rec_mean_effect_delta) else np.nan,
            "sum_effect_delta_2020": float(r20s.rec_sum_effect_delta) if np.isfinite(r20s.rec_sum_effect_delta) else np.nan,
            "share_of_total_2020_net_worsening": contribution,
            "materially_harmful_2020": material,
            "supporting_harmful_seasons_2021_2025": ",".join(map(str, supporting_seasons)),
            "supporting_harmful_season_count": len(supporting_seasons),
            "supporting_harmful_n": supporting_n,
            "replicated_harmful_state": replicated,
        }
        forensic_rows.append(row)
        if material:
            material_states.append(row)
        if replicated:
            replicated_states.append(row)

    if replicated_states:
        disposition = "WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED"
    elif material_states:
        disposition = "WEEK1_2020_FAILURE_LOCALIZED_NOT_REPLICATED"
    else:
        disposition = "WEEK1_FAILURE_MECHANISM_UNRESOLVED"

    result = {
        "candidate": "RB_R26F_2020_WEEK1_VACANCY_FAILURE_FORENSIC_ATLAS_V1",
        "scientific_label": "NO_REFIT_FORENSIC_DIAGNOSTIC",
        "disposition": disposition,
        "child_candidate_design_authorized": bool(replicated_states),
        "prospective_shadow_authorized": False,
        "production_promotion_authorized": False,
        "r26e_disposition_unchanged": "WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW",
        "integrity": {
            "r26_predictions_regenerated": False,
            "r9_refit": False,
            "sportsbook_inputs_added": 0,
            "production_parameters_changed": False,
            "same_week_depth_added": False,
            "required_seasons_present": sorted(int(x) for x in inc.season.unique()) == list(SEASONS),
        },
        "source_only_quartile_edges": {"target_move": target_edges, "reception_move": reception_edges, "r9_magnitude": r9_edges},
        "room_diagnosis": {
            "classification": room_diagnosis,
            "2020_room_total_reception_ae_delta_sum": room_delta,
            "2020_summed_player_reception_ae_delta_sum": player_delta,
        },
        "total_2020_signed_reception_ae_worsening": total_2020_signed_delta,
        "materially_harmful_state_count": len(material_states),
        "replicated_harmful_state_count": len(replicated_states),
        "replicated_harmful_states": replicated_states,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    inc.to_csv(a.out_dir / "r26f_week1_vacancy_incumbent_effects.csv", index=False)
    states.to_csv(a.out_dir / "r26f_state_season_metrics.csv", index=False)
    rooms.to_csv(a.out_dir / "r26f_room_total_vs_allocation.csv", index=False)
    pd.DataFrame(forensic_rows).to_csv(a.out_dir / "r26f_harmful_state_replication.csv", index=False)
    (a.out_dir / "r26f_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== harmful-state replication ===")
    print(pd.DataFrame(forensic_rows).loc[lambda x: x.materially_harmful_2020].to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
