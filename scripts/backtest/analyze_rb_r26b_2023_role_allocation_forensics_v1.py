#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = list(range(2020, 2026))
WINNING_SEASONS = [2020, 2021, 2022, 2024, 2025]
PARENT_RUN = 34356222339
PARENT_ARTIFACT = 10106271075
PARENT_DIGEST = "sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e"


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing required parent artifact file: {path}")
    return pd.read_csv(path, low_memory=False)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metric(actual: pd.Series, pred: pd.Series) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    z = z[np.isfinite(z.actual) & np.isfinite(z.pred)]
    if z.empty:
        return {k: np.nan for k in ["mae","rmse","bias","median_abs_error","p75_abs_error","p90_abs_error","pearson","spearman"]} | {"n": 0}
    e = z.pred - z.actual
    ae = e.abs()
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e.to_numpy(float))))),
        "bias": float(e.mean()),
        "median_abs_error": float(ae.median()),
        "p75_abs_error": float(ae.quantile(0.75)),
        "p90_abs_error": float(ae.quantile(0.90)),
        "pearson": float(z.pred.corr(z.actual, method="pearson")) if len(z) > 1 else np.nan,
        "spearman": float(z.pred.corr(z.actual, method="spearman")) if len(z) > 1 else np.nan,
    }


def add_room_ordering(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["season", "week", "event_id", "team"]
    x = x.copy()
    x["baseline_rank_calc"] = x.groupby(keys)["baseline_room_share"].rank(method="first", ascending=False).astype(int)
    x["candidate_rank_calc"] = x.groupby(keys)["candidate_room_share"].rank(method="first", ascending=False).astype(int)
    x["rank_move"] = np.select(
        [x.candidate_rank_calc < x.baseline_rank_calc, x.candidate_rank_calc > x.baseline_rank_calc],
        ["moved_up", "moved_down"], default="unchanged"
    )
    x["share_delta"] = num(x.candidate_room_share) - num(x.baseline_room_share)

    rows = []
    for k, g in x.groupby(keys, sort=False):
        bi = g["baseline_room_share"].astype(float).idxmax()
        ci = g["candidate_room_share"].astype(float).idxmax()
        rows.append({
            **dict(zip(keys, k)),
            "baseline_top_key": str(g.loc[bi, "player_clean_key"]),
            "candidate_top_key": str(g.loc[ci, "player_clean_key"]),
            "baseline_top_player": str(g.loc[bi, "player"]),
            "candidate_top_player": str(g.loc[ci, "player"]),
            "baseline_top_share": float(g.loc[bi, "baseline_room_share"]),
            "candidate_top_share": float(g.loc[ci, "candidate_room_share"]),
            "baseline_hhi": float(np.square(num(g.baseline_room_share).fillna(0.0)).sum()),
            "candidate_hhi": float(np.square(num(g.candidate_room_share).fillna(0.0)).sum()),
        })
    room = pd.DataFrame(rows)
    room["top_agreement"] = np.where(room.baseline_top_key.eq(room.candidate_top_key), "same_top", "top_changed")
    room["hhi_delta"] = room.candidate_hhi - room.baseline_hhi
    room["hhi_change"] = np.select(
        [room.hhi_delta > 0.02, room.hhi_delta < -0.02],
        ["more_concentrated", "less_concentrated"], default="approx_unchanged"
    )
    x = x.merge(room[keys + ["top_agreement", "hhi_delta", "hhi_change"]], on=keys, how="left", validate="many_to_one")
    return x, room


def add_categories(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["baseline_share_band"] = pd.cut(
        num(x.baseline_room_share), [-np.inf, .25, .50, .75, np.inf], right=False,
        labels=["<0.25", "0.25-0.50", "0.50-0.75", ">=0.75"]
    )
    x["residual_band"] = pd.cut(
        num(x.r8_raw_residual), [-np.inf, -.25, .25, np.inf], right=True, include_lowest=True,
        labels=["<-0.25", "-0.25_to_0.25", ">0.25"]
    )
    x["share_delta_band"] = pd.cut(
        num(x.share_delta), [-np.inf, -.10, -.025, .025, .10, np.inf], right=False,
        labels=["<-0.10", "-0.10_to_-0.025", "-0.025_to_0.025", "0.025_to_0.10", ">0.10"]
    )
    x["room_size_band"] = num(x.current_rb_room_n).map(lambda v: "1" if v == 1 else "2" if v == 2 else "3" if v == 3 else "4+" if np.isfinite(v) else "missing")
    x["exits_band"] = np.where(num(x.room_exits_n).fillna(0).ge(2), "2+", "1")
    x["entrants_band"] = np.where(num(x.room_entrants_n).fillna(0).ge(1), "1+", "0")
    delta = num(x.current_rb_room_n) - num(x.prior_rb_room_n)
    x["room_size_delta"] = np.select([delta < 0, delta > 0], ["shrank", "expanded"], default="unchanged")
    x["continuity"] = np.select(
        [num(x.continuing_same_team).fillna(0).eq(1), num(x.new_to_team_veteran).fillna(0).eq(1), num(x.no_prior_nfl_roster).fillna(0).eq(1)],
        ["same_team_incumbent", "new_to_team_veteran", "no_prior_nfl"], default="other"
    )
    x["week1_bucket"] = np.where(num(x.week).eq(1), "W1", "W2+")
    x["phase"] = pd.cut(num(x.week), [0,4,9,13,18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    depth_avail = num(x.prior_depth_available).fillna(0).eq(1)
    depth_ord = num(x.prior_depth_team)
    x["prior_role_compat"] = np.where(
        ~depth_avail, "no_depth",
        np.where(depth_ord.eq(num(x.baseline_rank_calc)), "exact_rank_match", "rank_mismatch")
    )
    return x


def add_errors(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for v in ("baseline", "candidate"):
        x[f"{v}_rec_error"] = num(x[f"{v}_receptions"]) - num(x.actual_receptions)
        x[f"{v}_rec_abs_error"] = x[f"{v}_rec_error"].abs()
        x[f"{v}_target_error"] = num(x[f"{v}_targets"]) - num(x.actual_targets)
        x[f"{v}_target_abs_error"] = x[f"{v}_target_error"].abs()
    x["candidate_minus_baseline_rec_abs_error"] = x.candidate_rec_abs_error - x.baseline_rec_abs_error
    x["candidate_minus_baseline_target_abs_error"] = x.candidate_target_abs_error - x.baseline_target_abs_error
    return x


def slice_metrics(g: pd.DataFrame) -> dict:
    br = metric(g.actual_receptions, g.baseline_receptions)
    cr = metric(g.actual_receptions, g.candidate_receptions)
    bt = metric(g.actual_targets, g.baseline_targets)
    ct = metric(g.actual_targets, g.candidate_targets)
    return {
        "n": br["n"],
        **{f"baseline_rec_{k}": v for k, v in br.items() if k != "n"},
        **{f"candidate_rec_{k}": v for k, v in cr.items() if k != "n"},
        "delta_rec_mae": cr["mae"] - br["mae"],
        "delta_rec_abs_bias": abs(cr["bias"]) - abs(br["bias"]),
        "delta_rec_p90": cr["p90_abs_error"] - br["p90_abs_error"],
        "baseline_target_mae": bt["mae"],
        "candidate_target_mae": ct["mae"],
        "delta_target_mae": ct["mae"] - bt["mae"],
        "baseline_target_rmse": bt["rmse"],
        "candidate_target_rmse": ct["rmse"],
        "baseline_target_bias": bt["bias"],
        "candidate_target_bias": ct["bias"],
    }


def build_slices(primary: pd.DataFrame, full_vacancy: pd.DataFrame) -> pd.DataFrame:
    dims = [
        "role", "baseline_share_band", "room_size_band", "exits_band", "entrants_band",
        "room_size_delta", "continuity", "residual_band", "rank_move", "top_agreement",
        "share_delta_band", "hhi_change", "prior_role_compat", "week1_bucket", "phase",
    ]
    rows = []
    for population_name, pop in (("VACANCY_INCUMBENT", primary), ("ALL_VACANCY_ACTIVE", full_vacancy)):
        for dimension in dims:
            for value, g in pop.groupby(dimension, observed=False, dropna=False):
                if g.empty:
                    continue
                val = "NA" if pd.isna(value) else str(value)
                for season, sg in g.groupby("season", sort=True):
                    rows.append({
                        "population": population_name,
                        "dimension": dimension,
                        "value": val,
                        "season_bucket": str(int(season)),
                        "low_n": int(metric(sg.actual_receptions, sg.baseline_receptions)["n"] < 20),
                        **slice_metrics(sg),
                    })
                win = g.loc[g.season.isin(WINNING_SEASONS)]
                if not win.empty:
                    rows.append({
                        "population": population_name,
                        "dimension": dimension,
                        "value": val,
                        "season_bucket": "WINNING_POOL",
                        "low_n": int(metric(win.actual_receptions, win.baseline_receptions)["n"] < 20),
                        **slice_metrics(win),
                    })
    return pd.DataFrame(rows)


def build_room_attribution(x: pd.DataFrame, room_order: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "event_id", "team"]
    vac = x.loc[num(x.vacancy_active).fillna(0).eq(1)].copy()
    rows = []
    for k, g in vac.groupby(keys, sort=False):
        m = room_order.copy()
        for c, v in zip(keys, k):
            m = m.loc[m[c].astype(str).eq(str(v))]
        if len(m) != 1:
            raise RuntimeError(f"room ordering lookup failed for {k}: {len(m)}")
        m = m.iloc[0]
        actual = float(num(g.actual_receptions).fillna(0.0).sum())
        base = float(num(g.baseline_receptions).fillna(0.0).sum())
        cand = float(num(g.candidate_receptions).fillna(0.0).sum())
        rows.append({
            **dict(zip(keys, k)),
            "current_rb_room_n": float(num(g.current_rb_room_n).iloc[0]),
            "prior_rb_room_n": float(num(g.prior_rb_room_n).iloc[0]),
            "exits_n": int(num(g.room_exits_n).iloc[0]),
            "entrants_n": int(num(g.room_entrants_n).iloc[0]),
            "baseline_top_player": m.baseline_top_player,
            "candidate_top_player": m.candidate_top_player,
            "baseline_top_share": float(m.baseline_top_share),
            "candidate_top_share": float(m.candidate_top_share),
            "top_changed": int(m.baseline_top_key != m.candidate_top_key),
            "baseline_hhi": float(m.baseline_hhi),
            "candidate_hhi": float(m.candidate_hhi),
            "hhi_delta": float(m.hhi_delta),
            "labeled_player_count": int(num(g.actual_receptions).notna().sum()),
            "room_player_count": int(len(g)),
            "sum_player_baseline_abs_rec_error": float(num(g.baseline_rec_abs_error).sum()),
            "sum_player_candidate_abs_rec_error": float(num(g.candidate_rec_abs_error).sum()),
            "player_allocation_error_delta": float(num(g.candidate_rec_abs_error).sum() - num(g.baseline_rec_abs_error).sum()),
            "actual_team_rb_receptions_zero_for_unlabeled": actual,
            "baseline_team_rb_receptions": base,
            "candidate_team_rb_receptions": cand,
            "baseline_team_rb_rec_abs_error": abs(base - actual),
            "candidate_team_rb_rec_abs_error": abs(cand - actual),
            "team_rb_rec_error_delta": abs(cand - actual) - abs(base - actual),
        })
    return pd.DataFrame(rows)


def season_comparison(primary: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in primary.groupby("season", sort=True):
        rg = rooms.loc[rooms.season.eq(season)]
        d = slice_metrics(g)
        rows.append({
            "season": int(season),
            "raw_vacancy_incumbent_rows": int(len(g)),
            "labeled_vacancy_incumbent_n": int(d["n"]),
            "delta_rec_mae": d["delta_rec_mae"],
            "baseline_rec_mae": d["baseline_rec_mae"],
            "candidate_rec_mae": d["candidate_rec_mae"],
            "delta_target_mae": d["delta_target_mae"],
            "week1_labeled_n": int(metric(g.loc[g.week.eq(1)].actual_receptions, g.loc[g.week.eq(1)].baseline_receptions)["n"]),
            "week1_delta_rec_mae": slice_metrics(g.loc[g.week.eq(1)])["delta_rec_mae"] if g.week.eq(1).any() else np.nan,
            "weeks2plus_delta_rec_mae": slice_metrics(g.loc[g.week.ne(1)])["delta_rec_mae"] if g.week.ne(1).any() else np.nan,
            "top_changed_rate": float(g.top_agreement.eq("top_changed").mean()),
            "more_concentrated_rate": float(g.hhi_change.eq("more_concentrated").mean()),
            "mean_abs_share_delta": float(num(g.share_delta).abs().mean()),
            "mean_exits": float(num(g.room_exits_n).mean()),
            "mean_entrants": float(num(g.room_entrants_n).mean()),
            "mean_current_room_n": float(num(g.current_rb_room_n).mean()),
            "mean_prior_room_n": float(num(g.prior_rb_room_n).mean()),
            "prior_depth_coverage": float(num(g.prior_depth_available).fillna(0).eq(1).mean()),
            "mean_room_player_allocation_error_delta": float(num(rg.player_allocation_error_delta).mean()) if not rg.empty else np.nan,
            "mean_room_total_rec_error_delta": float(num(rg.team_rb_rec_error_delta).mean()) if not rg.empty else np.nan,
        })
    return pd.DataFrame(rows)


def summarize(slice_df: pd.DataFrame, season_df: pd.DataFrame, x: pd.DataFrame) -> dict:
    p = slice_df.loc[(slice_df.population.eq("VACANCY_INCUMBENT")) & slice_df.low_n.eq(0)].copy()
    r23 = p.loc[p.season_bucket.eq("2023") & p.delta_rec_mae.gt(0)].copy()
    replicated = []
    for _, row in r23.iterrows():
        same = p.loc[
            p.dimension.eq(row.dimension) & p.value.eq(row.value) &
            ~p.season_bucket.isin(["2023", "WINNING_POOL"])
        ].copy()
        harm = same.loc[same.delta_rec_mae.gt(0)]
        if not harm.empty:
            replicated.append({
                "dimension": row.dimension,
                "value": row.value,
                "n_2023": int(row.n),
                "delta_rec_mae_2023": float(row.delta_rec_mae),
                "other_harm_seasons": [int(s) for s in harm.season_bucket.astype(int).tolist()],
                "other_harm_deltas": [float(v) for v in harm.delta_rec_mae.tolist()],
            })
    replicated = sorted(replicated, key=lambda d: (-len(d["other_harm_seasons"]), -d["delta_rec_mae_2023"]))

    s23 = season_df.loc[season_df.season.eq(2023)].iloc[0]
    wins = season_df.loc[season_df.season.isin(WINNING_SEASONS)]
    q = {
        "parent_run": PARENT_RUN,
        "parent_artifact": PARENT_ARTIFACT,
        "parent_digest": PARENT_DIGEST,
        "prior_rb_room_share_dimension": "NOT_OBSERVABLE_FROM_IMMUTABLE_PARENT_ARTIFACT",
        "rows_all": int(len(x)),
        "raw_rows_vacancy_incumbent": int(num(x.vacancy_incumbent).fillna(0).eq(1).sum()),
        "labeled_rows_vacancy_incumbent": int(metric(
            x.loc[num(x.vacancy_incumbent).fillna(0).eq(1), "actual_receptions"],
            x.loc[num(x.vacancy_incumbent).fillna(0).eq(1), "baseline_receptions"]
        )["n"]),
        "2023": {
            "delta_rec_mae": float(s23.delta_rec_mae),
            "week1_delta_rec_mae": float(s23.week1_delta_rec_mae),
            "weeks2plus_delta_rec_mae": float(s23.weeks2plus_delta_rec_mae),
            "room_player_allocation_error_delta": float(s23.mean_room_player_allocation_error_delta),
            "room_total_rec_error_delta": float(s23.mean_room_total_rec_error_delta),
            "top_changed_rate": float(s23.top_changed_rate),
            "more_concentrated_rate": float(s23.more_concentrated_rate),
        },
        "winning_season_means": {
            "room_player_allocation_error_delta": float(wins.mean_room_player_allocation_error_delta.mean()),
            "room_total_rec_error_delta": float(wins.mean_room_total_rec_error_delta.mean()),
            "top_changed_rate": float(wins.top_changed_rate.mean()),
            "more_concentrated_rate": float(wins.more_concentrated_rate.mean()),
        },
        "replicated_harmful_predeclared_slices": replicated,
    }
    allocation_dominant = abs(float(s23.mean_room_player_allocation_error_delta)) > 3.0 * max(abs(float(s23.mean_room_total_rec_error_delta)), 1e-12)
    q["allocation_dominant_2023"] = bool(allocation_dominant)
    q["replicated_harmful_slice_count"] = int(len(replicated))
    q["disposition"] = "FORENSIC_MECHANISM_IDENTIFIED" if allocation_dominant and len(replicated) > 0 else "FORENSIC_2023_IDIOSYNCRATIC_NO_ROUTER"
    q["frozen_question_answers"] = {
        "2023_top_rank_disagreement_unusually_high": bool(s23.top_changed_rate > wins.top_changed_rate.mean()),
        "2023_candidate_concentration_unusually_high": bool(s23.more_concentrated_rate > wins.more_concentrated_rate.mean()),
        "2023_failure_is_week1": bool(float(s23.week1_delta_rec_mae) > 0),
        "2023_failure_is_weeks2plus": bool(float(s23.weeks2plus_delta_rec_mae) > 0),
        "2023_loss_is_within_room_allocation_dominant": bool(allocation_dominant),
    }
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    preds = []
    for season in SEASONS:
        path = a.artifact_root / "data" / "backtests" / f"r26_vacancy_gated_r9_{season}" / "r26_predictions.csv"
        sx = read(path)
        if int(num(sx.season).dropna().iloc[0]) != season:
            raise RuntimeError(f"season mismatch in {path}")
        preds.append(sx)
    pred = pd.concat(preds, ignore_index=True, sort=False)

    state_path = a.artifact_root / "data" / "backtests" / "rb_r26_safe_transition_state" / "r26_safe_transition_player_state.csv"
    state = read(state_path).rename(columns={"player_key": "player_clean_key"})
    join_cols = [
        "season", "week", "team", "player_clean_key", "current_rb_room_n", "prior_rb_room_n",
        "prior_depth_position", "prior_depth_team", "prior_depth_team_club", "prior_depth_available",
    ]
    missing = [c for c in join_cols if c not in state.columns]
    if missing:
        raise RuntimeError(f"safe transition state missing required columns: {missing}")
    x = pred.merge(state[join_cols], on=["season","week","team","player_clean_key"], how="left", suffixes=("", "_safe"), validate="many_to_one")
    state_missing_rate = float(num(x.current_rb_room_n).isna().mean())
    if state_missing_rate > 0.005:
        raise RuntimeError(f"transition-state join missing rate too high: {state_missing_rate}")

    if int(num(x.sportsbook_inputs_used).fillna(0).sum()) != 0:
        raise RuntimeError("parent predictions show sportsbook inputs")
    if int(num(x.future_outcomes_used).fillna(0).sum()) != 0:
        raise RuntimeError("parent predictions show future outcomes")

    x, room_order = add_room_ordering(x)
    x = add_categories(x)
    x = add_errors(x)
    full_vacancy = x.loc[num(x.vacancy_active).fillna(0).eq(1)].copy()
    primary = x.loc[num(x.vacancy_incumbent).fillna(0).eq(1)].copy()
    if primary.empty:
        raise RuntimeError("zero vacancy-incumbent rows")

    slices = build_slices(primary, full_vacancy)
    rooms = build_room_attribution(x, room_order)
    seasons = season_comparison(primary, rooms)
    summary = summarize(slices, seasons, x)
    integrity = {
        "parent_run": PARENT_RUN,
        "parent_artifact": PARENT_ARTIFACT,
        "parent_digest": PARENT_DIGEST,
        "sportsbook_inputs": int(num(x.sportsbook_inputs_used).fillna(0).sum()),
        "future_outcomes_in_features": int(num(x.future_outcomes_used).fillna(0).sum()),
        "transition_state_missing_rate": state_missing_rate,
        "same_week_depth_used": False,
        "new_external_source_used": False,
        "model_fit_performed": False,
        "candidate_rerun_performed": False,
        "production_parameters_changed": False,
        "prior_rb_room_share_observable": False,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "r26b_player_forensic_atlas.csv", index=False)
    slices.to_csv(a.out_dir / "r26b_slice_metrics.csv", index=False)
    rooms.to_csv(a.out_dir / "r26b_team_room_attribution.csv", index=False)
    seasons.to_csv(a.out_dir / "r26b_season_comparison.csv", index=False)
    (a.out_dir / "r26b_2023_vs_winning_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (a.out_dir / "r26b_integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(integrity, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
