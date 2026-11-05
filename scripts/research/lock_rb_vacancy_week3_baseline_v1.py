#!/usr/bin/env python3
"""Freeze RB Vacancy Opportunity V1 Week-3 baseline/candidate football means.

Research-only prospective lock. This script is intended to run from the exact
production-source checkout that created the source Full Slate artifact.

It:
- reconstructs the sportsbook-independent canonical full-roster simulation,
- freezes the production generic MC+ML+State Week-3 rushing means,
- applies ONLY the already-frozen V1 transfer to rules_rush_share,
- re-runs the same deterministic production simulation,
- leaves YPC/efficiency, ML, State, ensemble weights, and every other football
  input unchanged,
- attaches no target-game outcomes and reads no sportsbook data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.run_pricing_with_full_roster_universe_v1 import _canonical_game
from scripts.simulation_v2 import lookup, simulate

DATA = Path("data")
FORBIDDEN = {
    "actual", "actual_rushes", "actual_rush_yards", "target_game_snaps",
    "line", "source_line", "over_odds", "under_odds", "odds", "book",
    "book_title", "sportsbook", "bookmaker", "market_prob", "edge_pct",
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _key(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _pos_family(s: pd.Series) -> pd.Series:
    out = s.astype("string").fillna("").str.upper().str.strip()
    return out.replace({"HB": "RB", "TB": "RB"})


def _forbidden_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in FORBIDDEN or lc.startswith("sportsbook_"):
            out.append(str(c))
    return sorted(out)


def _diag_value(df: pd.DataFrame, team: str, player: str, col: str) -> float:
    q = df.loc[
        df["team"].astype(str).str.upper().eq(team)
        & _key(df["player_clean_key"]).eq(player)
    ]
    if len(q) != 1:
        raise RuntimeError(f"expected one diagnostic row {team}/{player} in {col}, got {len(q)}")
    return float(pd.to_numeric(q.iloc[0].get(col), errors="coerce"))


def _trace_frame(rows: list[dict], prefix: str) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    if t.empty:
        raise RuntimeError(f"{prefix} allocation trace empty")
    keep = [
        "event_id", "team", "player_clean_key", "raw_player_rush_share",
        "raw_team_rush_share_sum", "final_player_probability",
        "residual_probability", "team_rush_total_mean",
        "expected_carries_from_final_probability",
        "realized_multinomial_mean_carries",
    ]
    missing = set(keep) - set(t.columns)
    if missing:
        raise RuntimeError(f"{prefix} trace missing {sorted(missing)}")
    t = t[keep].drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
    return t.rename(columns={c: f"{prefix}_{c}" for c in keep if c not in {"event_id", "team", "player_clean_key"}})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vacancy", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--source-run", type=int, required=True)
    p.add_argument("--source-artifact", type=int, required=True)
    p.add_argument("--source-digest", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--lock-run", type=int, required=True)
    p.add_argument("--lock-artifact", type=int, required=True)
    p.add_argument("--lock-digest", required=True)
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vacancy = _read(Path(a.vacancy), "frozen vacancy state")
    if _forbidden_columns(vacancy):
        raise RuntimeError(f"vacancy artifact contains forbidden fields: {_forbidden_columns(vacancy)}")
    if not pd.to_numeric(vacancy["target_season"], errors="coerce").eq(2026).all():
        raise RuntimeError("vacancy artifact target season drift")
    if not pd.to_numeric(vacancy["target_week"], errors="coerce").eq(3).all():
        raise RuntimeError("vacancy artifact target week drift")
    if vacancy["team"].nunique() != 2 or len(vacancy) != 4:
        raise RuntimeError(f"unexpected frozen vacancy cohort rows={len(vacancy)} teams={vacancy['team'].nunique()}")

    consensus = _read(DATA / "player_form_consensus.csv", "pregame player consensus")
    ml = _read(DATA / "model_ml_diagnostics.csv", "pregame ML diagnostics")
    state = _read(DATA / "model_state_diagnostics.csv", "pregame State diagnostics")
    weights = _read(DATA / "model_ensemble_weights.csv", "promoted ensemble weights")
    availability = _read(DATA / "current_player_availability.csv", "pregame availability")

    for label, df in [("consensus", consensus), ("ml", ml), ("state", state), ("weights", weights)]:
        bad = _forbidden_columns(df)
        if bad:
            raise RuntimeError(f"{label} contains forbidden target/market fields: {bad}")

    consensus["team"] = consensus["team"].astype(str).str.upper().str.strip()
    consensus["player_clean_key"] = _key(consensus["player_clean_key"])
    consensus["position_family"] = _pos_family(consensus["position"])

    event_teams = sorted(vacancy["team"].astype(str).str.upper().unique())
    unavailable_keys = set(
        availability.loc[
            availability["team"].astype(str).str.upper().isin(event_teams)
            & pd.to_numeric(availability["definitive_unavailable"], errors="coerce").fillna(0).eq(1)
            & availability["position_group"].astype(str).str.upper().isin(["RB", "FB"]),
            "player_clean_key",
        ].astype(str).str.strip()
    )
    active_keys = set(consensus.loc[consensus["team"].isin(event_teams), "player_clean_key"])
    if unavailable_keys & active_keys:
        raise RuntimeError(f"definitive unavailable RB leaked into active consensus: {sorted(unavailable_keys & active_keys)}")

    # Exact current production full-roster football-universe preparation.
    prepared = consensus.copy()
    prepared["event_id"] = [
        _canonical_game(t, o, s, w)
        for t, o, s, w in zip(prepared["team"], prepared["opponent"], prepared["season"], prepared["week"])
    ]
    prepared["market"] = "football_universe"
    prepared = apply_bayesian_to_metrics(prepared)
    prepared = apply_rules_to_metrics(prepared)
    if not pd.to_numeric(prepared["rules_applied"], errors="coerce").fillna(0).eq(1).all():
        sample = prepared.loc[
            ~pd.to_numeric(prepared["rules_applied"], errors="coerce").fillna(0).eq(1),
            ["team", "player_clean_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"canonical rules did not apply to full roster: {sample}")
    bad = _forbidden_columns(prepared)
    if bad:
        raise RuntimeError(f"prepared football universe contains sportsbook/outcome fields: {bad}")
    if "team_wp" in prepared.columns:
        raise RuntimeError("market-derived team_wp leaked into football-only lock")

    baseline = prepared.copy()
    candidate = prepared.copy()
    transfers = vacancy[["team", "successor_player_clean_key", "transfer_rush_share"]].copy()
    transfers["team"] = transfers["team"].astype(str).str.upper().str.strip()
    transfers["successor_player_clean_key"] = _key(transfers["successor_player_clean_key"])
    transfers["transfer_rush_share"] = pd.to_numeric(transfers["transfer_rush_share"], errors="raise")

    applied = 0
    for _, tr in transfers.iterrows():
        mask = (
            candidate["team"].astype(str).str.upper().eq(tr["team"])
            & _key(candidate["player_clean_key"]).eq(tr["successor_player_clean_key"])
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"expected one active successor {tr['team']}/{tr['successor_player_clean_key']}, got {int(mask.sum())}")
        old = pd.to_numeric(candidate.loc[mask, "rules_rush_share"], errors="coerce")
        if old.isna().any():
            raise RuntimeError(f"successor lacks production rules_rush_share {tr['team']}/{tr['successor_player_clean_key']}")
        candidate.loc[mask, "rules_rush_share"] = old + float(tr["transfer_rush_share"])
        applied += 1
    if applied != len(transfers):
        raise RuntimeError("not every frozen transfer was applied exactly once")

    # Prove the candidate mutates only the frozen opportunity field.
    compare_cols = [c for c in baseline.columns if c != "rules_rush_share"]
    for c in compare_cols:
        b = baseline[c]
        q = candidate[c]
        equal = b.eq(q) | (b.isna() & q.isna())
        if not bool(equal.all()):
            raise RuntimeError(f"candidate changed forbidden football field: {c}")
    if not np.allclose(
        pd.to_numeric(baseline["rules_ypc"], errors="coerce"),
        pd.to_numeric(candidate["rules_ypc"], errors="coerce"),
        equal_nan=True,
    ):
        raise RuntimeError("candidate altered YPC/efficiency")

    baseline_trace: list[dict] = []
    candidate_trace: list[dict] = []
    baseline_sim = simulate(baseline, allocation_trace=baseline_trace)
    candidate_sim = simulate(candidate, allocation_trace=candidate_trace)
    if baseline_sim.iterations != candidate_sim.iterations or baseline_sim.iterations != 25000:
        raise RuntimeError(f"unexpected simulation iterations {baseline_sim.iterations}/{candidate_sim.iterations}")

    bt = _trace_frame(baseline_trace, "baseline")
    ct = _trace_frame(candidate_trace, "candidate")
    trace = bt.merge(ct, on=["event_id", "team", "player_clean_key"], how="inner", validate="one_to_one")

    # Freeze every active RB/FB in a vacancy-event team. Direct recipients are
    # flagged separately because finite-volume normalization can move nonrecipient
    # teammates as well.
    cohort = baseline.loc[
        baseline["team"].astype(str).str.upper().isin(event_teams)
        & _pos_family(baseline["position"]).isin(["RB", "FB"])
    ].copy()
    cohort = cohort.sort_values(["team", "player_clean_key"]).drop_duplicates(["team", "player_clean_key"])
    if cohort.empty:
        raise RuntimeError("vacancy team active RB/FB cohort is empty")

    ml["team"] = ml["team"].astype(str).str.upper().str.strip()
    ml["player_clean_key"] = _key(ml["player_clean_key"])
    state["team"] = state["team"].astype(str).str.upper().str.strip()
    state["player_clean_key"] = _key(state["player_clean_key"])

    transfer_map = {
        (str(r["team"]), str(r["successor_player_clean_key"])): float(r["transfer_rush_share"])
        for _, r in transfers.iterrows()
    }

    rows = []
    for _, player in cohort.iterrows():
        team = str(player["team"]).upper().strip()
        pkey = str(player["player_clean_key"]).strip()
        event = str(player["event_id"])
        transfer = transfer_map.get((team, pkey), 0.0)
        for market in ("rush_att", "rush_yards"):
            bdraw = lookup(baseline_sim, player, market)
            cdraw = lookup(candidate_sim, player, market)
            if bdraw is None or cdraw is None:
                raise RuntimeError(f"missing simulation distribution {team}/{pkey}/{market}")
            bmc = float(np.mean(np.asarray(bdraw, dtype=float)))
            cmc = float(np.mean(np.asarray(cdraw, dtype=float)))
            mlv = _diag_value(ml, team, pkey, f"ml_{market}")
            stv = _diag_value(state, team, pkey, f"state_{market}")
            comp = pd.DataFrame([
                {"arm": "baseline", "market": market, "mc_proj": bmc, "ml_proj": mlv, "state_proj": stv},
                {"arm": "candidate", "market": market, "mc_proj": cmc, "ml_proj": mlv, "state_proj": stv},
            ])
            ens = apply_ensemble(comp, weights=weights)
            b = ens.loc[ens["arm"].eq("baseline")].iloc[0]
            c = ens.loc[ens["arm"].eq("candidate")].iloc[0]
            rows.append({
                "target_season": 2026,
                "target_week": 3,
                "event_id": event,
                "team": team,
                "opponent": str(player.get("opponent", "")),
                "player": player.get("player"),
                "player_clean_key": pkey,
                "position": player.get("position"),
                "direct_transfer_recipient": int(transfer > 0),
                "frozen_transfer_rush_share": transfer,
                "baseline_rules_rush_share": float(pd.to_numeric(pd.Series([player.get("rules_rush_share")]), errors="coerce").iloc[0]),
                "candidate_rules_rush_share": float(pd.to_numeric(pd.Series([player.get("rules_rush_share")]), errors="coerce").iloc[0]) + transfer,
                "frozen_rules_ypc": float(pd.to_numeric(pd.Series([player.get("rules_ypc")]), errors="coerce").iloc[0]),
                "market": market,
                "baseline_mc_proj": bmc,
                "candidate_mc_proj": cmc,
                "frozen_ml_proj": mlv,
                "frozen_state_proj": stv,
                "baseline_ensemble_proj": float(b["ensemble_proj"]),
                "candidate_ensemble_proj": float(c["ensemble_proj"]),
                "ensemble_delta": float(c["ensemble_proj"] - b["ensemble_proj"]),
                "ensemble_status": str(b["ensemble_status"]),
                "ensemble_method": str(b["ensemble_method"]),
                "ensemble_weight_mc": float(b["ensemble_weight_mc"]),
                "ensemble_weight_ml": float(b["ensemble_weight_ml"]),
                "ensemble_weight_state": float(b["ensemble_weight_state"]),
                "ensemble_calibration_rows": int(b["ensemble_calibration_rows"]),
                "simulation_iterations": int(baseline_sim.iterations),
                "simulation_seed": 42,
            })

    projections = pd.DataFrame(rows)
    projections = projections.merge(trace, on=["event_id", "team", "player_clean_key"], how="left", validate="many_to_one")
    if projections[["baseline_final_player_probability", "candidate_final_player_probability"]].isna().any().any():
        raise RuntimeError("allocation trace failed to cover locked RB/FB cohort")

    # The no-outcome artifact remains free of target-game actuals and sportsbook data.
    if _forbidden_columns(projections):
        raise RuntimeError(f"projection lock accidentally contains forbidden fields: {_forbidden_columns(projections)}")
    projections.to_csv(outdir / "rb_vacancy_week3_pregame_projection_lock.csv", index=False)

    team_audit = projections.loc[projections["market"].eq("rush_att")].groupby(["event_id", "team"], as_index=False).agg(
        active_rb_fb=("player_clean_key", "nunique"),
        direct_transfer_recipients=("direct_transfer_recipient", "sum"),
        frozen_transfer_rush_share=("frozen_transfer_rush_share", "sum"),
        baseline_rbfb_rush_att_mean=("baseline_ensemble_proj", "sum"),
        candidate_rbfb_rush_att_mean=("candidate_ensemble_proj", "sum"),
        baseline_mc_team_rush_total=("baseline_team_rush_total_mean", "first"),
        candidate_mc_team_rush_total=("candidate_team_rush_total_mean", "first"),
        baseline_raw_team_rush_share_sum=("baseline_raw_team_rush_share_sum", "first"),
        candidate_raw_team_rush_share_sum=("candidate_raw_team_rush_share_sum", "first"),
        baseline_residual_probability=("baseline_residual_probability", "first"),
        candidate_residual_probability=("candidate_residual_probability", "first"),
    )
    team_audit["rbfb_rush_att_mean_delta"] = team_audit["candidate_rbfb_rush_att_mean"] - team_audit["baseline_rbfb_rush_att_mean"]
    team_audit.to_csv(outdir / "rb_vacancy_week3_team_allocation_audit.csv", index=False)

    payload = {
        "disposition": "RB_VACANCY_OPPORTUNITY_V1_WEEK3_PREGAME_BASELINE_AND_CANDIDATE_LOCKED",
        "target_season": 2026,
        "target_week": 3,
        "source_full_slate_run": int(a.source_run),
        "source_full_slate_artifact": int(a.source_artifact),
        "source_full_slate_digest": str(a.source_digest),
        "production_source_sha": str(a.source_sha),
        "vacancy_lock_run": int(a.lock_run),
        "vacancy_lock_artifact": int(a.lock_artifact),
        "vacancy_lock_digest": str(a.lock_digest),
        "event_teams": event_teams,
        "active_rb_fb_rows": int(cohort[["team", "player_clean_key"]].drop_duplicates().shape[0]),
        "direct_transfer_rows": int(len(transfers)),
        "projection_rows": int(len(projections)),
        "markets": ["rush_att", "rush_yards"],
        "simulation_iterations": int(baseline_sim.iterations),
        "simulation_seed": 42,
        "candidate_mutated_fields": ["rules_rush_share"],
        "ypc_efficiency_changed": False,
        "ml_state_components_changed": False,
        "ensemble_weights_changed": False,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_attached": 0,
        "candidate_variants_scored": 1,
        "future_grading_cohort": "all production-eligible RB/FB on frozen vacancy-event teams; direct recipients flagged separately",
        "predeclared_high_volume_actual_carry_slices": ["actual_rushes>=20", "actual_rushes>=25"],
        "grading_forbidden_until_games_complete": True,
    }
    (outdir / "rb_vacancy_week3_pregame_lock_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("\nLOCKED PROJECTIONS")
    print(projections[[
        "team","player","market","direct_transfer_recipient","frozen_transfer_rush_share",
        "baseline_ensemble_proj","candidate_ensemble_proj","ensemble_delta",
        "baseline_final_player_probability","candidate_final_player_probability",
    ]].to_string(index=False))
    print("\nTEAM AUDIT")
    print(team_audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
