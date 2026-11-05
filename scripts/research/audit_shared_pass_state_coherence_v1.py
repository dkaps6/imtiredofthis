#!/usr/bin/env python3
"""Read-only current-stack audit for split QB/receiver Monte Carlo state.

This script introduces no football parameters and does not score outcomes.
It reconstructs the current football-only Full Slate universe without sportsbook
offers, applies the promoted M38 -> TE-R5P -> WR-R15 entitlement stack, runs the
canonical state-capturing simulator, then compares:

1. the final C2-selected QB pass-yard draw;
2. canonical production receiver arrays that currently remain installed; and
3. the receiver arrays generated inside the exact C2 completed-pass process.

C1/C3 are not used. Production files are not modified beyond normal research
artifacts written under data/research/shared_pass_state_coherence_v1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
from scripts._opponent_map import canon_team
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.qb_c2_production_adapter_v1 import (
    annotate_primary_qbs,
    apply_qb_c2_selector,
)
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.simulation_c2_qb_candidate import (
    C2_RESIDUAL_CATCH_RATE,
    C2_RESIDUAL_YPT,
    C2_YPR_MAX,
    C2_YPR_MIN,
    PASS_CATCHER_POSITIONS,
    _target_shares,
)
from scripts.simulation_v2 import _allocate_counts, _clip_prob, _num, _player_key

OUT = Path("data/research/shared_pass_state_coherence_v1")
SUMMARY = OUT / "summary.json"
TEAM_CSV = OUT / "team_draw_coherence.csv"
PLAYER_CSV = OUT / "player_shadow_divergence.csv"

ITERATIONS = 5000
BASE_SEED = 42
C2_SEED = 5601
SKILL = {"QB", "RB", "FB", "WR", "TE"}


def _position_family(value: object) -> str:
    p = str(value or "").upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    if p.startswith("QB"):
        return "QB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p or "OTHER"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float)
    return {
        "p10": float(np.quantile(x, .10)),
        "p25": float(np.quantile(x, .25)),
        "p50": float(np.quantile(x, .50)),
        "p75": float(np.quantile(x, .75)),
        "p90": float(np.quantile(x, .90)),
    }


def _build_synthetic_pricing_metrics() -> pd.DataFrame:
    """Create football-only comparison rows; sportsbook coverage defines nothing."""
    path = Path("data/player_form_consensus.csv")
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError("player_form_consensus missing")
    form = pd.read_csv(path, low_memory=False)
    form.columns = [str(c).strip().lower() for c in form.columns]
    required = {"player", "team", "opponent", "season", "week", "position"}
    missing = sorted(required - set(form.columns))
    if missing:
        raise RuntimeError(f"player_form_consensus missing columns: {missing}")
    if "player_clean_key" not in form.columns:
        form["player_clean_key"] = form["player"].map(lambda x: "".join(ch.lower() for ch in str(x) if ch.isalnum()))
    form["team"] = form["team"].map(canon_team)
    form["opponent"] = form["opponent"].map(canon_team)
    form["_position_family"] = form["position"].map(_position_family)
    form = form.loc[form["_position_family"].isin(SKILL)].drop(columns="_position_family").copy()
    if form.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("synthetic football-only comparison frame has duplicate player/team")
    if form["team"].nunique() != 32:
        raise RuntimeError(f"synthetic football-only comparison frame expected 32 teams, got {form['team'].nunique()}")
    form["event_id"] = [
        base._canonical_game(t, o, s, w)
        for t, o, s, w in zip(form["team"], form["opponent"], form["season"], form["week"])
    ]
    form["market"] = "football_universe"
    forbidden = sorted(base.FORBIDDEN_SIM_COLUMNS & set(form.columns))
    if forbidden:
        raise RuntimeError(f"sportsbook fields leaked into synthetic football frame: {forbidden}")
    frame = apply_bayesian_to_metrics(form)
    frame = apply_rules_to_metrics(frame)
    if not pd.to_numeric(frame["bayes_applied"], errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("Bayesian context missing from synthetic football frame")
    if not pd.to_numeric(frame["rules_applied"], errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("rule context missing from synthetic football frame")
    return frame


def _capture_c2_shadow(base_state, metrics: pd.DataFrame, anchors: dict[tuple[str, str], float]) -> dict:
    """Mirror apply_c2 exactly while retaining the shadow receiver arrays."""
    rng = np.random.default_rng(C2_SEED)
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)
    game_key = "event_id"
    players = frame.sort_values([game_key, "team", "player_clean_key"]).drop_duplicates(
        [game_key, "team", "player_clean_key"], keep="last"
    )
    out: dict[tuple[str, str], dict] = {}
    for game, gdf in players.groupby(game_key, dropna=False):
        for team, tdf0 in gdf.groupby("team", dropna=False):
            if pd.isna(team) or not str(team).strip():
                continue
            gs, ts = str(game), str(team)
            tdf = tdf0.reset_index(drop=True)
            pass_att = np.asarray(base_state.team_states[(gs, ts, "pass_att")], int)
            pass_eff = np.asarray(base_state.team_states[(gs, ts, "pass_eff_shock")], float)
            shares = _target_shares(tdf)
            positions = tdf.get("position", pd.Series("", index=tdf.index)).fillna("").astype(str).str.upper().str.strip().to_numpy()
            mask = np.isin(positions, list(PASS_CATCHER_POSITIONS))
            shares = np.where(mask, shares, 0.0)
            targets = _allocate_counts(rng, pass_att, shares)
            residual_targets = np.maximum(0, pass_att - targets.sum(1))
            player_yards: dict[str, np.ndarray] = {}
            player_receptions: dict[str, np.ndarray] = {}
            for j, (_, row) in enumerate(tdf.iterrows()):
                if not mask[j]:
                    continue
                pk = _player_key(row)
                if not pk:
                    continue
                catch = _clip_prob(
                    _num(row, "rules_catch_rate", "bayes_receptions_per_target", "receptions_per_target", "catch_rate", default=C2_RESIDUAL_CATCH_RATE),
                    C2_RESIDUAL_CATCH_RATE,
                )
                recs = rng.binomial(targets[:, j], catch)
                ypt = _num(row, "rules_ypt", "bayes_ypt", "ypt")
                ypt = C2_RESIDUAL_YPT if not np.isfinite(ypt) or ypt <= 0 else float(ypt)
                ypr = float(np.clip(ypt / catch, C2_YPR_MIN, C2_YPR_MAX))
                vol = float(np.clip(_num(row, "rules_volatility_mult", default=1.0), .75, 1.5))
                mu = recs.astype(float) * ypr * pass_eff
                sd = np.maximum(3.0, np.sqrt(np.maximum(recs, 1)) * ypr * .55) * vol
                y = np.clip(rng.normal(mu, sd), 0.0, None)
                y = np.where(recs > 0, y, 0.0)
                player_yards[pk] = y
                player_receptions[pk] = recs.astype(float)
            rr = rng.binomial(residual_targets, C2_RESIDUAL_CATCH_RATE)
            rypr = C2_RESIDUAL_YPT / C2_RESIDUAL_CATCH_RATE
            rmu = rr.astype(float) * rypr * pass_eff
            rsd = np.maximum(3.0, np.sqrt(np.maximum(rr, 1)) * rypr * .55)
            residual_yards = np.where(rr > 0, np.clip(rng.normal(rmu, rsd), 0.0, None), 0.0)
            modeled_total = (
                np.sum(np.vstack(list(player_yards.values())), axis=0)
                if player_yards else np.zeros(base_state.iterations)
            )
            raw_total = modeled_total + residual_yards
            anchor = float(anchors[(gs, ts)])
            if not np.isfinite(anchor) or anchor <= 0 or not np.isfinite(raw_total.mean()) or raw_total.mean() <= 0:
                raise RuntimeError(f"invalid C2 shadow anchor/raw total {gs} {ts} anchor={anchor}")
            scale = anchor / float(raw_total.mean())
            out[(gs, ts)] = {
                "player_yards_raw": player_yards,
                "player_yards_scaled": {k: v * scale for k, v in player_yards.items()},
                "player_receptions": player_receptions,
                "residual_yards_scaled": residual_yards * scale,
                "modeled_total_scaled": modeled_total * scale,
                "total_scaled": raw_total * scale,
                "scale": float(scale),
                "pass_att": pass_att,
            }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic = _build_synthetic_pricing_metrics()

    # Preserve the exact production full-universe implementation reference.
    base._identity_frame = v2._canonical_identity_frame
    universe, _, universe_audit = v3._build_with_promoted_entitlement_specialists(synthetic)

    seasons = sorted(pd.to_numeric(universe["season"], errors="coerce").dropna().astype(int).unique())
    weeks = sorted(pd.to_numeric(universe["week"], errors="coerce").dropna().astype(int).unique())
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"audit expected one season/week, got seasons={seasons} weeks={weeks}")
    season, week = seasons[0], weeks[0]

    stateful = v3.simulate_with_states(universe, iterations=ITERATIONS, seed=BASE_SEED)
    selected_state, selector_audit, selector_payload = apply_qb_c2_selector(
        stateful, universe, season=season, week=week
    )

    annotated, _ = annotate_primary_qbs(universe, season=season, week=week)
    primary = annotated.loc[pd.to_numeric(annotated["qb_projection_eligible"], errors="coerce").eq(1)].copy()
    anchors: dict[tuple[str, str], float] = {}
    primary_by_team: dict[str, pd.Series] = {}
    for _, row in primary.iterrows():
        game, team, pk = str(row["event_id"]), canon_team(row["team"]), str(row["player_clean_key"])
        arr = np.asarray(stateful.values[(game, pk, "pass_yards")], float)
        anchors[(game, team)] = float(arr.mean())
        primary_by_team[team] = row

    shadow = _capture_c2_shadow(stateful, annotated, anchors)
    selected_teams = set(selector_audit.loc[selector_audit["selector_c2_selected"].eq(1), "team"].astype(str))

    # Verify the independently captured shadow recreates the installed selected C2 QB arrays exactly.
    max_shadow_qb_gap = 0.0
    for team in sorted(selected_teams):
        row = primary_by_team[team]
        game, pk = str(row["event_id"]), str(row["player_clean_key"])
        installed = np.asarray(selected_state.values[(game, pk, "pass_yards")], float)
        captured = np.asarray(shadow[(game, team)]["total_scaled"], float)
        max_shadow_qb_gap = max(max_shadow_qb_gap, float(np.max(np.abs(installed - captured))))
    if max_shadow_qb_gap > 1e-10:
        raise RuntimeError(f"research shadow capture drifted from production C2 selected arrays: {max_shadow_qb_gap}")

    player_rows: list[dict] = []
    team_rows: list[dict] = []

    lookup = annotated.copy()
    lookup["_position_family"] = lookup["position"].map(_position_family)
    lookup = lookup.loc[lookup["_position_family"].isin({"WR", "TE", "RB", "FB"})].copy()

    for team in sorted(selected_teams):
        qrow = primary_by_team[team]
        game, qpk = str(qrow["event_id"]), str(qrow["player_clean_key"])
        final_qb = np.asarray(selected_state.values[(game, qpk, "pass_yards")], float)
        team_part = lookup.loc[(lookup["team"].astype(str).eq(team)) & (lookup["event_id"].astype(str).eq(game))].copy()
        canonical_parts = []
        canonical_recs = []
        for _, row in team_part.iterrows():
            pk = str(row["player_clean_key"])
            canonical_parts.append(np.asarray(stateful.values[(game, pk, "rec_yards")], float))
            canonical_recs.append(np.asarray(stateful.values[(game, pk, "receptions")], float))
        canonical_total = np.sum(np.vstack(canonical_parts), axis=0) if canonical_parts else np.zeros(ITERATIONS)

        sh = shadow[(game, team)]
        shadow_modeled = np.asarray(sh["modeled_total_scaled"], float)
        shadow_residual = np.asarray(sh["residual_yards_scaled"], float)
        shadow_total = np.asarray(sh["total_scaled"], float)
        identity_gap = float(np.max(np.abs(final_qb - shadow_total)))

        q_qb = _quantiles(final_qb)
        q_can = _quantiles(canonical_total)
        opp_hi_low = float(np.mean((final_qb >= q_qb["p75"]) & (canonical_total <= q_can["p25"])))
        opp_low_hi = float(np.mean((final_qb <= q_qb["p25"]) & (canonical_total >= q_can["p75"])))
        gap = final_qb - canonical_total

        sel = selector_audit.loc[selector_audit["team"].astype(str).eq(team)].iloc[0]
        team_rows.append({
            "season": season,
            "week": week,
            "event_id": game,
            "team": team,
            "qb_player": str(qrow["player"]),
            "selector_delta_pass_attempts": float(sel["selector_delta_pass_attempts"]),
            "qb_mean": float(final_qb.mean()),
            "qb_sd": float(final_qb.std(ddof=1)),
            "canonical_modeled_receiver_mean": float(canonical_total.mean()),
            "canonical_modeled_receiver_sd": float(canonical_total.std(ddof=1)),
            "shadow_modeled_receiver_mean": float(shadow_modeled.mean()),
            "shadow_modeled_receiver_sd": float(shadow_modeled.std(ddof=1)),
            "shadow_residual_mean": float(shadow_residual.mean()),
            "shadow_residual_share_of_qb_mean": float(shadow_residual.mean() / final_qb.mean()),
            "qb_vs_canonical_receiver_corr": _corr(final_qb, canonical_total),
            "qb_vs_shadow_total_corr": _corr(final_qb, shadow_total),
            "canonical_vs_shadow_modeled_corr": _corr(canonical_total, shadow_modeled),
            "canonical_to_shadow_modeled_mean_ratio": float(canonical_total.mean() / shadow_modeled.mean()) if shadow_modeled.mean() else np.nan,
            "canonical_to_shadow_modeled_sd_ratio": float(canonical_total.std(ddof=1) / shadow_modeled.std(ddof=1)) if shadow_modeled.std(ddof=1) else np.nan,
            "qb_minus_canonical_mean_gap": float(gap.mean()),
            "qb_minus_canonical_p10": float(np.quantile(gap, .10)),
            "qb_minus_canonical_p25": float(np.quantile(gap, .25)),
            "qb_minus_canonical_p50": float(np.quantile(gap, .50)),
            "qb_minus_canonical_p75": float(np.quantile(gap, .75)),
            "qb_minus_canonical_p90": float(np.quantile(gap, .90)),
            "opposing_tail_qb_hi_receiver_low_rate": opp_hi_low,
            "opposing_tail_qb_low_receiver_hi_rate": opp_low_hi,
            "c2_shadow_identity_max_gap": identity_gap,
            "c2_scale": float(sh["scale"]),
        })

        for _, row in team_part.iterrows():
            pk = str(row["player_clean_key"])
            if pk not in sh["player_yards_scaled"]:
                continue
            canonical = np.asarray(stateful.values[(game, pk, "rec_yards")], float)
            canonical_rec = np.asarray(stateful.values[(game, pk, "receptions")], float)
            candidate = np.asarray(sh["player_yards_scaled"][pk], float)
            shadow_rec = np.asarray(sh["player_receptions"][pk], float)
            ae = np.abs(canonical - candidate)
            player_rows.append({
                "season": season,
                "week": week,
                "event_id": game,
                "team": team,
                "player": str(row["player"]),
                "player_clean_key": pk,
                "position_family": _position_family(row["position"]),
                "entitlement_tgt_share": float(pd.to_numeric(pd.Series([row.get("entitlement_tgt_share")]), errors="coerce").fillna(0).iloc[0]),
                "canonical_mean": float(canonical.mean()),
                "shadow_scaled_mean": float(candidate.mean()),
                "mean_gap_shadow_minus_canonical": float(candidate.mean() - canonical.mean()),
                "canonical_sd": float(canonical.std(ddof=1)),
                "shadow_scaled_sd": float(candidate.std(ddof=1)),
                "sd_ratio_shadow_to_canonical": float(candidate.std(ddof=1) / canonical.std(ddof=1)) if canonical.std(ddof=1) else np.nan,
                "array_corr": _corr(canonical, candidate),
                "p90_abs_draw_gap": float(np.quantile(ae, .90)),
                "canonical_zero_rec_positive_yards_rate": float(np.mean((canonical_rec <= 0) & (canonical > 1e-12))),
                "shadow_zero_rec_positive_yards_rate": float(np.mean((shadow_rec <= 0) & (candidate > 1e-12))),
            })

    team_df = pd.DataFrame(team_rows)
    player_df = pd.DataFrame(player_rows)
    if team_df.empty or player_df.empty:
        raise RuntimeError("coherence audit produced empty selected-team/player outputs")
    if float(team_df["c2_shadow_identity_max_gap"].max()) > 1e-10:
        raise RuntimeError("C2 shadow identity failed")
    if float(player_df["shadow_zero_rec_positive_yards_rate"].max()) > 0:
        raise RuntimeError("C2 shadow produced receiving yards with zero receptions")

    try:
        player_df["entitlement_quartile"] = pd.qcut(
            player_df["entitlement_tgt_share"].rank(method="first"),
            4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        ).astype(str)
    except Exception:
        player_df["entitlement_quartile"] = "NA"

    by_position = {}
    for pos, part in player_df.groupby("position_family"):
        by_position[str(pos)] = {
            "players": int(len(part)),
            "median_array_corr": float(part["array_corr"].median()),
            "mean_abs_mean_gap": float(part["mean_gap_shadow_minus_canonical"].abs().mean()),
            "median_p90_abs_draw_gap": float(part["p90_abs_draw_gap"].median()),
            "canonical_zero_rec_positive_yards_rate": float(
                np.average(
                    part["canonical_zero_rec_positive_yards_rate"],
                    weights=np.maximum(part["entitlement_tgt_share"], 1e-9),
                )
            ),
        }

    by_entitlement = {}
    for quartile, part in player_df.groupby("entitlement_quartile"):
        by_entitlement[str(quartile)] = {
            "players": int(len(part)),
            "median_array_corr": float(part["array_corr"].median()),
            "mean_abs_mean_gap": float(part["mean_gap_shadow_minus_canonical"].abs().mean()),
            "median_p90_abs_draw_gap": float(part["p90_abs_draw_gap"].median()),
        }

    payload = {
        "study": "SHARED_PASS_STATE_COHERENCE_V1",
        "status": "READ_ONLY_AUDIT_COMPLETE",
        "season": int(season),
        "week": int(week),
        "iterations": ITERATIONS,
        "base_seed": BASE_SEED,
        "c2_seed": C2_SEED,
        "sportsbook_inputs_used": False,
        "target_game_outcomes_used": False,
        "production_changed": False,
        "c1_used": False,
        "c3_used": False,
        "selected_qb_teams": int(len(team_df)),
        "receiver_player_rows": int(len(player_df)),
        "shadow_capture_vs_installed_c2_max_gap": float(max_shadow_qb_gap),
        "c2_shadow_identity_max_gap": float(team_df["c2_shadow_identity_max_gap"].max()),
        "median_qb_vs_canonical_receiver_corr": float(team_df["qb_vs_canonical_receiver_corr"].median()),
        "median_canonical_vs_shadow_modeled_corr": float(team_df["canonical_vs_shadow_modeled_corr"].median()),
        "median_abs_qb_minus_canonical_mean_gap": float(team_df["qb_minus_canonical_mean_gap"].abs().median()),
        "mean_opposing_tail_rate": float(
            (team_df["opposing_tail_qb_hi_receiver_low_rate"] + team_df["opposing_tail_qb_low_receiver_hi_rate"]).mean()
        ),
        "median_shadow_residual_share_of_qb_mean": float(team_df["shadow_residual_share_of_qb_mean"].median()),
        "canonical_zero_rec_positive_yards_rate_player_median": float(player_df["canonical_zero_rec_positive_yards_rate"].median()),
        "shadow_zero_rec_positive_yards_rate_max": float(player_df["shadow_zero_rec_positive_yards_rate"].max()),
        "by_position": by_position,
        "by_entitlement_quartile": by_entitlement,
        "selector_payload": selector_payload,
        "universe_audit": {
            "football_player_rows": int(universe_audit["football_player_rows"]),
            "football_teams": int(universe_audit["football_teams"]),
            "canonical_games": int(universe_audit["canonical_games"]),
            "sportsbook_rows_used_to_define_player_universe": int(universe_audit["sportsbook_rows_used_to_define_player_universe"]),
        },
        "team_output": str(TEAM_CSV),
        "player_output": str(PLAYER_CSV),
    }

    team_df.to_csv(TEAM_CSV, index=False)
    player_df.to_csv(PLAYER_CSV, index=False)
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[shared_pass_state_coherence_v1] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
