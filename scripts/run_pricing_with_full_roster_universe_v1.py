#!/usr/bin/env python3
"""Price the slate from a sportsbook-independent full football simulation universe.

Production-hardening contract:
- PlayerForm/Ourlads + schedule define who exists in the football simulation.
- Sportsbook offer availability NEVER defines target/carry competition.
- Book/line/odds and market-derived team win probability are absent from the
  simulation universe.
- Provider event ids are installed only *after* simulation as lookup aliases so
  existing pricing provenance/reconciliation remains exact.
- Promoted RB P3 rush+receiving conservation is preserved for priced RB/FB
  rush+receiving markets.

This wrapper changes plumbing, not research parameters.  It deliberately leaves
model-quality research (finite entitlement, TE-R5P, C2, ATD replacement) behind
separate frozen promotion gates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_v2 as pricing
from scripts._opponent_map import canon_team
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.rb_pricing_adapter_v1 import load_rb_context, lookup_rb_projection
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.simulation_v2 import MARKET_MAP, _player_key, lookup, simulate as canonical_simulate

DATA = Path("data")
AUDIT_CSV = DATA / "football_simulation_universe_audit.csv"
AUDIT_JSON = DATA / "football_simulation_universe_audit.json"
UNIVERSE_CSV = DATA / "football_simulation_universe.csv"
RB_AUDIT_CSV = DATA / "rb_rush_rec_conservation_input_audit.csv"
RB_AUDIT_JSON = DATA / "rb_rush_rec_conservation_input_audit.json"

FORBIDDEN_SIM_COLUMNS = {
    "line", "source_line", "over_odds", "under_odds", "book", "book_title",
    "vegas_line", "vegas_odds", "market_prob", "edge_pct", "edge_abs",
    "home_wp", "away_wp", "team_wp",
}
SKILL_FAMILIES = {"QB", "RB", "FB", "WR", "TE"}
CORE_COMPARE = (
    "rules_plays_est", "rules_pass_rate", "rules_tgt_share", "rules_rush_share",
    "rules_ypt", "rules_ypc", "rules_ypa", "rules_catch_rate",
    "rules_volatility_mult", "rules_pass_eff_mult", "rules_rush_eff_mult",
    "offensive_td_rate", "rz_share", "rz_tgt_share", "rz_carry_share",
)


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"full-roster simulation required {label} missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"full-roster simulation required {label} has zero rows: {path}")
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _position_family(value) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
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
    return p


def _canonical_game(team, opponent, season, week) -> str:
    t = canon_team(team)
    o = canon_team(opponent)
    if not t or not o:
        raise RuntimeError(f"cannot construct canonical simulation game: team={team} opponent={opponent}")
    try:
        s = int(float(season))
        w = int(float(week))
    except Exception as exc:
        raise RuntimeError(f"invalid season/week for canonical simulation game: {season}/{week}") from exc
    a, b = sorted([t, o])
    return f"{s}_{w:02d}_{a}_{b}"


def _identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team"] = out["team"].map(canon_team)
    source = out["player_clean_key"] if "player_clean_key" in out.columns else out["player"]
    out["player_clean_key"] = source.astype("string").fillna("").str.strip()
    return out


def _build_full_universe(pricing_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict]:
    form = _read(DATA / "player_form_consensus.csv", "PlayerForm consensus")
    context = _read(DATA / "model_context_bridge.csv", "model context bridge")
    need = {"player", "team", "opponent", "season", "week", "position"}
    missing = need - set(form.columns)
    if missing:
        raise RuntimeError(f"PlayerForm consensus missing full-universe columns: {sorted(missing)}")

    form = _identity_frame(form)
    form["team"] = form["team"].map(canon_team)
    form["opponent"] = form["opponent"].map(canon_team)
    form["position_family"] = form["position"].map(_position_family)
    form = form.loc[form["position_family"].isin(SKILL_FAMILIES)].copy()
    if form.empty:
        raise RuntimeError("football simulation universe has zero QB/RB/FB/WR/TE rows")
    if form.duplicated(["team", "player_clean_key"]).any():
        sample = form.loc[form.duplicated(["team", "player_clean_key"], keep=False), ["player", "team", "player_clean_key"]].head(20).to_dict("records")
        raise RuntimeError(f"football simulation universe duplicate player/team identities: {sample}")
    if form["team"].nunique() != 32:
        raise RuntimeError(f"football simulation universe must cover 32 teams, found {form['team'].nunique()}")

    # model_context_bridge is the already-certified current football context
    # roster.  Require exact identity equality rather than merely a row count.
    context = _identity_frame(context)
    context["position_family"] = context["position"].map(_position_family)
    context = context.loc[context["position_family"].isin(SKILL_FAMILIES)].copy()
    form_keys = set(zip(form["team"].astype(str), form["player_clean_key"].astype(str)))
    context_keys = set(zip(context["team"].astype(str), context["player_clean_key"].astype(str)))
    missing_from_form = sorted(context_keys - form_keys)
    extra_in_form = sorted(form_keys - context_keys)
    if missing_from_form or extra_in_form:
        raise RuntimeError(
            "football simulation universe != certified model-context roster; "
            f"missing_from_form={missing_from_form[:20]} extra_in_form={extra_in_form[:20]}"
        )

    # Canonical game identity comes only from team/opponent/schedule state.
    form["event_id"] = [
        _canonical_game(t, o, s, w)
        for t, o, s, w in zip(form["team"], form["opponent"], form["season"], form["week"])
    ]
    form["market"] = "football_universe"

    forbidden_present = sorted(FORBIDDEN_SIM_COLUMNS & set(form.columns))
    if forbidden_present:
        raise RuntimeError(f"sportsbook/market fields leaked into football simulation universe: {forbidden_present}")

    # Bayesian/rule layers are player/team football context and are therefore
    # rebuilt on the complete roster before Monte Carlo allocation.
    universe = apply_bayesian_to_metrics(form)
    if not pd.to_numeric(universe.get("bayes_applied", 0), errors="coerce").fillna(0).eq(1).all():
        sample = universe.loc[~pd.to_numeric(universe.get("bayes_applied", 0), errors="coerce").fillna(0).eq(1), ["player", "team"]].head(20).to_dict("records")
        raise RuntimeError(f"full football universe missing Bayesian context: {sample}")
    universe = apply_rules_to_metrics(universe)
    if not pd.to_numeric(universe.get("rules_applied", 0), errors="coerce").fillna(0).eq(1).all():
        sample = universe.loc[~pd.to_numeric(universe.get("rules_applied", 0), errors="coerce").fillna(0).eq(1), ["player", "team"]].head(20).to_dict("records")
        raise RuntimeError(f"full football universe missing canonical rules: {sample}")

    # Current pricing rows should carry identical per-player football assumptions
    # for every overlapping player.  The one intentional exception is team_wp,
    # which is market-derived and is excluded from the full football universe.
    priced = _identity_frame(pricing_metrics)
    priced = priced.sort_values(["team", "player_clean_key"]).drop_duplicates(["team", "player_clean_key"], keep="last")
    compare = universe.merge(
        priced[["team", "player_clean_key", *[c for c in CORE_COMPARE if c in priced.columns]]],
        on=["team", "player_clean_key"], how="inner", suffixes=("_football", "_priced"), validate="one_to_one",
    )
    max_diffs: dict[str, float] = {}
    for col in CORE_COMPARE:
        a, b = f"{col}_football", f"{col}_priced"
        if a not in compare.columns or b not in compare.columns:
            continue
        av = pd.to_numeric(compare[a], errors="coerce")
        bv = pd.to_numeric(compare[b], errors="coerce")
        both = av.notna() & bv.notna()
        max_diff = float((av.loc[both] - bv.loc[both]).abs().max()) if both.any() else 0.0
        max_diffs[col] = max_diff
        if max_diff > 1e-9:
            raise RuntimeError(f"full-universe football assumption drift for {col}: max_abs_diff={max_diff}")

    priced_keys = set(zip(priced["team"].astype(str), priced["player_clean_key"].astype(str)))
    priced_missing = sorted(priced_keys - form_keys)
    if priced_missing:
        raise RuntimeError(f"priced players absent from football universe: {priced_missing[:20]}")

    # Provider event ids are NOT used to create or group the simulation. Build a
    # post-simulation alias map only, so existing exact offer reconciliation can
    # keep using the provider event identity.
    pm = pricing_metrics.copy()
    for c in ("team", "opponent"):
        pm[c] = pm[c].map(canon_team)
    pm["canonical_event_id"] = [
        _canonical_game(t, o, s, w)
        for t, o, s, w in zip(pm["team"], pm["opponent"], pm["season"], pm["week"])
    ]
    aliases: dict[str, str] = {}
    for canonical, part in pm.groupby("canonical_event_id", dropna=False):
        ids = sorted(set(part["event_id"].dropna().astype(str).str.strip()) - {""})
        if len(ids) != 1:
            raise RuntimeError(f"provider event identity ambiguous for canonical game={canonical}: {ids}")
        aliases[str(canonical)] = ids[0]

    raw_team = universe.groupby(["event_id", "team"], as_index=False).agg(
        raw_target_share_sum=("rules_tgt_share", "sum"),
        raw_rush_share_sum=("rules_rush_share", "sum"),
        players=("player_clean_key", "nunique"),
    )
    audit = {
        "disposition": "FOOTBALL_SIMULATION_UNIVERSE_BUILT",
        "source": "PLAYERFORM_OURLADS_PLUS_SCHEDULE",
        "football_player_rows": int(len(universe)),
        "football_players": int(len(form_keys)),
        "football_teams": int(universe["team"].nunique()),
        "canonical_games": int(universe["event_id"].nunique()),
        "model_context_players": int(len(context_keys)),
        "priced_unique_players": int(len(priced_keys)),
        "football_players_without_priced_offers": int(len(form_keys - priced_keys)),
        "priced_players_missing_from_football_universe": 0,
        "sportsbook_rows_used_to_define_player_universe": 0,
        "sportsbook_line_odds_book_fields_present": [],
        "team_wp_present_in_simulation_universe": bool("team_wp" in universe.columns),
        "provider_event_ids_used_during_simulation": False,
        "provider_event_ids_installed_post_simulation_for_lookup_only": True,
        "provider_event_aliases": int(len(aliases)),
        "overlap_player_rows_compared": int(len(compare)),
        "max_abs_football_assumption_diff": max_diffs,
        "raw_target_share_sum_min": float(pd.to_numeric(raw_team["raw_target_share_sum"], errors="coerce").min()),
        "raw_target_share_sum_median": float(pd.to_numeric(raw_team["raw_target_share_sum"], errors="coerce").median()),
        "raw_target_share_sum_max": float(pd.to_numeric(raw_team["raw_target_share_sum"], errors="coerce").max()),
        "raw_rush_share_sum_min": float(pd.to_numeric(raw_team["raw_rush_share_sum"], errors="coerce").min()),
        "raw_rush_share_sum_median": float(pd.to_numeric(raw_team["raw_rush_share_sum"], errors="coerce").median()),
        "raw_rush_share_sum_max": float(pd.to_numeric(raw_team["raw_rush_share_sum"], errors="coerce").max()),
    }
    if audit["team_wp_present_in_simulation_universe"]:
        raise RuntimeError("market-derived team_wp leaked into football simulation universe")

    UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(UNIVERSE_CSV, index=False)
    raw_team.to_csv(AUDIT_CSV, index=False)
    return universe, aliases, audit


def _install_event_aliases(result, aliases: dict[str, str]) -> int:
    additions = {}
    for (game, pkey, market), values in list(result.values.items()):
        provider = aliases.get(str(game))
        if provider:
            additions[(provider, pkey, market)] = values
    result.values.update(additions)
    return len(additions)


def _validate_priced_distribution_coverage(result, metrics: pd.DataFrame) -> tuple[int, list[dict]]:
    unique = metrics.sort_values(["event_id", "team", "player_clean_key", "market"]).drop_duplicates(
        ["event_id", "team", "player_clean_key", "market"], keep="last"
    )
    missing = []
    for _, row in unique.iterrows():
        outcomes = lookup(result, row, str(row.get("market", "")))
        if outcomes is None or len(outcomes) == 0:
            missing.append({
                "event_id": row.get("event_id"), "player": row.get("player"),
                "team": row.get("team"), "market": row.get("market"),
            })
    return int(len(unique)), missing


def _apply_rb_rush_rec_conservation(result, pricing_metrics: pd.DataFrame) -> dict:
    frame = pricing_metrics.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    canonical_market = frame.get("market", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().map(
        lambda value: MARKET_MAP.get(value, value)
    )
    frame["_canonical_market"] = canonical_market
    pos_col = next((c for c in ("position_group", "position", "alignment_position") if c in frame.columns), None)
    if pos_col is None:
        raise RuntimeError("RB conservation requires current position column")
    frame["_position_family"] = frame[pos_col].map(_position_family)
    eligible = frame.loc[
        frame["_canonical_market"].eq("rush_rec_yards")
        & frame["_position_family"].isin({"RB", "FB"})
    ].copy()
    identity_cols = [c for c in ("event_id", "team", "player_clean_key", "player") if c in eligible.columns]
    eligible = eligible.drop_duplicates(identity_cols, keep="first")

    if eligible.empty:
        pd.DataFrame(columns=["player", "team"]).to_csv(RB_AUDIT_CSV, index=False)
        payload = {"disposition": "NO_ELIGIBLE_RB_RUSH_REC_ROWS", "players": 0, "sportsbook_inputs_used": False}
        RB_AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    rb_context = load_rb_context()
    audit_rows = []
    for _, row in eligible.iterrows():
        rush = lookup(result, row, "rush_yards")
        rec = lookup(result, row, "rec_yards")
        combo_before = lookup(result, row, "rush_rec_yards")
        if rush is None or rec is None or combo_before is None:
            raise RuntimeError(f"RB conservation missing simulation component player={row.get('player')} team={row.get('team')}")
        rush = np.asarray(rush, dtype=float)
        rec = np.asarray(rec, dtype=float)
        combo_before = np.asarray(combo_before, dtype=float)
        if len(rush) != len(rec) or len(rush) != len(combo_before):
            raise RuntimeError(f"RB conservation simulation length mismatch player={row.get('player')}")
        meta = lookup_rb_projection(row, rb_context)
        p3_mean = float(meta["rb_synthesis_proj"])
        raw_rush_mean = float(np.mean(rush))
        rec_mean = float(np.mean(rec))
        if raw_rush_mean > 0:
            scaled_rush = rush * (p3_mean / raw_rush_mean)
        elif abs(p3_mean) <= 1e-12:
            scaled_rush = np.zeros_like(rush)
        else:
            raise RuntimeError(f"cannot conserve positive P3 mean from zero rush distribution player={row.get('player')}")
        conserved = scaled_rush + rec
        provider_game = str(row.get("event_id"))
        canonical_game = _canonical_game(row.get("team"), row.get("opponent"), row.get("season"), row.get("week"))
        pkey = _player_key(row)
        result.values[(provider_game, pkey, "rush_rec_yards")] = conserved
        result.values[(canonical_game, pkey, "rush_rec_yards")] = conserved
        conserved_mean = float(np.mean(conserved))
        gap = conserved_mean - (p3_mean + rec_mean)
        audit_rows.append({
            "event_id": provider_game, "canonical_event_id": canonical_game,
            "player": row.get("player"), "team": row.get("team"), "opponent": row.get("opponent"),
            "p3_rush_mean": p3_mean, "raw_mc_rush_mean": raw_rush_mean, "rec_mc_mean": rec_mean,
            "legacy_combo_mean": float(np.mean(combo_before)), "conserved_combo_mean": conserved_mean,
            "conservation_gap": gap, "rb_synthesis_version": meta.get("rb_synthesis_version"),
            "rb_synthesis_route": meta.get("rb_synthesis_route"), "sportsbook_inputs_used": False,
        })

    audit = pd.DataFrame(audit_rows)
    max_gap = float(pd.to_numeric(audit["conservation_gap"], errors="coerce").abs().max())
    if not np.isfinite(max_gap) or max_gap > 1e-8:
        raise RuntimeError(f"RB rush+receiving conservation arithmetic failed max_gap={max_gap}")
    if not audit["rb_synthesis_version"].astype(str).eq("RB_P3_SYNTHESIS_V1").all():
        raise RuntimeError("RB conservation consumed non-promoted synthesis version")
    if not audit["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all():
        raise RuntimeError("RB conservation consumed non-Week1 synthesis route")
    audit.to_csv(RB_AUDIT_CSV, index=False)
    payload = {
        "disposition": "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3",
        "players": int(len(audit)), "max_arithmetic_gap": max_gap,
        "sportsbook_inputs_used": False, "audit": str(RB_AUDIT_CSV),
    }
    RB_AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[rb_rush_rec_conservation] " + json.dumps(payload, sort_keys=True))
    return payload


def _full_roster_simulate(pricing_metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    universe, aliases, audit = _build_full_universe(pricing_metrics)
    result = canonical_simulate(universe, iterations=iterations, seed=seed, allocation_trace=allocation_trace)
    alias_distribution_keys = _install_event_aliases(result, aliases)

    # Every football player must actually receive simulation output.  The MC
    # emits receiving/rushing/combo for every roster player, QB passing for QBs,
    # and ATD whenever the football scoring prior is available.
    sim_player_keys = {(str(team), str(key)) for team, key in zip(universe["team"], universe["player_clean_key"])}
    missing_core = []
    for _, row in universe.iterrows():
        game = str(row["event_id"]); pkey = _player_key(row)
        for market in ("receptions", "rec_yards", "rush_att", "rush_yards", "rush_rec_yards"):
            if (game, pkey, market) not in result.values:
                missing_core.append({"player": row.get("player"), "team": row.get("team"), "market": market})
        if _position_family(row.get("position")) == "QB" and (game, pkey, "pass_yards") not in result.values:
            missing_core.append({"player": row.get("player"), "team": row.get("team"), "market": "pass_yards"})
        if pd.notna(pd.to_numeric(pd.Series([row.get("offensive_td_rate")]), errors="coerce").iloc[0]) and (game, pkey, "anytime_td") not in result.values:
            missing_core.append({"player": row.get("player"), "team": row.get("team"), "market": "anytime_td"})
    if missing_core:
        raise RuntimeError(f"full football roster did not receive complete simulation output: {missing_core[:20]}")

    priced_distribution_keys, missing_priced = _validate_priced_distribution_coverage(result, pricing_metrics)
    if missing_priced:
        raise RuntimeError(f"priced offers missing from pre-generated football distributions: {missing_priced[:20]}")

    rb_payload = _apply_rb_rush_rec_conservation(result, pricing_metrics)
    audit.update({
        "disposition": "FOOTBALL_SIMULATION_UNIVERSE_CERTIFIED",
        "simulation_result_keys_before_provider_alias_accounting": int(len(result.values) - alias_distribution_keys),
        "provider_alias_distribution_keys": int(alias_distribution_keys),
        "priced_unique_player_market_keys_checked": int(priced_distribution_keys),
        "priced_distribution_misses": 0,
        "football_players_with_simulation_output": int(len(sim_player_keys)),
        "rb_conservation_disposition": rb_payload.get("disposition"),
        "sportsbook_inputs_used_to_generate_football_distributions": False,
        "sportsbook_event_identity_used_for_post_simulation_lookup_only": True,
    })
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[football_simulation_universe] " + json.dumps(audit, sort_keys=True))
    return result


def main() -> int:
    pricing.simulate = _full_roster_simulate
    return int(pricing.main())


if __name__ == "__main__":
    raise SystemExit(main())
