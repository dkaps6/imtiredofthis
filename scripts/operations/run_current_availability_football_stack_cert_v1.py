#!/usr/bin/env python3
"""Exercise the real promoted football stack from an immutable availability candidate.

No sportsbook input is read or required. Synthetic lookup rows contain only
football identity plus canonical market labels; they carry no line, odds, book,
market probability, or sportsbook event identity. The production full-roster
football wrapper therefore exercises M38 -> TE-R5P -> WR-R15 -> QB C2 -> R22 ->
R26 plus the outer promoted P3 rush/receiving conservation seam.

This is integration certification plumbing for the frozen 35-gate availability
plan; it does not alter scientific parameters or authorize production by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
import scripts.run_pricing_with_full_roster_universe_v5_production as v5

DATA = Path("data")
OUT = DATA / "current_availability_football_stack_certification.json"
CORE_MARKETS = ("receptions", "rec_yards", "rush_att", "rush_yards", "rush_rec_yards")


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required certification input missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"required certification input has zero rows: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def build_lookup_metrics(form: pd.DataFrame) -> pd.DataFrame:
    need = {"player", "player_clean_key", "team", "opponent", "season", "week", "position"}
    miss = need - set(form.columns)
    if miss:
        raise RuntimeError(f"PlayerForm consensus missing certification columns: {sorted(miss)}")
    rows = []
    for r in form.itertuples(index=False):
        team = canon_team(getattr(r, "team")); opp = canon_team(getattr(r, "opponent"))
        event = base._canonical_game(team, opp, getattr(r, "season"), getattr(r, "week"))
        rec = r._asdict()
        rec["team"] = team; rec["opponent"] = opp; rec["event_id"] = event
        pos = str(getattr(r, "position", "")).upper().strip()
        markets = list(CORE_MARKETS) + (["pass_yards"] if pos.startswith("QB") else [])
        for market in markets:
            q = dict(rec); q["market"] = market; rows.append(q)
    x = pd.DataFrame(rows)
    forbidden = {"line", "source_line", "over_odds", "under_odds", "book", "book_title", "vegas_line", "vegas_odds", "market_prob", "edge_pct", "edge_abs", "team_wp", "home_wp", "away_wp"}
    leaked = sorted(forbidden & set(x.columns))
    if leaked:
        raise RuntimeError(f"sportsbook field leaked into certification lookup frame: {leaked}")
    return x


def main() -> int:
    form = _read(DATA / "player_form_consensus.csv")
    active = _read(DATA / "roles_current_production_eligible_v1.csv")
    avail = _read(DATA / "current_player_availability.csv")
    cert = _read(DATA / "current_player_availability_game_certification.csv")

    eligible_teams = set(active.team.map(canon_team).dropna().astype(str))
    if len(eligible_teams) != 30:
        raise RuntimeError(f"frozen candidate expected 30 eligible teams, got {len(eligible_teams)}")
    form_teams = set(form.team.map(canon_team).dropna().astype(str))
    if form_teams != eligible_teams:
        raise RuntimeError(f"PlayerForm team set != certified active-role set missing={sorted(eligible_teams-form_teams)} extra={sorted(form_teams-eligible_teams)}")

    unavailable = avail.loc[pd.to_numeric(avail.definitive_unavailable, errors="coerce").fillna(0).eq(1)].copy()
    bad_keys = set(zip(unavailable.team.map(canon_team).astype(str), unavailable.player_clean_key.astype(str)))
    form_keys = set(zip(form.team.map(canon_team).astype(str), form.player_clean_key.astype(str)))
    if bad_keys & form_keys:
        raise RuntimeError(f"definitive unavailable player survived football universe: {sorted(bad_keys & form_keys)}")

    lookup = build_lookup_metrics(form)
    # Mirror the promoted public V5 wiring, but stop before sportsbook pricing.
    base._identity_frame = v2._canonical_identity_frame
    base._validate_priced_distribution_coverage = v2._install_provider_player_aliases_and_validate
    base._build_full_universe = v3._build_with_promoted_entitlement_specialists
    base.canonical_simulate = v5._simulate_v5
    result = base._full_roster_simulate(lookup, iterations=4000, seed=20260910)

    required_json = {
        "r22": DATA / "rb_receiving_tail_production_audit.json",
        "r26": DATA / "rb_r26_receptions_production_audit.json",
        "ent": DATA / "target_entitlement_v1_audit.json",
        "qb": DATA / "qb_c2_production_integration_audit.json",
        "universe": DATA / "football_simulation_universe_audit.json",
        "p3": DATA / "rb_rush_rec_conservation_input_audit.json",
    }
    payloads = {}
    for name, path in required_json.items():
        if not path.is_file():
            raise RuntimeError(f"promoted football stack did not emit required audit: {path}")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    r22, r26, ent, qb, universe, p3 = (payloads[k] for k in ["r22", "r26", "ent", "qb", "universe", "p3"])

    sim_players = {str(k[1]) for k in result.values}
    unavailable_in_sim = sorted({k for _, k in bad_keys if k in sim_players})
    if unavailable_in_sim:
        raise RuntimeError(f"unavailable players received simulation arrays: {unavailable_in_sim}")

    withheld = cert.loc[~pd.to_numeric(cert.production_eligible, errors="coerce").fillna(0).eq(1)]
    football_teams = int(universe.get("football_teams", 0))
    if football_teams != 30:
        raise RuntimeError(f"football wrapper did not preserve 30-team eligible universe: {football_teams}")
    payload = {
        "disposition": "CURRENT_PLAYER_AVAILABILITY_FOOTBALL_STACK_EXECUTION_PASS",
        "sportsbook_inputs_used": 0,
        "synthetic_lookup_rows_only": True,
        "synthetic_lookup_has_book_line_odds": False,
        "source_candidate_artifact_id": 10140425929,
        "source_candidate_digest": "sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37",
        "eligible_teams": len(eligible_teams),
        "eligible_games": int(pd.to_numeric(cert.production_eligible, errors="coerce").fillna(0).eq(1).sum()),
        "withheld_games": int(len(withheld)),
        "football_players": int(universe.get("football_players", 0)),
        "football_teams": football_teams,
        "canonical_games": int(universe.get("canonical_games", 0)),
        "simulation_keys": int(len(result.values)),
        "unavailable_rows_in_audit": int(len(unavailable)),
        "unavailable_players_in_football_universe": 0,
        "unavailable_players_with_simulation_arrays": 0,
        "m38_explicit_entitlement_materialized": bool(ent.get("explicit_target_entitlement_materialized", False) or ent.get("disposition") == "EXPLICIT_TARGET_ENTITLEMENT_MATERIALIZED"),
        "te_r5p_pool_preserved": bool(ent.get("te_r5p_team_pool_preserved", False)),
        "wr_r15_wr1_anchor_preserved": bool(ent.get("wr_r15_m38_wr1_anchor_preserved", False)),
        "wr_r15_wr2plus_pool_preserved": bool(ent.get("wr_r15_wr2plus_pool_preserved", False)),
        "wr_r15_wr_room_mass_preserved": bool(ent.get("wr_r15_wr_room_mass_preserved", False)),
        "wr_r15_non_wr_entitlement_preserved": bool(ent.get("wr_r15_non_wr_entitlement_preserved", False)),
        "qb_c2_disposition": qb.get("disposition"),
        "r22_disposition": r22.get("disposition"),
        "r22_max_mean_delta": r22.get("max_mean_delta"),
        "r26_disposition": r26.get("disposition"),
        "r26_sportsbook_inputs": r26.get("sportsbook_inputs_used", r26.get("sportsbook_inputs_to_r26_football", 0)),
        "p3_conservation_disposition": p3.get("disposition"),
        "p3_max_arithmetic_gap": p3.get("max_arithmetic_gap"),
        "priced_distribution_misses": int(universe.get("priced_distribution_misses", -1)),
    }
    for key in ["te_r5p_pool_preserved", "wr_r15_wr1_anchor_preserved", "wr_r15_wr2plus_pool_preserved", "wr_r15_wr_room_mass_preserved", "wr_r15_non_wr_entitlement_preserved"]:
        if not payload[key]:
            raise RuntimeError(f"promoted entitlement conservation failed: {key}")
    if payload["p3_conservation_disposition"] != "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3":
        raise RuntimeError(f"P3 outer conservation seam failed: {p3}")
    if int(payload["priced_distribution_misses"]) != 0:
        raise RuntimeError("football-only certification lookup missed generated distributions")
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
