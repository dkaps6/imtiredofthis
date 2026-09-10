#!/usr/bin/env python3
"""Exercise the real promoted football stack from an immutable availability candidate.

No sportsbook input is read or required. The input frame is built only from the
already-materialized PlayerForm/current-role football universe. This is integration
certification plumbing for the frozen 35-gate availability plan; it does not alter
scientific parameters or authorize production by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.run_pricing_with_full_roster_universe_v1 import _canonical_game
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
import scripts.run_pricing_with_full_roster_universe_v5_production as v5

DATA = Path("data")
OUT = DATA / "current_availability_football_stack_certification.json"


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
    x = form.copy()
    x["team"] = x.team.map(canon_team)
    x["opponent"] = x.opponent.map(canon_team)
    x["event_id"] = [
        _canonical_game(t, o, s, w)
        for t, o, s, w in zip(x.team, x.opponent, x.season, x.week)
    ]
    # These rows are lookup scaffolding only. They contain no book, line, odds or
    # market probability and therefore cannot define the football universe.
    x["market"] = "football_certification_lookup"
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
    final, aliases, build_audit = v3._build_with_promoted_entitlement_specialists(lookup)
    if set(final.team.map(canon_team).astype(str)) != eligible_teams:
        raise RuntimeError("promoted entitlement universe changed certified team set")
    result = v5._simulate_v5(final, iterations=4000, seed=20260910)

    r22_path = DATA / "rb_receiving_tail_production_audit.json"
    r26_path = DATA / "rb_r26_receptions_production_audit.json"
    ent_path = DATA / "target_entitlement_v1_audit.json"
    qb_path = DATA / "qb_c2_production_integration_audit.json"
    for p in [r22_path, r26_path, ent_path, qb_path]:
        if not p.is_file():
            raise RuntimeError(f"promoted football stack did not emit required audit: {p}")
    r22 = json.loads(r22_path.read_text(encoding="utf-8"))
    r26 = json.loads(r26_path.read_text(encoding="utf-8"))
    ent = json.loads(ent_path.read_text(encoding="utf-8"))
    qb = json.loads(qb_path.read_text(encoding="utf-8"))

    # Every unavailable player must remain absent from all generated simulation keys.
    sim_players = {str(k[1]) for k in result.values}
    unavailable_in_sim = sorted({k for _, k in bad_keys if k in sim_players})
    if unavailable_in_sim:
        raise RuntimeError(f"unavailable players received simulation arrays: {unavailable_in_sim}")

    position = final.get("position_family", final.get("position", pd.Series("", index=final.index))).fillna("").astype(str).str.upper()
    rb_rows = final.loc[position.isin(["RB", "FB", "HB", "TB"])]
    qb_rows = final.loc[position.eq("QB")]
    withheld = cert.loc[~pd.to_numeric(cert.production_eligible, errors="coerce").fillna(0).eq(1)]

    payload = {
        "disposition": "CURRENT_PLAYER_AVAILABILITY_FOOTBALL_STACK_EXECUTION_PASS",
        "sportsbook_inputs_used": 0,
        "source_candidate_artifact_id": 10140425929,
        "source_candidate_digest": "sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37",
        "eligible_teams": len(eligible_teams),
        "eligible_games": int(pd.to_numeric(cert.production_eligible, errors="coerce").fillna(0).eq(1).sum()),
        "withheld_games": int(len(withheld)),
        "football_players": int(final.player_clean_key.nunique()),
        "football_rows": int(len(final)),
        "rb_fb_rows": int(len(rb_rows)),
        "qb_rows": int(len(qb_rows)),
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
        "provider_event_aliases": int(len(aliases)),
        "build_audit_source": build_audit.get("source"),
    }
    if not payload["te_r5p_pool_preserved"]:
        raise RuntimeError("TE-R5P room conservation failed")
    for key in ["wr_r15_wr1_anchor_preserved", "wr_r15_wr2plus_pool_preserved", "wr_r15_wr_room_mass_preserved", "wr_r15_non_wr_entitlement_preserved"]:
        if not payload[key]:
            raise RuntimeError(f"WR-R15 conservation failed: {key}")
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
