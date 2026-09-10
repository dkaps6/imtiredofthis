#!/usr/bin/env python3
"""Withhold timing-ineligible games from reconciled active roles before opportunity.

Consumes only football availability/timing artifacts. No sportsbook input is read.
The unfiltered reconciled active-role artifact remains available for audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team


def build(active: pd.DataFrame, certification: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = active.copy()
    a.columns = [str(c).strip().lower() for c in a.columns]
    c = certification.copy()
    c.columns = [str(x).strip().lower() for x in c.columns]
    need_a = {"team", "player", "role", "position", "player_clean_key"}
    need_c = {"away_team", "home_team", "production_eligible", "certification_state"}
    if need_a - set(a.columns):
        raise RuntimeError(f"active roles missing {sorted(need_a-set(a.columns))}")
    if need_c - set(c.columns):
        raise RuntimeError(f"timing certification missing {sorted(need_c-set(c.columns))}")
    a["team"] = a.team.map(canon_team)
    c["away_team"] = c.away_team.map(canon_team)
    c["home_team"] = c.home_team.map(canon_team)
    eligible = pd.to_numeric(c.production_eligible, errors="coerce").fillna(0).eq(1)
    eligible_teams = set(c.loc[eligible, "away_team"]) | set(c.loc[eligible, "home_team"])
    withheld_teams = (set(c.away_team) | set(c.home_team)) - eligible_teams
    # A team may appear only once in a weekly slate. Reject malformed certification
    # rather than allowing one eligible duplicate game to mask a withheld row.
    team_rows = pd.concat([c[["away_team"]].rename(columns={"away_team":"team"}), c[["home_team"]].rename(columns={"home_team":"team"})])
    if team_rows.team.duplicated().any():
        raise RuntimeError("timing certification is not one game per team")
    out = a[a.team.isin(eligible_teams)].copy().reset_index(drop=True)
    if set(out.team) & withheld_teams:
        raise RuntimeError("withheld team survived production-eligible active-role filter")
    if out.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("duplicate production-eligible active role identity")
    states = c.certification_state.astype(str).value_counts().to_dict()
    meta = {
        "input_active_rows": int(len(a)),
        "output_active_rows": int(len(out)),
        "eligible_games": int(eligible.sum()),
        "withheld_games": int((~eligible).sum()),
        "eligible_teams": sorted(eligible_teams),
        "withheld_teams": sorted(withheld_teams),
        "certification_state_counts": {str(k): int(v) for k, v in states.items()},
        "sportsbook_inputs_used": 0,
        "production_candidate_only": True,
    }
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", type=Path, default=Path("data/roles_ourlads_active_v1.csv"))
    ap.add_argument("--certification", type=Path, default=Path("data/current_player_availability_game_certification.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/roles_current_production_eligible_v1.csv"))
    ap.add_argument("--status", type=Path, default=Path("data/roles_current_production_eligible_v1_status.json"))
    args = ap.parse_args()
    active = pd.read_csv(args.active)
    cert = pd.read_csv(args.certification)
    out, meta = build(active, cert)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.status.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
