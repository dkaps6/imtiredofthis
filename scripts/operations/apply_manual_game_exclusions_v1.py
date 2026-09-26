#!/usr/bin/env python3
"""Withhold a verified game from the production-eligible slate.

Sometimes one game's football inputs are internally inconsistent while the rest
of the slate is sound. Pricing the whole slate anyway is wrong, and discarding
the whole slate costs every other game for one bad one. This is the narrow,
auditable middle: name the game, say why, and let the certified slate shrink by
exactly that game.

It runs between the timing certification and its consumers, so everything
downstream -- eligible roles, PlayerForm, the model bridges, the simulation
universe and the board -- is built from the reduced slate consistently rather
than patched after the fact. The board's own universe audit then reports
`certified_slate_teams` and `full_league_slate`, so an excluded game is visible
on the artifact rather than implied by a missing row.

Every row must carry a reason, a verified source and a date. This is not a way
to make a red run green: it records a decision that a game's inputs could not be
trusted, which is a different claim from "the gate was inconvenient".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

DATA = Path("data")
CERTIFICATION = DATA / "current_player_availability_game_certification.csv"
EXCLUSIONS = DATA / "manual_game_exclusions.csv"
AUDIT = DATA / "manual_game_exclusion_audit.json"
REQUIRED = ["season", "week", "team", "reason", "verified_source", "verified_date"]


def _blank(v) -> bool:
    if v is None or pd.isna(v):
        return True
    return str(v).strip().lower() in {"", "nan", "none", "null", "<na>"}


def _load_exclusions() -> pd.DataFrame:
    if not EXCLUSIONS.exists() or not EXCLUSIONS.stat().st_size:
        return pd.DataFrame(columns=REQUIRED)
    df = pd.read_csv(EXCLUSIONS)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise RuntimeError(f"manual game exclusions missing columns: {sorted(missing)}")
    if not isinstance(df.index, pd.RangeIndex):
        raise RuntimeError(
            "manual game exclusions parsed with a non-default index; usually an unquoted "
            "comma in a field shifted a row"
        )
    for row in df.itertuples(index=False):
        for col in REQUIRED:
            if _blank(getattr(row, col, None)):
                raise RuntimeError(
                    f"manual game exclusion row is missing {col!r}; every exclusion must "
                    f"record what was excluded, why, and how it was verified: {row}"
                )
    return df


def apply(*, season: int, week: int) -> dict:
    if not CERTIFICATION.exists() or not CERTIFICATION.stat().st_size:
        raise RuntimeError(f"game certification missing/empty: {CERTIFICATION}")
    cert = pd.read_csv(CERTIFICATION)
    need = {"season", "week", "away_team", "home_team", "production_eligible"}
    missing = need - set(cert.columns)
    if missing:
        raise RuntimeError(f"game certification missing columns: {sorted(missing)}")

    excl = _load_exclusions()
    excl = excl.loc[
        pd.to_numeric(excl.get("season"), errors="coerce").eq(int(season))
        & pd.to_numeric(excl.get("week"), errors="coerce").eq(int(week))
    ] if len(excl) else excl

    eligible_before = int(pd.to_numeric(cert["production_eligible"], errors="coerce").fillna(0).eq(1).sum())
    applied: list[dict] = []
    if len(excl):
        away = cert["away_team"].map(canon_team).astype(str)
        home = cert["home_team"].map(canon_team).astype(str)
        for row in excl.itertuples(index=False):
            team = str(canon_team(row.team))
            hit = away.eq(team) | home.eq(team)
            if not hit.any():
                raise RuntimeError(
                    f"manual game exclusion names {team} but no {season} week {week} game "
                    "carries that team; a stale exclusion must be corrected, not ignored"
                )
            already = pd.to_numeric(cert.loc[hit, "production_eligible"], errors="coerce").fillna(0).eq(0).all()
            cert.loc[hit, "production_eligible"] = False
            if "failure_reason" in cert.columns:
                # An all-empty failure_reason round-trips through CSV as float64,
                # so the column has to be text before a reason can be written.
                cert["failure_reason"] = (
                    cert["failure_reason"].astype("object").where(cert["failure_reason"].notna(), "").astype(str)
                )
                cert.loc[cert["failure_reason"].isin(("nan", "None")), "failure_reason"] = ""
                tag = f"manual_exclusion:{team}"
                cert.loc[hit, "failure_reason"] = [
                    tag if not v else (v if tag in v else f"{v}|{tag}")
                    for v in cert.loc[hit, "failure_reason"]
                ]
            applied.append({
                "team": team,
                "games": int(hit.sum()),
                "already_ineligible": bool(already),
                "reason": str(row.reason),
                "verified_source": str(row.verified_source),
                "verified_date": str(row.verified_date),
            })

    eligible_after = int(pd.to_numeric(cert["production_eligible"], errors="coerce").fillna(0).eq(1).sum())
    if eligible_after <= 0:
        raise RuntimeError(
            "manual game exclusions would leave zero production-eligible games; "
            "that is not a partial slate, it is no slate"
        )
    # Only rewrite when something was actually excluded. A "no-op" that still
    # rewrites the file is not a no-op: it reserializes booleans and empty
    # fields, and a downstream consumer that parses the certification can then
    # behave differently for a run that changed nothing. Observed on 2026 Week
    # 3 run 36278356889, where an empty exclusions file still rewrote the
    # certification and broke a coverage seam that had passed moments before.
    if applied:
        cert.to_csv(CERTIFICATION, index=False)

    el = pd.to_numeric(cert["production_eligible"], errors="coerce").fillna(0).eq(1)
    teams = sorted({
        str(canon_team(t))
        for t in [*cert.loc[el, "away_team"], *cert.loc[el, "home_team"]]
        if str(t).strip()
    })
    result = {
        "disposition": "MANUAL_GAME_EXCLUSIONS_APPLIED" if applied else "NO_MANUAL_GAME_EXCLUSIONS",
        "season": int(season),
        "week": int(week),
        "eligible_games_before": eligible_before,
        "eligible_games_after": eligible_after,
        "eligible_teams_after": len(teams),
        "full_league_slate": bool(len(teams) == 32),
        "exclusions_applied": applied,
        "sportsbook_inputs_used": 0,
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[game_exclusions] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    a = ap.parse_args()
    apply(season=a.season, week=a.week)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
