#!/usr/bin/env python3
"""Catch a provider name spelling gap BEFORE the paid odds fetch, not after.

2026 Week 9 lost two paid runs to one tight end.

Ourlads lists the Baltimore TE2 as "Matt Hibner". nflverse and every
sportsbook call him "Matthew Hibner". Run 42094118645 bought the Thursday
odds, captured 738 alternate-line offers, and then failed closed in
repair_live_prop_identity because the prop rows named a player the roster
index had under a different spelling. Run 42099277230 bought the full Week 9
slate and failed closed five steps later, in materialize_live_props_for_model,
on the same name against PlayerForm.

Both failures are correct behavior -- an unmappable player on a priced market
is a real identity gap and must not be silently priced. The defect is WHEN it
is discovered. The check costs nothing and needs no sportsbook data, so there
is no reason to pay for odds first and find out afterwards.

data/manual_name_overrides.csv is the designed, auditable remedy and already
carries 25 such rows. This audit finds the ones nobody has noticed yet and
prints the exact line to add.

Scope note: a mismatch is only reported when the provider's spelling and the
nflverse canonical spelling share a surname and first initial ON THE SAME
TEAM, and when exactly one nflverse player matches. Anything ambiguous is left
alone -- this names a candidate for a human-verified override, it never
rewrites an identity by itself.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name
from scripts.utils.player_identity_v3 import player_name_key

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = ROOT / "data" / "manual_name_overrides.csv"
# The positions a sportsbook actually posts yardage/reception props for. A
# spelling gap on a guard cannot kill a board; one on a TE2 just did, twice.
#
# Matched as a SUBSTRING against both the position and the role column,
# because Ourlads encodes alignment rather than position: a wide receiver is
# LWR, RWR or SWR, never "WR". An exact-set membership test silently passed
# every receiver name gap straight through this gate -- caught on Joshua/Josh
# Palmer (BUF) and Mitch/Mitchell Tinsley (CIN), both Week 9 teams. No
# non-skill NFL position code contains any of these as a substring.
PRICED_POSITIONS = ("QB", "RB", "WR", "TE", "FB", "HB")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _initial_last(value) -> str:
    tokens = [
        t for t in (re.sub(r"[^a-z0-9]", "", w) for w in str(value).lower().split())
        if t and t not in SUFFIXES
    ]
    return f"{tokens[0][0]}{tokens[-1]}" if len(tokens) >= 2 else ""


def _load_nflverse(season: int) -> pd.DataFrame | None:
    try:
        import nflreadpy as nfl

        ros = nfl.load_rosters(seasons=[int(season)]).to_pandas()
    except Exception as exc:  # provider/network failure is not an identity defect
        print(f"[name_parity] nflverse rosters unavailable ({type(exc).__name__}: {exc}); "
              f"cannot check provider name parity this run", file=sys.stderr)
        return None
    need = {"team", "full_name"}
    if ros is None or ros.empty or not need.issubset(ros.columns):
        print("[name_parity] nflverse roster artifact unusable; skipping", file=sys.stderr)
        return None
    ros = ros.loc[:, [c for c in ("team", "full_name", "position") if c in ros.columns]].dropna(
        subset=["team", "full_name"]
    ).copy()
    ros["team"] = ros["team"].map(canon_team)
    ros["key"] = ros["full_name"].map(lambda v: player_name_key(v, strip_suffix=True))
    ros["il"] = ros["full_name"].map(_initial_last)
    return ros


def find_mismatches(roles: pd.DataFrame, nflverse: pd.DataFrame) -> pd.DataFrame:
    pos_cols = [c for c in ("position", "pos", "role") if c in roles.columns]
    roles = roles.dropna(subset=["team", "player"]).copy()
    roles["team"] = roles["team"].map(canon_team)
    # Canonicalize first: a name already covered by manual_name_overrides.csv
    # is not a gap, and must not be reported again every week.
    roles["canonical"] = roles["player"].map(canonicalize_player_name)
    roles["key"] = roles["canonical"].map(lambda v: player_name_key(v, strip_suffix=True))
    roles["il"] = roles["canonical"].map(_initial_last)

    known = set(zip(nflverse["team"], nflverse["key"]))
    by_il = nflverse.groupby(["team", "il"])["full_name"].agg(lambda s: sorted(set(s)))

    out = []
    for row in roles.itertuples(index=False):
        data = row._asdict()
        team, key, il = data["team"], data["key"], data["il"]
        if not team or not key or not il or (team, key) in known:
            continue
        candidates = by_il.get((team, il))
        if candidates is None or len(candidates) != 1:
            continue  # absent from nflverse, or ambiguous -- not this audit's call
        labels = [str(data.get(c, "") or "").upper() for c in pos_cols]
        position = "/".join([x for x in labels if x]) or ""
        priced = any(code in label for label in labels for code in PRICED_POSITIONS)
        out.append({
            "team": team,
            "position": position,
            "provider_name": str(data["player"]),
            "nflverse_name": candidates[0],
            "priced_position": bool(priced),
        })
    return pd.DataFrame(
        out, columns=["team", "position", "provider_name", "nflverse_name", "priced_position"]
    ).drop_duplicates()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=int(os.getenv("SEASON", "2026")))
    ap.add_argument("--roles", type=Path, default=None,
                    help="roles CSV to check; defaults to ACTIVE_ROLES_CSV then data/roles_ourlads.csv")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when a priced-position name gap is found")
    a = ap.parse_args()

    path = a.roles or Path(
        os.getenv("ACTIVE_ROLES_CSV") or os.getenv("ROLES_CSV") or "data/roles_ourlads.csv"
    )
    if not path.exists() or not path.stat().st_size:
        print(f"[name_parity] roles artifact absent at {path}; nothing to check", file=sys.stderr)
        return 0
    roles = pd.read_csv(path, low_memory=False)
    if roles.empty or not {"team", "player"}.issubset(roles.columns):
        print(f"[name_parity] roles artifact at {path} unusable; nothing to check", file=sys.stderr)
        return 0

    nflverse = _load_nflverse(a.season)
    if nflverse is None:
        return 0

    found = find_mismatches(roles, nflverse)
    priced = found.loc[found["priced_position"]] if not found.empty else found

    print(f"[name_parity] roles={path} rows={len(roles)} season={a.season} "
          f"gaps={len(found)} priced_position_gaps={len(priced)}")
    if found.empty:
        print("[name_parity] provider names agree with nflverse canonical names")
        return 0

    for row in found.itertuples(index=False):
        flag = "PRICED" if row.priced_position else "other"
        print(f"  [{flag}] {row.team} {row.position or '??'}: "
              f"provider={row.provider_name!r} nflverse={row.nflverse_name!r}")

    if not priced.empty:
        print("\nAdd each of these to data/manual_name_overrides.csv after verifying it, "
              "then re-run. Suggested lines (fill in your own source and date):")
        for row in priced.itertuples(index=False):
            print(f'  {row.provider_name},{row.nflverse_name},'
                  f'"Provider lists the short/abbreviated form; nflverse {a.season} rosters give '
                  f'full_name {row.nflverse_name} ({row.team}, {row.position}).",'
                  f'nflreadpy load_rosters({a.season}),<YYYY-MM-DD>')

    if a.strict and not priced.empty:
        print(
            f"\n[name_parity] FATAL: {len(priced)} priced-position name gap(s). A sportsbook "
            f"posts the nflverse spelling, so these rows would fail the identity gate AFTER "
            f"the odds are paid for -- which is exactly what cost 2026 Week 9 two paid runs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
