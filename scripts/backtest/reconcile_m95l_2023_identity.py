"""M95L-only historical identity bridge for the sealed 2023 confirmation.

The 2023 nflverse weekly-roster source and weekly player-stat source use different
name aliases for five RBs that are present in the frozen M95B trace. M95L Run #2
therefore lost 20 otherwise-valid player/week rows before any sealed metrics were
computed.

This script repairs only identity representation. It uses stable nflverse/GSIS
player IDs plus exact target-week roster snapshots; it does not read target-week
outcomes, carry totals, M95L tail probabilities, sportsbook data, or confirmation
metrics. The frozen M95B spelling is used as the common M95L key so historical
player logs, pregame universes, and exact-week injury names resolve consistently.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team


# Verified by the source-only M95L identity diagnostic (run 33426776411).
# Stable GSIS identity is the authority; names are aliases only.
IDENTITY = {
    "00-0034115": {"display": "Jeff Wilson", "key": "jeffwilson"},
    "00-0036919": {"display": "Kenny Gainwell", "key": "kennygainwell"},
    "00-0038611": {"display": "Chris Rodriguez Jr", "key": "chrisrodriguezjr"},
    "00-0038134": {"display": "Kenneth Walker III", "key": "kennethwalkeriii"},
    "00-0038685": {"display": "Chris Brooks", "key": "chrisbrooks"},
}


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    return pd.DataFrame(value)


def load_roster_identity() -> pd.DataFrame:
    import nflreadpy as nfl

    raw = to_pandas(nfl.load_rosters_weekly(2023))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {"season", "week", "team", "gsis_id"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"M95L roster identity source missing {sorted(required - set(raw.columns))}")
    name_col = "full_name" if "full_name" in raw.columns else "football_name" if "football_name" in raw.columns else None
    if not name_col:
        raise RuntimeError("M95L roster identity source missing full_name/football_name")

    x = raw.loc[pd.to_numeric(raw["season"], errors="coerce").eq(2023)].copy()
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["provider_name"] = x[name_col].astype(str).str.strip()
    x["provider_key"] = x["provider_name"].map(norm)
    x["gsis_id"] = x["gsis_id"].astype(str).str.strip()
    x = x.loc[x["gsis_id"].isin(IDENTITY)].copy()
    if x.empty:
        raise RuntimeError("M95L stable-ID roster bridge returned zero rows")

    # Exact week/team/provider-name must resolve to one stable player ID.
    dup = x.duplicated(["week", "team", "provider_key"], keep=False)
    if dup.any():
        bad = x.loc[dup, ["week", "team", "provider_name", "gsis_id"]]
        raise RuntimeError(f"M95L ambiguous roster identity rows: {bad.to_dict('records')[:10]}")
    return x[["week", "team", "provider_name", "provider_key", "gsis_id"]].drop_duplicates()


def reconcile_player_logs(path: Path, audit_rows: list[dict]) -> None:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "player_id" not in x.columns or "player_clean_key" not in x.columns or "player" not in x.columns:
        raise RuntimeError("M95L player logs lack player_id/player/player_clean_key")
    x["player_id"] = x["player_id"].astype(str).str.strip()

    for gsis, info in IDENTITY.items():
        mask = x["player_id"].eq(gsis)
        before_names = "|".join(sorted(set(x.loc[mask, "player"].dropna().astype(str))))
        before_keys = "|".join(sorted(set(x.loc[mask, "player_clean_key"].dropna().astype(str))))
        x.loc[mask, "player"] = info["display"]
        x.loc[mask, "player_clean_key"] = info["key"]
        audit_rows.append({
            "surface": "player_logs", "gsis_id": gsis, "rows_changed": int(mask.sum()),
            "before_names": before_names, "before_keys": before_keys,
            "after_name": info["display"], "after_key": info["key"],
        })

    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys + ["player_id"]].head(20)
        raise RuntimeError(f"M95L identity bridge created duplicate player-log keys: {bad.to_dict('records')}")
    x.to_csv(path, index=False)


def reconcile_universes(universe_dir: Path, roster: pd.DataFrame, audit_rows: list[dict]) -> None:
    lookup = {
        (int(r.week), str(r.team), str(r.provider_key)): str(r.gsis_id)
        for r in roster.itertuples(index=False)
    }
    changed_by_id = {gsis: 0 for gsis in IDENTITY}

    for path in sorted(universe_dir.glob("2023_week_*.csv")):
        x = pd.read_csv(path, low_memory=False)
        x.columns = [str(c).strip().lower() for c in x.columns]
        if not {"player", "team", "week"}.issubset(x.columns):
            raise RuntimeError(f"M95L pregame universe schema incomplete: {path}")
        week = int(pd.to_numeric(x["week"], errors="coerce").dropna().iloc[0])
        teams = x["team"].map(canon_team)
        pkeys = x["player"].map(norm)

        for idx in x.index:
            gsis = lookup.get((week, str(teams.loc[idx]), str(pkeys.loc[idx])))
            if gsis not in IDENTITY:
                continue
            x.at[idx, "player"] = IDENTITY[gsis]["display"]
            if "player_clean_key" in x.columns:
                x.at[idx, "player_clean_key"] = IDENTITY[gsis]["key"]
            changed_by_id[gsis] += 1

        # The identity rewrite must not collapse two distinct pregame rows.
        check_key = x["player"].map(norm)
        if pd.DataFrame({"team": x["team"].map(canon_team), "key": check_key}).duplicated().any():
            raise RuntimeError(f"M95L identity bridge created duplicate pregame players in {path.name}")
        x.to_csv(path, index=False)

    for gsis, count in changed_by_id.items():
        audit_rows.append({
            "surface": "pregame_universe", "gsis_id": gsis, "rows_changed": int(count),
            "before_names": "source_weekly_roster", "before_keys": "source_weekly_roster",
            "after_name": IDENTITY[gsis]["display"], "after_key": IDENTITY[gsis]["key"],
        })
        if count == 0:
            raise RuntimeError(f"M95L pregame identity bridge changed zero rows for {gsis}")


def reconcile_injuries(path: Path, roster: pd.DataFrame, audit_rows: list[dict]) -> None:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if not {"player", "team", "season", "week"}.issubset(x.columns):
        raise RuntimeError("M95L injury history schema incomplete")

    alias_by_team_key: dict[tuple[str, str], str] = {}
    for r in roster.itertuples(index=False):
        alias_by_team_key[(str(r.team), str(r.provider_key))] = str(r.gsis_id)
    # Also accept the frozen spelling in case the injury source already uses it.
    teams_by_id = roster.groupby("gsis_id")["team"].unique().to_dict()
    for gsis, info in IDENTITY.items():
        for team in teams_by_id.get(gsis, []):
            alias_by_team_key[(str(team), norm(info["display"]))] = gsis

    counts = {gsis: 0 for gsis in IDENTITY}
    teams = x["team"].map(canon_team)
    pkeys = x["player"].map(norm)
    for idx in x.index:
        gsis = alias_by_team_key.get((str(teams.loc[idx]), str(pkeys.loc[idx])))
        if gsis not in IDENTITY:
            continue
        x.at[idx, "player"] = IDENTITY[gsis]["display"]
        counts[gsis] += 1
    x.to_csv(path, index=False)

    for gsis, count in counts.items():
        audit_rows.append({
            "surface": "injuries", "gsis_id": gsis, "rows_changed": int(count),
            "before_names": "pregame_injury_alias", "before_keys": "pregame_injury_alias",
            "after_name": IDENTITY[gsis]["display"], "after_key": IDENTITY[gsis]["key"],
        })


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    root = args.root
    player_logs = root / "player_game_logs_history.csv"
    injuries = root / "injuries_history.csv"
    universe_dir = root / "pregame_universe"
    for path in (player_logs, injuries):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"M95L identity bridge missing required file: {path}")
    if not universe_dir.exists():
        raise RuntimeError(f"M95L identity bridge missing universe directory: {universe_dir}")

    roster = load_roster_identity()
    audit_rows: list[dict] = []
    reconcile_player_logs(player_logs, audit_rows)
    reconcile_universes(universe_dir, roster, audit_rows)
    reconcile_injuries(injuries, roster, audit_rows)

    audit = pd.DataFrame(audit_rows)
    audit_path = root / "m95l_identity_bridge_audit.csv"
    audit.to_csv(audit_path, index=False)
    print("[m95l_identity] stable-ID identity bridge applied")
    print(audit.to_string(index=False))
    print(f"[m95l_identity] audit={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
