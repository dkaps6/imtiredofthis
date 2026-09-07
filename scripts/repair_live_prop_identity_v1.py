#!/usr/bin/env python3
"""Repair and validate live sportsbook player -> team/opponent identity.

This is a deterministic production boundary adapter. It does not change lines,
odds, projections, or any football feature. It only reconciles provider naming
variants against the event-scoped Ourlads roster and fails closed when a core
player market cannot be assigned to exactly one team in that event.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scripts._opponent_map import CANON_TEAM_CODES, canon_team

DATA = Path("data")
OUTPUTS = Path("outputs")
ROLES = DATA / "roles_ourlads.csv"
CORE_MARKETS = {
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}
AUDIT_CSV = DATA / "live_prop_identity_audit.csv"
STATUS_JSON = DATA / "live_prop_identity_status.json"


def _read(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"required live-prop identity artifact missing/empty: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if required and df.empty:
        raise RuntimeError(f"required live-prop identity artifact has zero rows: {path}")
    return df


def _missing_text(value) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def _name_keys(value) -> set[str]:
    """Return exact and provider-tolerant first/last matching keys.

    Ourlads intentionally drops suffixes/middle surname tokens in some depth-chart
    names (for example Marvin Harrison Jr. -> Marvin Harrison and Amon-Ra St.
    Brown -> Amon-Ra Brown). We therefore keep an exact normalized key plus a
    relaxed first/last key, and only accept a relaxed match when it identifies
    exactly one of the two teams in the player's event.
    """
    if _missing_text(value):
        return set()
    raw = str(value).strip().lower().replace("’", "'")
    tokens: list[str] = []
    for token in re.split(r"\s+", raw):
        clean = re.sub(r"[^a-z0-9-]", "", token).replace("-", "")
        if clean:
            tokens.append(clean)
    while tokens and tokens[-1] in SUFFIXES:
        tokens.pop()
    if not tokens:
        return set()
    keys = {"".join(tokens)}
    if len(tokens) >= 2:
        keys.add(tokens[0] + tokens[-1])
    return {k for k in keys if k}


def _build_roster_index(roles: pd.DataFrame) -> dict[str, set[str]]:
    required = {"team", "player"}
    missing = required - set(roles.columns)
    if missing:
        raise RuntimeError(f"roles_ourlads missing live-prop identity columns: {sorted(missing)}")
    roster: dict[str, set[str]] = {}
    for row in roles.itertuples(index=False):
        team = canon_team(getattr(row, "team", ""))
        if team not in CANON_TEAM_CODES:
            continue
        roster.setdefault(team, set()).update(_name_keys(getattr(row, "player", "")))
    if set(roster) != set(CANON_TEAM_CODES):
        missing_teams = sorted(set(CANON_TEAM_CODES) - set(roster))
        raise RuntimeError(f"Ourlads roster identity index missing teams: {missing_teams}")
    return roster


def _event_map(enriched: pd.DataFrame, raw_data: pd.DataFrame) -> dict[str, tuple[str, str]]:
    candidates: list[pd.DataFrame] = []
    if not enriched.empty and {"event_id", "home_team_abbr", "away_team_abbr"}.issubset(enriched.columns):
        candidates.append(enriched[["event_id", "home_team_abbr", "away_team_abbr"]])
    if {"event_id", "home_team_abbr", "away_team_abbr"}.issubset(raw_data.columns):
        candidates.append(raw_data[["event_id", "home_team_abbr", "away_team_abbr"]])
    if not candidates:
        raise RuntimeError("No event home/away identity available for live props")
    x = pd.concat(candidates, ignore_index=True).dropna(subset=["event_id"]).drop_duplicates()
    out: dict[str, tuple[str, str]] = {}
    for row in x.itertuples(index=False):
        event_id = str(getattr(row, "event_id"))
        home = canon_team(getattr(row, "home_team_abbr", ""))
        away = canon_team(getattr(row, "away_team_abbr", ""))
        if home not in CANON_TEAM_CODES or away not in CANON_TEAM_CODES or home == away:
            continue
        pair = (home, away)
        old = out.get(event_id)
        if old is not None and old != pair:
            raise RuntimeError(f"Conflicting home/away mapping for event_id={event_id}: {old} vs {pair}")
        out[event_id] = pair
    if not out:
        raise RuntimeError("Event identity map contains zero valid NFL events")
    return out


def _infer_team(player, event_id, roster: dict[str, set[str]], events: dict[str, tuple[str, str]]) -> str:
    pair = events.get(str(event_id))
    if pair is None:
        return ""
    keys = _name_keys(player)
    if not keys:
        return ""
    matches = [team for team in pair if keys & roster.get(team, set())]
    matches = list(dict.fromkeys(matches))
    return matches[0] if len(matches) == 1 else ""


def _opp(team: str, event_id, events: dict[str, tuple[str, str]]) -> str:
    pair = events.get(str(event_id))
    if not pair or team not in pair:
        return ""
    return pair[1] if pair[0] == team else pair[0]


def _player_col(df: pd.DataFrame) -> str:
    for col in ("canonical_player_name", "player_canonical", "player", "player_name_raw"):
        if col in df.columns:
            return col
    raise RuntimeError(f"No player-name column available in live props columns={list(df.columns)}")


def _repair_frame(
    df: pd.DataFrame,
    *,
    roster: dict[str, set[str]],
    events: dict[str, tuple[str, str]],
    team_cols: tuple[str, ...],
    opp_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, int]:
    if df.empty or "event_id" not in df.columns:
        return df, 0
    out = df.copy()
    pcol = _player_col(out)
    resolved = []
    changed = 0
    for row in out.itertuples(index=False):
        data = row._asdict()
        existing = ""
        for c in team_cols:
            if c in out.columns and not _missing_text(data.get(c)):
                candidate = canon_team(data.get(c))
                if candidate in CANON_TEAM_CODES:
                    existing = candidate
                    break
        team = existing or _infer_team(data.get(pcol), data.get("event_id"), roster, events)
        if not existing and team:
            changed += 1
        resolved.append((team, _opp(team, data.get("event_id"), events) if team else ""))
    teams = [x[0] for x in resolved]
    opps = [x[1] for x in resolved]
    for c in team_cols:
        if c in out.columns:
            out[c] = teams
    for c in opp_cols:
        if c in out.columns:
            out[c] = opps
    return out, changed


def repair_live_prop_identity() -> dict:
    roles = _read(ROLES)
    compact = _read(OUTPUTS / "props_raw.csv")
    raw_data = _read(DATA / "props_raw.csv")
    enriched = _read(DATA / "props_enriched.csv", required=False)
    roster = _build_roster_index(roles)
    events = _event_map(enriched, raw_data)

    compact, changed_compact = _repair_frame(
        compact, roster=roster, events=events,
        team_cols=("team_abbr",), opp_cols=("opponent_abbr",),
    )
    raw_data, changed_raw = _repair_frame(
        raw_data, roster=roster, events=events,
        team_cols=("team_abbr", "team"), opp_cols=("opponent_abbr", "opponent"),
    )
    if not enriched.empty:
        enriched, changed_enriched = _repair_frame(
            enriched, roster=roster, events=events,
            team_cols=("player_team_abbr",), opp_cols=("opponent_team_abbr",),
        )
    else:
        changed_enriched = 0

    # Fail closed on the markets that actually feed the core player projection/pricing path.
    pcol = _player_col(compact)
    market = compact.get("market", pd.Series("", index=compact.index)).astype("string")
    core = compact.loc[market.isin(CORE_MARKETS)].copy()
    if core.empty:
        raise RuntimeError("Live props contain zero core QB/RB/WR/TE yardage/reception rows")
    core_player = core[pcol].astype("string").fillna("").str.strip()
    blank_player = core_player.eq("")
    team = core.get("team_abbr", pd.Series("", index=core.index)).astype("string").fillna("").str.strip()
    opp = core.get("opponent_abbr", pd.Series("", index=core.index)).astype("string").fillna("").str.strip()
    unresolved = blank_player | team.eq("") | opp.eq("")

    audit = core[[c for c in ("event_id", "market", pcol, "team_abbr", "opponent_abbr") if c in core.columns]].copy()
    audit = audit.rename(columns={pcol: "player"})
    audit["identity_resolved"] = (~unresolved).astype(int).to_numpy()
    audit = audit.drop_duplicates().sort_values(["identity_resolved", "market", "player"], ascending=[True, True, True])
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)

    status = {
        "core_rows": int(len(core)),
        "core_unique_players": int(core_player[~blank_player].nunique()),
        "core_unresolved_rows": int(unresolved.sum()),
        "core_unresolved_players": sorted(core.loc[unresolved, pcol].dropna().astype(str).unique().tolist()),
        "event_count": int(len(events)),
        "repaired_compact_rows": int(changed_compact),
        "repaired_raw_offer_rows": int(changed_raw),
        "repaired_enriched_rows": int(changed_enriched),
        "disposition": "LIVE_PROP_IDENTITY_READY" if not unresolved.any() else "LIVE_PROP_IDENTITY_FAILURE",
    }
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    compact.to_csv(OUTPUTS / "props_raw.csv", index=False)
    raw_data.to_csv(DATA / "props_raw.csv", index=False)
    if not enriched.empty:
        enriched.to_csv(DATA / "props_enriched.csv", index=False)

    # Keep the duplicate output enrichment artifact synchronized when present.
    out_enriched_path = OUTPUTS / "props_enriched.csv"
    if out_enriched_path.exists() and out_enriched_path.stat().st_size > 0:
        out_enriched = _read(out_enriched_path)
        out_enriched, _ = _repair_frame(
            out_enriched, roster=roster, events=events,
            team_cols=("team_abbr", "team"), opp_cols=("opponent_abbr", "opponent"),
        )
        out_enriched.to_csv(out_enriched_path, index=False)

    print("[live_prop_identity] " + json.dumps(status, sort_keys=True))
    if unresolved.any():
        raise RuntimeError(
            f"Core live prop player/team identity unresolved rows={int(unresolved.sum())}; "
            f"players={status['core_unresolved_players'][:30]}"
        )
    return status


def main() -> int:
    repair_live_prop_identity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
