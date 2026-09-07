#!/usr/bin/env python3
"""Repair and validate live sportsbook player -> team/opponent identity.

The active 2026 Ourlads roster plus the sportsbook event participants are the
identity authority. Historical team affiliation is intentionally irrelevant:
trades, free agency and rookies must resolve to their current team.

This adapter never changes lines, odds or football projections. It only repairs
provider naming variants and fails closed when a real core player offer cannot
be assigned to exactly one of the two teams in its event. Bookmaker-missing
placeholder rows are preserved but excluded from player-identity failure counts.
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
    # Ourlads occasionally drops a middle surname token (for example St. in
    # Amon-Ra St. Brown), so keep a first+last key as a provider-tolerant key.
    if len(tokens) >= 2:
        keys.add(tokens[0] + tokens[-1])
    return {k for k in keys if k}


def _build_roster_index(roles: pd.DataFrame) -> dict[str, set[str]]:
    missing = {"team", "player"} - set(roles.columns)
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
    for frame in (enriched, raw_data):
        if not frame.empty and {"event_id", "home_team_abbr", "away_team_abbr"}.issubset(frame.columns):
            candidates.append(frame[["event_id", "home_team_abbr", "away_team_abbr"]])
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


def _player_col(df: pd.DataFrame) -> str:
    for col in ("canonical_player_name", "player_canonical", "player", "player_name_raw"):
        if col in df.columns:
            return col
    raise RuntimeError(f"No player-name column available in live props columns={list(df.columns)}")


def _is_placeholder(row: dict) -> bool:
    try:
        if int(float(row.get("bookmaker_missing", 0) or 0)) == 1:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _infer_team(player, event_id, roster: dict[str, set[str]], events: dict[str, tuple[str, str]]) -> str:
    pair = events.get(str(event_id))
    keys = _name_keys(player)
    if pair is None or not keys:
        return ""
    matches = [team for team in pair if keys & roster.get(team, set())]
    matches = list(dict.fromkeys(matches))
    return matches[0] if len(matches) == 1 else ""


def _opponent(team: str, event_id, events: dict[str, tuple[str, str]]) -> str:
    pair = events.get(str(event_id))
    if not pair or team not in pair:
        return ""
    return pair[1] if pair[0] == team else pair[0]


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
    resolved: list[tuple[str, str]] = []
    changed = 0
    for row in out.itertuples(index=False):
        data = row._asdict()
        if _is_placeholder(data) or _missing_text(data.get(pcol)):
            resolved.append(("", ""))
            continue
        pair = events.get(str(data.get("event_id")))
        existing = ""
        for c in team_cols:
            if c in out.columns and not _missing_text(data.get(c)):
                candidate = canon_team(data.get(c))
                # Existing identity is accepted only if it is one of the two
                # teams in this current sportsbook event.
                if candidate in CANON_TEAM_CODES and pair and candidate in pair:
                    existing = candidate
                    break
        team = existing or _infer_team(data.get(pcol), data.get("event_id"), roster, events)
        if not existing and team:
            changed += 1
        resolved.append((team, _opponent(team, data.get("event_id"), events) if team else ""))
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

    pcol = _player_col(compact)
    market = compact.get("market", pd.Series("", index=compact.index)).astype("string")
    player = compact[pcol].astype("string").fillna("").str.strip()
    bookmaker_missing = pd.to_numeric(
        compact.get("bookmaker_missing", pd.Series(0, index=compact.index)), errors="coerce"
    ).fillna(0).eq(1)
    placeholder_mask = market.isin(CORE_MARKETS) & (bookmaker_missing | player.eq(""))
    actual_core_mask = market.isin(CORE_MARKETS) & ~bookmaker_missing & player.ne("")
    core = compact.loc[actual_core_mask].copy()
    if core.empty:
        raise RuntimeError("Live props contain zero actual core QB/RB/WR/TE yardage/reception player rows")

    core_team = core.get("team_abbr", pd.Series("", index=core.index)).astype("string").fillna("").str.strip()
    core_opp = core.get("opponent_abbr", pd.Series("", index=core.index)).astype("string").fillna("").str.strip()
    unresolved = core_team.eq("") | core_opp.eq("")

    audit = core[[c for c in ("event_id", "market", pcol, "team_abbr", "opponent_abbr") if c in core.columns]].copy()
    audit = audit.rename(columns={pcol: "player"})
    audit["identity_resolved"] = (~unresolved).astype(int).to_numpy()
    audit = audit.drop_duplicates().sort_values(
        ["identity_resolved", "market", "player"], ascending=[True, True, True]
    )
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)

    status = {
        "core_actual_rows": int(len(core)),
        "core_unique_players": int(core[pcol].astype(str).nunique()),
        "core_placeholder_rows_ignored": int(placeholder_mask.sum()),
        "core_unresolved_rows": int(unresolved.sum()),
        "core_unresolved_players": sorted(core.loc[unresolved, pcol].dropna().astype(str).unique().tolist()),
        "event_count": int(len(events)),
        "repaired_compact_rows": int(changed_compact),
        "repaired_raw_offer_rows": int(changed_raw),
        "repaired_enriched_rows": int(changed_enriched),
        "roster_authority": "current_ourlads_plus_current_event_participants",
        "historical_team_affiliation_used": False,
        "disposition": "LIVE_PROP_IDENTITY_READY" if not unresolved.any() else "LIVE_PROP_IDENTITY_FAILURE",
    }
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    compact.to_csv(OUTPUTS / "props_raw.csv", index=False)
    raw_data.to_csv(DATA / "props_raw.csv", index=False)
    if not enriched.empty:
        enriched.to_csv(DATA / "props_enriched.csv", index=False)

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
            f"Actual core live prop player/team identity unresolved rows={int(unresolved.sum())}; "
            f"players={status['core_unresolved_players'][:30]}"
        )
    return status


def main() -> int:
    repair_live_prop_identity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
