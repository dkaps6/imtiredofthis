#!/usr/bin/env python3
"""Semantic audit for current-slate roster, sportsbook, and historical identities.

Structural identity validation answers whether every row has a key. This audit
checks whether live sportsbook players map to exactly one current roster row and
whether temporary identities resemble a genuinely unresolved historical alias.
Review candidates are never auto-attached to history.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

DATA = Path("data")
OUTPUTS = Path("outputs")
FORM = DATA / "player_form.csv"
REGISTRY = DATA / "player_identity_registry.csv"
SLATE = DATA / "player_identity_slate.csv"
COMPACT = OUTPUTS / "props_raw_compact.csv"
AUDIT = DATA / "player_identity_semantic_audit.csv"
SUMMARY = DATA / "player_identity_semantic_audit.json"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}


def _read(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        if required:
            raise RuntimeError(f"identity semantic artifact missing/empty: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"identity semantic artifact has zero rows: {path}")
    return df


def _tokens(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    parts = [p for p in re.sub(r"[^a-z0-9]+", " ", text).split() if p]
    while parts and parts[-1] in SUFFIXES:
        parts.pop()
    return parts


def _first_last(value) -> tuple[str, str]:
    parts = _tokens(value)
    return (parts[0], parts[-1]) if parts else ("", "")


def _position(value) -> str:
    x = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if x in {"HB", "TB", "FB"}:
        return "RB"
    if x.startswith("WR") or x in {"LWR", "RWR", "SWR"}:
        return "WR"
    return x


def _first_name_variant(a: str, b: str) -> bool:
    """Conservative review-only nickname/long-form signal.

    Same first initial was far too broad (for example Kevin/Keon, Jalon/Jayden,
    Kaytron/Kazmeir). Exact first names or a >=4-character prefix relationship
    catches provider shortening such as Josh/Joshua and Chris/Christopher without
    turning unrelated players into certification blockers. Non-prefix nicknames
    must be handled through the verified identity alias configuration.
    """
    a = str(a or "").lower().strip()
    b = str(b or "").lower().strip()
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _plausible_registry_candidates(player: str, position: str, registry: pd.DataFrame) -> list[dict]:
    first, last = _first_last(player)
    if not first or not last or registry.empty:
        return []
    candidates: list[dict] = []
    seen: set[str] = set()
    for row in registry.itertuples(index=False):
        rplayer = getattr(row, "player", "")
        rfirst, rlast = _first_last(rplayer)
        if not rfirst or rlast != last or not _first_name_variant(first, rfirst):
            continue
        rpos = _position(getattr(row, "position", ""))
        if position and rpos and rpos != position:
            continue
        identity = str(getattr(row, "player_identity_key", "") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        candidates.append({
            "player": str(rplayer),
            "identity": identity,
            "last_team": canon_team(getattr(row, "team", "")),
            "position": rpos,
        })
    return candidates[:10]


def audit() -> dict:
    form = _read(FORM)
    registry = _read(REGISTRY, required=False)
    slate = _read(SLATE)
    compact = _read(COMPACT, required=False)

    required = {
        "player", "team", "position", "player_identity_key", "identity_resolution",
        "identity_confidence",
    }
    missing = required - set(form.columns)
    if missing:
        raise RuntimeError(f"PlayerForm missing semantic identity columns: {sorted(missing)}")

    form = form.copy()
    form["team"] = form["team"].map(canon_team)
    form["position_group_semantic"] = form["position"].map(_position)
    form["name_base_key_semantic"] = form["player"].map(lambda v: player_name_key(v, strip_suffix=True))
    if form[["team", "name_base_key_semantic"]].astype("string").fillna("").eq("").any().any():
        raise RuntimeError("PlayerForm has blank current roster name/team keys")
    if form.duplicated(["team", "name_base_key_semantic"]).any():
        sample = form.loc[
            form.duplicated(["team", "name_base_key_semantic"], keep=False),
            ["player", "team", "position", "player_identity_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"current roster is ambiguous at normalized team/name grain: {sample}")

    if not {"player", "team"}.issubset(slate.columns):
        raise RuntimeError("player_identity_slate missing player/team")
    slate = slate.copy()
    slate["team"] = slate["team"].map(canon_team)
    slate["name_base_key_semantic"] = slate["player"].map(lambda v: player_name_key(v, strip_suffix=True))
    form_keys = set(zip(form["team"], form["name_base_key_semantic"]))
    slate_keys = set(zip(slate["team"], slate["name_base_key_semantic"]))
    if form_keys != slate_keys:
        raise RuntimeError(
            "PlayerForm current roster drifted from Player Identity slate; "
            f"missing_from_form={list(slate_keys-form_keys)[:20]} extra_in_form={list(form_keys-slate_keys)[:20]}"
        )

    prop_markets: dict[tuple[str, str], set[str]] = {}
    prop_rows = 0
    if not compact.empty:
        need = {"player", "team_abbr", "market"}
        if not need.issubset(compact.columns):
            raise RuntimeError(f"compact sportsbook artifact missing roster-audit columns: {sorted(need-set(compact.columns))}")
        compact = compact.copy()
        compact["team_semantic"] = compact["team_abbr"].map(canon_team)
        compact["name_base_key_semantic"] = compact["player"].map(lambda v: player_name_key(v, strip_suffix=True))
        mismatches = []
        for row in compact.itertuples(index=False):
            key = (str(row.team_semantic), str(row.name_base_key_semantic))
            if key not in form_keys:
                mismatches.append({"player": row.player, "team": row.team_semantic, "market": row.market})
                continue
            prop_markets.setdefault(key, set()).add(str(row.market))
            prop_rows += 1
        if mismatches:
            raise RuntimeError(
                "sportsbook players do not map exactly to current roster after identity repair; "
                f"rows={len(mismatches)} sample={mismatches[:30]}"
            )

    registry = registry.copy()
    if not registry.empty:
        if "team" in registry.columns:
            registry["team"] = registry["team"].map(canon_team)
        if "position" not in registry.columns:
            registry["position"] = ""

    audit_rows = []
    suspicious = temp = temp_props = stable = trade = 0
    for row in form.itertuples(index=False):
        player = str(row.player)
        team = str(row.team)
        position = str(row.position_group_semantic)
        identity = str(row.player_identity_key)
        resolution = str(row.identity_resolution)
        confidence = float(row.identity_confidence)
        key = (team, str(row.name_base_key_semantic))
        markets = sorted(prop_markets.get(key, set()))
        candidates: list[dict] = []
        if identity.startswith("temp:"):
            temp += 1
            temp_props += int(bool(markets))
            candidates = _plausible_registry_candidates(player, position, registry)
            if candidates:
                semantic_status = "TEMP_POSSIBLE_HISTORICAL_ALIAS_REVIEW_REQUIRED"
                suspicious += 1
            else:
                semantic_status = "TEMP_NO_PLAUSIBLE_REGISTRY_NAME_MATCH"
        else:
            stable += 1
            if "trade" in resolution:
                trade += 1
                semantic_status = "STABLE_TRADE_RESOLVED"
            else:
                semantic_status = "STABLE_CURRENT_TEAM_RESOLVED"
        audit_rows.append({
            "player": player,
            "team": team,
            "position": position,
            "player_identity_key": identity,
            "identity_resolution": resolution,
            "identity_confidence": confidence,
            "semantic_status": semantic_status,
            "sportsbook_market_count": len(markets),
            "sportsbook_markets": "|".join(markets),
            "historical_candidate_count": len(candidates),
            "historical_candidates_json": json.dumps(candidates, sort_keys=True),
        })

    out = pd.DataFrame(audit_rows)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT, index=False)
    result = {
        "disposition": "IDENTITY_SEMANTIC_REVIEW_REQUIRED" if suspicious else "IDENTITY_SEMANTICALLY_CONSISTENT_WITH_NEW_UNMAPPED_PLAYERS",
        "slate_players": int(len(form)),
        "stable_players": int(stable),
        "trade_resolved_players": int(trade),
        "temporary_players": int(temp),
        "temporary_sportsbook_players": int(temp_props),
        "temporary_possible_historical_aliases": int(suspicious),
        "sportsbook_compact_rows_checked": int(prop_rows),
        "sportsbook_current_roster_mismatches": 0,
        "playerform_slate_key_match": True,
        "alias_review_heuristic": "same_surname_position_and_exact_or_prefix_first_name",
        "audit": str(AUDIT),
    }
    SUMMARY.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[identity_semantics] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
