#!/usr/bin/env python3
"""Stable player identity resolution for 2026 production.

Names are aliases, not primary identifiers.  Historical nflverse/GSIS player IDs
are authoritative whenever available.  Name matching is used only to bridge live
providers that do not expose a stable NFL player ID (for example Ourlads and
sportsbook prop feeds).

Resolution order for a live roster/prop identity:
1. explicit stable player ID when supplied by the provider;
2. exact normalized name + current team;
3. suffix-insensitive name + current team (e.g. Ourlads drops Jr/II/III);
4. globally unique exact name (supports offseason trades);
5. globally unique suffix-insensitive name (supports trades + suffix variance);
6. deterministic temporary team/name identity for genuinely new/unmapped players.

Any ambiguous candidate set is fatal in strict mode.  Production should fail
rather than silently attach another player's historical usage.
"""
from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

import pandas as pd

from scripts._opponent_map import canon_team

SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_player_id(value) -> str:
    text = _text(value)
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def _tokens(value) -> list[str]:
    text = _text(value)
    if not text:
        return []
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().replace("’", "'")
    # Punctuation is identity-neutral.  Apostrophes/hyphens/spaces all collapse
    # to the same alphanumeric token stream.
    normalized = _NON_ALNUM.sub(" ", normalized)
    return [part for part in normalized.split() if part]


def player_name_key(value, *, strip_suffix: bool = False) -> str:
    parts = _tokens(value)
    if strip_suffix:
        while parts and parts[-1] in SUFFIX_TOKENS:
            parts.pop()
    return "".join(parts)


def historical_identity_key(player_id, player_name, team="") -> str:
    pid = clean_player_id(player_id)
    if pid:
        return f"gsis:{pid}"
    base = player_name_key(player_name, strip_suffix=True)
    tm = canon_team(team)
    if not base:
        return ""
    # Missing provider IDs should be rare in nflverse weekly data.  Keep the
    # fallback team-scoped so two unrelated same-name players cannot merge.
    return f"hist-name:{tm}:{base}" if tm else f"hist-name:{base}"


def attach_historical_identity(
    frame: pd.DataFrame,
    *,
    id_col: str = "player_id",
    name_col: str = "player",
    team_col: str = "team",
) -> pd.DataFrame:
    out = frame.copy()
    ids = out[id_col] if id_col in out.columns else pd.Series("", index=out.index)
    names = out[name_col] if name_col in out.columns else pd.Series("", index=out.index)
    teams = out[team_col] if team_col in out.columns else pd.Series("", index=out.index)
    out["player_id"] = ids.map(clean_player_id)
    out["identity_full_name_key"] = names.map(player_name_key)
    out["identity_base_name_key"] = names.map(lambda v: player_name_key(v, strip_suffix=True))
    out["player_identity_key"] = [
        historical_identity_key(pid, name, team)
        for pid, name, team in zip(out["player_id"], names, teams)
    ]
    return out


def build_identity_registry(logs: pd.DataFrame) -> pd.DataFrame:
    """Build a team-aware registry from pregame-eligible historical player logs."""
    if logs is None or logs.empty:
        return pd.DataFrame(
            columns=[
                "player_identity_key", "player_id", "player", "team", "position",
                "identity_full_name_key", "identity_base_name_key", "last_season", "last_week",
            ]
        )

    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "player_identity_key" not in x.columns:
        x = attach_historical_identity(x)
    if "identity_full_name_key" not in x.columns:
        x["identity_full_name_key"] = x.get("player", pd.Series("", index=x.index)).map(player_name_key)
    if "identity_base_name_key" not in x.columns:
        x["identity_base_name_key"] = x.get("player", pd.Series("", index=x.index)).map(
            lambda v: player_name_key(v, strip_suffix=True)
        )
    if "team" not in x.columns:
        x["team"] = ""
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x.get("player_id", pd.Series("", index=x.index)).map(clean_player_id)
    x["season"] = pd.to_numeric(x.get("season"), errors="coerce")
    x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
    x = x.loc[x["player_identity_key"].astype(str).str.len().gt(0)].copy()
    if x.empty:
        return pd.DataFrame()

    # Keep one latest observation per stable identity/team.  A traded player can
    # therefore have multiple team rows pointing to the same GSIS identity.
    x = x.sort_values(["season", "week"], kind="mergesort")
    rows = []
    for (identity, team), part in x.groupby(["player_identity_key", "team"], dropna=False):
        last = part.iloc[-1]
        rows.append(
            {
                "player_identity_key": str(identity),
                "player_id": clean_player_id(last.get("player_id")),
                "player": _text(last.get("player")),
                "team": str(team or ""),
                "position": _text(last.get("position")),
                "identity_full_name_key": _text(last.get("identity_full_name_key")),
                "identity_base_name_key": _text(last.get("identity_base_name_key")),
                "last_season": int(last["season"]) if pd.notna(last.get("season")) else pd.NA,
                "last_week": int(last["week"]) if pd.notna(last.get("week")) else pd.NA,
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry

    # A stable player ID must never point to more than one identity key.
    with_id = registry.loc[registry["player_id"].astype(str).str.len().gt(0)]
    if not with_id.empty:
        collisions = with_id.groupby("player_id")["player_identity_key"].nunique()
        bad = collisions.loc[collisions.gt(1)]
        if not bad.empty:
            raise RuntimeError(f"Stable player ID collision in identity registry: {bad.to_dict()}")
    return registry.reset_index(drop=True)


def _index_sets(registry: pd.DataFrame, key_cols: list[str]) -> dict[tuple, set[str]]:
    out: dict[tuple, set[str]] = defaultdict(set)
    if registry.empty:
        return out
    for _, row in registry.iterrows():
        key = tuple(_text(row.get(c)) for c in key_cols)
        if all(key):
            out[key].add(str(row["player_identity_key"]))
    return out


def _identity_metadata(registry: pd.DataFrame) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if registry.empty:
        return meta
    # Prefer the most recent registry row for display metadata.
    x = registry.copy()
    x["last_season"] = pd.to_numeric(x.get("last_season"), errors="coerce").fillna(-1)
    x["last_week"] = pd.to_numeric(x.get("last_week"), errors="coerce").fillna(-1)
    x = x.sort_values(["last_season", "last_week"], kind="mergesort")
    for identity, part in x.groupby("player_identity_key", dropna=False):
        last = part.iloc[-1]
        meta[str(identity)] = {
            "player_id": clean_player_id(last.get("player_id")),
            "registry_player": _text(last.get("player")),
            "registry_team": _text(last.get("team")),
        }
    return meta


def resolve_slate_identities(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    name_col: str = "player",
    team_col: str = "team",
    provider_id_col: str | None = None,
    strict_ambiguous: bool = True,
    allow_temporary: bool = True,
) -> pd.DataFrame:
    """Attach a stable identity key to current roster/prop rows.

    Temporary identities are valid for true rookies/new players with no
    historical registry match.  They are explicitly labeled and never inherit
    another player's history.
    """
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if name_col not in frame.columns or team_col not in frame.columns:
        raise RuntimeError(f"identity resolution requires {name_col}/{team_col}")

    reg = registry.copy() if registry is not None else pd.DataFrame()
    if not reg.empty:
        reg.columns = [str(c).strip().lower() for c in reg.columns]
        reg["team"] = reg["team"].map(canon_team)

    team_full = _index_sets(reg, ["team", "identity_full_name_key"])
    team_base = _index_sets(reg, ["team", "identity_base_name_key"])
    global_full = _index_sets(reg, ["identity_full_name_key"])
    global_base = _index_sets(reg, ["identity_base_name_key"])
    id_map: dict[str, set[str]] = defaultdict(set)
    if not reg.empty and "player_id" in reg.columns:
        for _, r in reg.iterrows():
            pid = clean_player_id(r.get("player_id"))
            if pid:
                id_map[pid].add(str(r["player_identity_key"]))
    meta = _identity_metadata(reg)

    out = frame.copy()
    out[team_col] = out[team_col].map(canon_team)
    full_keys = out[name_col].map(player_name_key)
    base_keys = out[name_col].map(lambda v: player_name_key(v, strip_suffix=True))

    resolved_keys: list[str] = []
    methods: list[str] = []
    confidences: list[float] = []
    ambiguity_notes: list[str] = []
    player_ids: list[str] = []
    registry_players: list[str] = []

    for idx, row in out.iterrows():
        team = canon_team(row.get(team_col))
        full = full_keys.loc[idx]
        base = base_keys.loc[idx]
        provider_id = clean_player_id(row.get(provider_id_col)) if provider_id_col and provider_id_col in out.columns else ""

        chosen = ""
        method = ""
        confidence = 0.0
        ambiguous: set[str] = set()

        candidates: list[tuple[str, float, set[str]]] = []
        if provider_id:
            candidates.append(("provider_player_id", 1.0, id_map.get(provider_id, set())))
        if team and full:
            candidates.append(("team_exact_name", 0.995, team_full.get((team, full), set())))
        if team and base:
            candidates.append(("team_suffix_alias", 0.985, team_base.get((team, base), set())))
        if full:
            candidates.append(("unique_name_trade", 0.95, global_full.get((full,), set())))
        if base:
            candidates.append(("unique_base_trade", 0.90, global_base.get((base,), set())))

        for candidate_method, candidate_conf, identities in candidates:
            identities = set(identities)
            if len(identities) == 1:
                chosen = next(iter(identities))
                method = candidate_method
                confidence = candidate_conf
                break
            if len(identities) > 1:
                ambiguous = identities
                method = f"ambiguous_{candidate_method}"
                # Do not fall through to a weaker criterion after a stronger
                # criterion is already ambiguous; that would be silent guessing.
                break

        if ambiguous:
            note = f"team={team} player={_text(row.get(name_col))} candidates={sorted(ambiguous)}"
            if strict_ambiguous:
                raise RuntimeError(f"Ambiguous player identity: {note}")
            ambiguity_notes.append(note)
        else:
            ambiguity_notes.append("")

        if not chosen:
            if not allow_temporary:
                raise RuntimeError(
                    f"Unresolved player identity team={team} player={_text(row.get(name_col))}"
                )
            fallback = base or full
            if not team or not fallback:
                raise RuntimeError(
                    f"Cannot create temporary player identity team={team} player={_text(row.get(name_col))}"
                )
            chosen = f"temp:{team}:{fallback}"
            method = "new_or_unmapped_roster"
            confidence = 0.50

        info = meta.get(chosen, {})
        resolved_keys.append(chosen)
        methods.append(method)
        confidences.append(confidence)
        player_ids.append(clean_player_id(info.get("player_id")))
        registry_players.append(_text(info.get("registry_player")))

    out["identity_full_name_key"] = full_keys
    out["identity_base_name_key"] = base_keys
    out["player_identity_key"] = resolved_keys
    out["player_id"] = player_ids
    out["identity_resolution"] = methods
    out["identity_confidence"] = confidences
    out["identity_ambiguity"] = ambiguity_notes
    out["identity_registry_player"] = registry_players
    return out


def assert_unique_roster_identities(frame: pd.DataFrame, *, context: str = "roster") -> None:
    if frame is None or frame.empty:
        return
    required = {"team", "player_identity_key"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{context} missing identity columns: {sorted(required - set(frame.columns))}")
    x = frame[["team", "player_identity_key"]].drop_duplicates()
    duplicate = x.duplicated(["team", "player_identity_key"], keep=False)
    if duplicate.any():
        sample = x.loc[duplicate].head(20).to_dict("records")
        raise RuntimeError(f"{context} contains duplicate resolved player identities: {sample}")
