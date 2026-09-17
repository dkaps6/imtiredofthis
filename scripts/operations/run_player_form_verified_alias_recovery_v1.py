#!/usr/bin/env python3
"""Replay PlayerForm while preserving verified current-name identity aliases.

This is an operational recovery wrapper for a preserved paid Full Slate artifact.
It does not change PlayerForm formulas or model features.  Before invoking the
canonical current-roles PlayerForm entry point, it decorates the existing identity
registry builder so verified aliases remain addressable even when a newer log for
the same stable GSIS identity/team uses a different name variant.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.run_player_form_current_roles_v1 as current_roles
from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_season
from scripts.utils.player_identity_v3 import clean_player_id, player_name_key

ALIASES = Path("data/player_identity_aliases.csv")


def _position(value) -> str:
    text = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if text in {"HB", "TB"} or text.startswith("RB"):
        return "RB"
    if text.startswith("WR") or text in {"LWR", "RWR", "SWR"}:
        return "WR"
    if text.startswith("TE"):
        return "TE"
    if text.startswith("QB"):
        return "QB"
    if text.startswith("FB"):
        return "FB"
    return text


def install_verified_alias_overlay() -> None:
    if not ALIASES.exists() or ALIASES.stat().st_size == 0:
        raise RuntimeError("verified identity alias config missing/empty")
    aliases = pd.read_csv(ALIASES, dtype="string").fillna("")
    required = {"current_name", "player_id", "current_team", "position"}
    missing = required - set(aliases.columns)
    if missing:
        raise RuntimeError(f"verified identity alias config missing columns: {sorted(missing)}")

    pf = current_roles.loader.runner.pf
    base_builder = pf.build_identity_registry
    season = int(resolve_season())

    def _builder_with_verified_aliases(logs: pd.DataFrame) -> pd.DataFrame:
        registry = base_builder(logs)
        if registry is None or registry.empty:
            raise RuntimeError("identity registry empty before verified alias overlay")
        out = registry.copy()
        additions: list[dict] = []

        for rec in aliases.to_dict("records"):
            current_name = str(rec["current_name"]).strip()
            pid = clean_player_id(rec["player_id"])
            team = canon_team(rec["current_team"])
            position = _position(rec["position"])
            identity = f"gsis:{pid}"
            if not current_name or not pid or not team or not position:
                raise RuntimeError(f"verified identity alias row is incomplete: {rec}")

            anchored = out.loc[
                out["player_identity_key"].astype(str).eq(identity)
                & out["player_id"].astype(str).eq(pid)
            ].copy()
            if anchored.empty:
                raise RuntimeError(
                    f"verified alias stable identity missing from rebuilt registry: {current_name} -> {identity}"
                )
            source_positions = set(anchored["position"].map(_position).dropna().astype(str))
            if position not in source_positions:
                raise RuntimeError(
                    f"verified alias position mismatch {current_name} alias={position} registry={sorted(source_positions)}"
                )

            full_key = player_name_key(current_name)
            base_key = player_name_key(current_name, strip_suffix=True)
            same_name_team = out.loc[
                out["team"].map(canon_team).astype(str).eq(team)
                & out["identity_full_name_key"].astype(str).eq(full_key)
            ]
            if not same_name_team.empty:
                identities = set(same_name_team["player_identity_key"].astype(str))
                if identities != {identity}:
                    raise RuntimeError(
                        f"verified alias collides with another identity: {current_name} team={team} identities={sorted(identities)}"
                    )
                continue

            additions.append({
                "player_identity_key": identity,
                "player_id": pid,
                "player": current_name,
                "team": team,
                "position": position,
                "identity_full_name_key": full_key,
                "identity_base_name_key": base_key,
                "last_season": season,
                "last_week": 0,
            })

        if additions:
            out = pd.concat([out, pd.DataFrame(additions)], ignore_index=True, sort=False)

        with_id = out.loc[out["player_id"].astype(str).str.len().gt(0)]
        collisions = with_id.groupby("player_id")["player_identity_key"].nunique()
        bad = collisions.loc[collisions.gt(1)]
        if not bad.empty:
            raise RuntimeError(f"stable player ID collision after verified alias overlay: {bad.to_dict()}")

        print(
            "[verified_alias_recovery] "
            f"configured_aliases={len(aliases)} added_registry_alias_rows={len(additions)} registry_rows={len(out)}"
        )
        return out.reset_index(drop=True)

    pf.build_identity_registry = _builder_with_verified_aliases


def main() -> int:
    install_verified_alias_overlay()
    return int(current_roles.main())


if __name__ == "__main__":
    raise SystemExit(main())
