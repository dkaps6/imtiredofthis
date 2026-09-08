#!/usr/bin/env python3
"""Provider lookup aliases for suffix-safe full-roster football distributions.

The football simulation owns one suffix-insensitive player identity. Sportsbook
rows may retain provider display keys such as ``marvinharrisonjr``. After the
football distribution is generated, install lookup aliases from the canonical
football key to the provider pricing key. This is lookup plumbing only: no player
is simulated twice and no sportsbook value enters the football distribution.
"""
from __future__ import annotations

import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
from scripts.simulation_v2 import MARKET_MAP
from scripts.utils.player_identity_v3 import player_name_key


def _suffix_safe_key(value) -> str:
    return str(player_name_key(value, strip_suffix=True) or "").strip()


def _install_provider_player_aliases_and_validate(result, metrics: pd.DataFrame) -> tuple[int, list[dict]]:
    frame = metrics.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    need = {"event_id", "player", "team", "opponent", "season", "week", "market"}
    missing_cols = need - set(frame.columns)
    if missing_cols:
        raise RuntimeError(f"provider distribution aliasing missing columns: {sorted(missing_cols)}")

    # One validation row per provider player/market/game is enough; book/line
    # variants all consume the same already-generated football distribution.
    unique = frame.sort_values(["event_id", "team", "player", "market"]).drop_duplicates(
        ["event_id", "team", "player", "market"], keep="last"
    )
    missing: list[dict] = []
    alias_count = 0
    for _, row in unique.iterrows():
        provider_game = str(row.get("event_id") or "").strip()
        canonical_game = base._canonical_game(
            row.get("team"), row.get("opponent"), row.get("season"), row.get("week")
        )
        canonical_player = _suffix_safe_key(row.get("player"))
        provider_player = base._player_key(row)
        market = MARKET_MAP.get(str(row.get("market", "")).lower(), str(row.get("market", "")).lower())
        if not canonical_player or not provider_player:
            missing.append({
                "event_id": provider_game, "player": row.get("player"), "team": row.get("team"),
                "market": row.get("market"), "reason": "blank_player_lookup_key",
            })
            continue

        values = result.values.get((canonical_game, canonical_player, market))
        if values is None:
            # Event aliases are installed before this function in V1, so this is
            # a useful defensive second lookup but never the simulation source.
            values = result.values.get((provider_game, canonical_player, market))
        if values is None or len(values) == 0:
            missing.append({
                "event_id": provider_game, "player": row.get("player"), "team": row.get("team"),
                "market": row.get("market"), "reason": "canonical_distribution_missing",
                "canonical_player_key": canonical_player,
            })
            continue

        for game in {canonical_game, provider_game}:
            key = (game, provider_player, market)
            if key not in result.values:
                result.values[key] = values
                alias_count += 1

    if missing:
        return int(len(unique)), missing

    # Final verification uses the unchanged downstream lookup contract. If this
    # passes, run_pricing_v2 can iterate its original provider rows safely.
    verify_missing: list[dict] = []
    from scripts.simulation_v2 import lookup
    for _, row in unique.iterrows():
        values = lookup(result, row, str(row.get("market", "")))
        if values is None or len(values) == 0:
            verify_missing.append({
                "event_id": row.get("event_id"), "player": row.get("player"),
                "team": row.get("team"), "market": row.get("market"),
                "reason": "provider_lookup_after_alias_failed",
            })
    if verify_missing:
        return int(len(unique)), verify_missing

    print(
        f"[football_simulation_universe] provider_player_lookup_aliases={alias_count} "
        f"validated_provider_player_market_keys={len(unique)}"
    )
    return int(len(unique)), []


def main() -> int:
    base._identity_frame = v2._canonical_identity_frame
    base._validate_priced_distribution_coverage = _install_provider_player_aliases_and_validate
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
