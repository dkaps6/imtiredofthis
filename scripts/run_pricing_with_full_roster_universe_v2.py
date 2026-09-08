#!/usr/bin/env python3
"""Suffix-safe identity + provider lookup contract for full-roster simulation.

Full Slate providers legitimately disagree on suffixes (Jr/Sr/II/III/etc.). The
football roster uses one deterministic Player Identity v3 base-name key. After
Monte Carlo generation, sportsbook/provider player keys are installed only as
lookup aliases to that already-generated distribution. No player is simulated
twice and no sportsbook value enters the football distribution.
"""
from __future__ import annotations

import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
from scripts.simulation_v2 import MARKET_MAP, lookup
from scripts.utils.player_identity_v3 import player_name_key


def _suffix_safe_key(value) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _canonical_identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team"] = out["team"].map(base.canon_team)
    if "player" not in out.columns:
        raise RuntimeError("full-roster identity frame requires player display name for suffix-safe identity")
    # Do not trust provider-specific player_clean_key representation here. Build
    # the identity key from display name exactly as metrics_v2 does.
    out["player_clean_key"] = out["player"].map(_suffix_safe_key)
    blank = out["player_clean_key"].astype("string").fillna("").str.strip().eq("")
    if blank.any():
        sample = out.loc[blank, [c for c in ("player", "team") if c in out.columns]].head(20).to_dict("records")
        raise RuntimeError(f"suffix-safe full-roster identity key unresolved: {sample}")

    # Pricing legitimately contains many rows per player (book/market/line). A
    # collision exists only when DISTINCT display names on the SAME team collapse
    # to one suffix-insensitive key. Repeated rows for Marvin Harrison Jr. are one
    # identity and must not be treated as an ambiguity.
    names = (
        out.assign(_display=out["player"].astype("string").fillna("").str.strip())
        .groupby(["team", "player_clean_key"], dropna=False)["_display"]
        .nunique(dropna=False)
    )
    bad_keys = names.loc[names.gt(1)]
    if not bad_keys.empty:
        bad_index = set(bad_keys.index.tolist())
        mask = [(str(t), str(k)) in bad_index for t, k in zip(out["team"], out["player_clean_key"])]
        sample = out.loc[mask, ["player", "team", "player_clean_key"]].drop_duplicates().sort_values(
            ["team", "player_clean_key", "player"], kind="mergesort"
        ).head(20).to_dict("records")
        raise RuntimeError(f"suffix-safe full-roster identity is ambiguous within current team: {sample}")
    return out


def _install_provider_player_aliases_and_validate(result, metrics: pd.DataFrame) -> tuple[int, list[dict]]:
    frame = metrics.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    need = {"event_id", "player", "team", "opponent", "season", "week", "market"}
    missing_cols = need - set(frame.columns)
    if missing_cols:
        raise RuntimeError(f"provider distribution aliasing missing columns: {sorted(missing_cols)}")

    # Book/line variants all consume one player/game/market football distribution.
    unique = frame.sort_values(["event_id", "team", "player", "market"]).drop_duplicates(
        ["event_id", "team", "player", "market"], keep="last"
    )
    missing: list[dict] = []
    aliases = 0
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
            # V1 installs event-id aliases before this callback. This fallback is
            # lookup-only and never changes which game/player was simulated.
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
                aliases += 1

    if missing:
        return int(len(unique)), missing

    # Prove the unchanged downstream pricing lookup can now resolve every row.
    verify_missing: list[dict] = []
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
        f"[football_simulation_universe] provider_player_lookup_aliases={aliases} "
        f"validated_provider_player_market_keys={len(unique)}"
    )
    return int(len(unique)), []


def main() -> int:
    base._identity_frame = _canonical_identity_frame
    base._validate_priced_distribution_coverage = _install_provider_player_aliases_and_validate
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
