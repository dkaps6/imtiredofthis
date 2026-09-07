#!/usr/bin/env python3
"""Materialize the live sportsbook artifact that modeling is actually allowed to consume.

Provider outputs are a superset of production model capability. This boundary
adapter never changes a line or odds value. It explicitly quarantines provider
sentinels, unsupported markets, and non-player/unmodeled anytime-TD entities,
while keeping all supported yardage/reception player markets fail-closed on
current player/team/opponent identity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
STATUS = DATA / "live_odds_status.json"
PROPS = OUTPUTS / "props_raw.csv"
QUARANTINE = DATA / "live_odds_placeholder_rows.csv"

# These are the markets the current pricing_v2 MARKET_MAP can actually price
# from the configured OddsAPI feed. Pass/rush TD markets may be collected by the
# provider adapter but are not current production pricing markets.
SUPPORTED_MARKETS = {
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
    "player_anytime_td",
}

# These markets require a real current player identity with no fallback/drop.
# An unresolved row in any of them is a production data failure.
STRICT_PLAYER_MARKETS = {
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
}


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _is_non_player_entity(name: str) -> bool:
    x = str(name or "").strip().lower()
    return bool(
        re.search(r"(?:\bd/st\b|\bdefense\b|\bdefence\b|\bspecial teams\b)", x)
    )


def _append_quarantine(parts: list[pd.DataFrame], frame: pd.DataFrame, reason: str) -> None:
    if frame.empty:
        return
    x = frame.copy()
    x["quarantine_reason"] = reason
    parts.append(x)


def materialize() -> dict:
    if not STATUS.exists() or STATUS.stat().st_size <= 0:
        raise RuntimeError("live_odds_status.json missing before model-facing prop materialization")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if not bool(status.get("available")):
        result = {
            "disposition": "MODEL_LIVE_PROPS_NOT_REQUIRED",
            "status": status.get("status", "unknown"),
            "rows": 0,
            "quarantined_rows": 0,
        }
        print("[model_live_props] " + json.dumps(result, sort_keys=True))
        return result

    if not PROPS.exists() or PROPS.stat().st_size <= 0:
        raise RuntimeError("live odds status says available but outputs/props_raw.csv is missing/empty")
    props = pd.read_csv(PROPS, low_memory=False)
    if props.empty:
        raise RuntimeError("live odds status says available but outputs/props_raw.csv has zero rows")
    props.columns = [str(c).strip().lower() for c in props.columns]

    for col in ("event_id", "market"):
        if col not in props.columns:
            raise RuntimeError(f"compact live props missing required provider column: {col}")

    pcol = next(
        (c for c in ("canonical_player_name", "player_canonical", "player", "book_player_name") if c in props.columns),
        None,
    )
    if pcol is None:
        raise RuntimeError(f"model-facing live props have no player column; columns={list(props.columns)}")

    quarantine_parts: list[pd.DataFrame] = []

    missing_flag = pd.to_numeric(
        props.get("bookmaker_missing", pd.Series(0, index=props.index)), errors="coerce"
    ).fillna(0).eq(1)
    _append_quarantine(quarantine_parts, props.loc[missing_flag], "BOOKMAKER_MISSING_SENTINEL")
    actual = props.loc[~missing_flag].copy()
    if actual.empty:
        raise RuntimeError("all compact live prop rows are bookmaker-missing sentinels")

    market = _text(actual["market"])
    unsupported = ~market.isin(SUPPORTED_MARKETS)
    _append_quarantine(quarantine_parts, actual.loc[unsupported], "UNSUPPORTED_PRODUCTION_MARKET")
    model = actual.loc[~unsupported].copy()
    if model.empty:
        raise RuntimeError("live props contain zero rows in production-supported markets")

    player = _text(model[pcol])
    model["player"] = player
    blank_player = player.eq("")
    strict_blank = blank_player & _text(model["market"]).isin(STRICT_PLAYER_MARKETS)
    if strict_blank.any():
        sample = model.loc[strict_blank, [c for c in ("event_id", "market", pcol) if c in model.columns]].head(20)
        raise RuntimeError(
            "supported yardage/reception rows contain blank player identity; "
            f"rows={int(strict_blank.sum())} sample={sample.to_dict('records')}"
        )
    _append_quarantine(quarantine_parts, model.loc[blank_player], "BLANK_NONCORE_PLAYER")
    model = model.loc[~blank_player].copy()

    for col in ("team_abbr", "opponent_abbr"):
        if col not in model.columns:
            raise RuntimeError(f"model-facing live props missing required identity column: {col}")

    team_bad = _text(model["team_abbr"]).eq("")
    opp_bad = _text(model["opponent_abbr"]).eq("")
    identity_bad = team_bad | opp_bad
    strict_mask = _text(model["market"]).isin(STRICT_PLAYER_MARKETS)
    strict_bad = identity_bad & strict_mask
    if strict_bad.any():
        sample = model.loc[
            strict_bad,
            [c for c in ("event_id", "market", "player", "team_abbr", "opponent_abbr") if c in model.columns],
        ].head(30)
        raise RuntimeError(
            "supported yardage/reception rows contain unresolved current player/team identity; "
            f"rows={int(strict_bad.sum())} sample={sample.to_dict('records')}"
        )

    non_player = model["player"].map(_is_non_player_entity)
    noncore_bad = identity_bad & ~strict_mask
    _append_quarantine(
        quarantine_parts,
        model.loc[noncore_bad & non_player],
        "NON_PLAYER_ENTITY",
    )
    _append_quarantine(
        quarantine_parts,
        model.loc[noncore_bad & ~non_player],
        "UNMODELED_NONCORE_PLAYER_IDENTITY",
    )
    model = model.loc[~noncore_bad].copy()

    # Every row that reaches PlayerForm/pricing must now be a real supported
    # player offer with complete current-slate identity.
    for col in ("event_id", "market", "player", "team_abbr", "opponent_abbr"):
        bad = _text(model[col]).eq("")
        if bad.any():
            raise RuntimeError(f"post-quarantine model-facing props contain blank {col}: rows={int(bad.sum())}")
    if model.duplicated().any():
        raise RuntimeError(f"model-facing compact props contain exact duplicate rows={int(model.duplicated().sum())}")

    # Confirm every strict market row present before filtering survived. Missing
    # market posting is allowed; dropping a posted strict row is not.
    for m in sorted(STRICT_PLAYER_MARKETS):
        before = int((_text(actual["market"]) == m).sum())
        after = int((_text(model["market"]) == m).sum())
        if after != before:
            raise RuntimeError(f"strict market row loss during model materialization market={m} before={before} after={after}")

    quarantine = (
        pd.concat(quarantine_parts, ignore_index=True, sort=False)
        if quarantine_parts
        else pd.DataFrame(columns=[*props.columns, "quarantine_reason"])
    )
    QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
    quarantine.to_csv(QUARANTINE, index=False)
    model.to_csv(PROPS, index=False)

    reasons = (
        quarantine["quarantine_reason"].value_counts().sort_index().astype(int).to_dict()
        if not quarantine.empty
        else {}
    )
    market_counts = _text(model["market"]).value_counts().sort_index().astype(int).to_dict()
    status["production_compact_disposition"] = "MODEL_LIVE_PROPS_READY"
    status["production_compact_rows"] = int(len(model))
    status["production_compact_unique_players"] = int(model["player"].nunique())
    status["production_quarantined_rows"] = int(len(quarantine))
    status["production_quarantine_reasons"] = reasons
    status["production_market_rows"] = market_counts
    status["strict_market_identity_failures"] = 0
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "MODEL_LIVE_PROPS_READY",
        "rows": int(len(model)),
        "unique_players": int(model["player"].nunique()),
        "quarantined_rows": int(len(quarantine)),
        "quarantine_reasons": reasons,
        "market_rows": market_counts,
        "strict_market_identity_failures": 0,
        "quarantine": str(QUARANTINE),
    }
    print("[model_live_props] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    materialize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
