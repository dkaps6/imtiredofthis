#!/usr/bin/env python3
"""Materialize the live sportsbook artifact that modeling is actually allowed to consume.

Provider outputs are a superset of production model capability. This boundary
adapter never changes a line or odds value. It explicitly quarantines provider
sentinels, unsupported markets, and non-player/unmodeled anytime-TD entities,
while keeping all supported yardage/reception player markets fail-closed on
current player/team/opponent identity.

The operation is deliberately idempotent. Once a live snapshot has been
materialized, later PlayerForm calls validate and reuse that exact compact layer
instead of filtering it again and erasing the original quarantine evidence.
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
COMPACT = OUTPUTS / "props_raw_compact.csv"
QUARANTINE = DATA / "live_odds_placeholder_rows.csv"

SUPPORTED_MARKETS = {
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
    "player_anytime_td",
}

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
    return bool(re.search(r"(?:\bd/st\b|\bdefense\b|\bdefence\b|\bspecial teams\b)", x))


def _append_quarantine(parts: list[pd.DataFrame], frame: pd.DataFrame, reason: str) -> None:
    if frame.empty:
        return
    x = frame.copy()
    x["quarantine_reason"] = reason
    parts.append(x)


def _read_props(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"required model-facing live props missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"required model-facing live props has zero rows: {path}")
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _validate_materialized(props: pd.DataFrame, status: dict) -> dict:
    """Validate and reuse an already materialized compact snapshot."""
    expected_rows = int(status.get("production_compact_rows", -1))
    if expected_rows <= 0 or len(props) != expected_rows:
        raise RuntimeError(
            "existing compact live-prop materialization row count drifted; "
            f"status={expected_rows} actual={len(props)}"
        )
    required = {"event_id", "market", "player", "team_abbr", "opponent_abbr", "offers_json"}
    missing = required - set(props.columns)
    if missing:
        raise RuntimeError(f"existing compact live props missing columns: {sorted(missing)}")
    for col in ("event_id", "market", "player", "team_abbr", "opponent_abbr"):
        bad = _text(props[col]).eq("")
        if bad.any():
            raise RuntimeError(f"existing compact live props contain blank {col}: rows={int(bad.sum())}")
    if props.duplicated().any():
        raise RuntimeError(f"existing compact live props contain exact duplicates={int(props.duplicated().sum())}")
    unsupported = ~_text(props["market"]).isin(SUPPORTED_MARKETS)
    if unsupported.any():
        raise RuntimeError(
            "existing compact live props contain unsupported markets: "
            f"{sorted(_text(props.loc[unsupported, 'market']).unique().tolist())}"
        )

    expected_quarantine = int(status.get("production_quarantined_rows", 0))
    expected_reasons = {
        str(k): int(v) for k, v in dict(status.get("production_quarantine_reasons", {})).items()
    }
    if expected_quarantine > 0:
        if not QUARANTINE.exists() or QUARANTINE.stat().st_size <= 0:
            raise RuntimeError(
                "live-prop status records quarantined rows but immutable quarantine artifact is missing"
            )
        q = pd.read_csv(QUARANTINE, low_memory=False)
        if len(q) != expected_quarantine:
            raise RuntimeError(
                "live-prop quarantine evidence row count drifted; "
                f"status={expected_quarantine} artifact={len(q)}"
            )
        if "quarantine_reason" not in q.columns:
            raise RuntimeError("live-prop quarantine evidence missing quarantine_reason")
        reasons = q["quarantine_reason"].astype(str).value_counts().sort_index().astype(int).to_dict()
        if reasons != expected_reasons:
            raise RuntimeError(
                "live-prop quarantine reason counts drifted; "
                f"status={expected_reasons} artifact={reasons}"
            )
    elif QUARANTINE.exists() and QUARANTINE.stat().st_size > 0:
        q = pd.read_csv(QUARANTINE, low_memory=False)
        if not q.empty:
            raise RuntimeError(
                "live-prop status says zero quarantined rows but quarantine artifact is non-empty"
            )

    if not COMPACT.exists() or COMPACT.stat().st_size <= 0:
        props.to_csv(COMPACT, index=False)
    else:
        compact = _read_props(COMPACT)
        if len(compact) != len(props) or list(compact.columns) != list(props.columns):
            raise RuntimeError("immutable compact live-prop snapshot does not match model-facing compact layer")
        left = compact.fillna("").astype(str).reset_index(drop=True)
        right = props.fillna("").astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise RuntimeError("immutable compact live-prop snapshot content drifted")

    market_counts = _text(props["market"]).value_counts().sort_index().astype(int).to_dict()
    result = {
        "disposition": "MODEL_LIVE_PROPS_READY",
        "rows": int(len(props)),
        "unique_players": int(props["player"].nunique()),
        "quarantined_rows": expected_quarantine,
        "quarantine_reasons": expected_reasons,
        "market_rows": market_counts,
        "strict_market_identity_failures": int(status.get("strict_market_identity_failures", 0)),
        "quarantine": str(QUARANTINE),
        "idempotent_reuse": True,
    }
    print("[model_live_props] " + json.dumps(result, sort_keys=True))
    return result


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

    props = _read_props(PROPS)
    if status.get("production_compact_disposition") == "MODEL_LIVE_PROPS_READY":
        return _validate_materialized(props, status)

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
    _append_quarantine(quarantine_parts, model.loc[noncore_bad & non_player], "NON_PLAYER_ENTITY")
    _append_quarantine(
        quarantine_parts,
        model.loc[noncore_bad & ~non_player],
        "UNMODELED_NONCORE_PLAYER_IDENTITY",
    )
    model = model.loc[~noncore_bad].copy()

    for col in ("event_id", "market", "player", "team_abbr", "opponent_abbr", "offers_json"):
        if col not in model.columns:
            raise RuntimeError(f"post-quarantine model-facing props missing {col}")
        bad = _text(model[col]).eq("")
        if bad.any():
            raise RuntimeError(f"post-quarantine model-facing props contain blank {col}: rows={int(bad.sum())}")
    if model.duplicated().any():
        raise RuntimeError(f"model-facing compact props contain exact duplicate rows={int(model.duplicated().sum())}")
    compact_key = ["event_id", "player", "market", "team_abbr", "opponent_abbr"]
    dup_key = model.duplicated(compact_key, keep=False)
    if dup_key.any():
        sample = model.loc[dup_key, compact_key].head(20).to_dict("records")
        raise RuntimeError(f"model-facing compact props are not unique at player/market grain: {sample}")

    for m in sorted(STRICT_PLAYER_MARKETS):
        before = int((_text(actual["market"]) == m).sum())
        after = int((_text(model["market"]) == m).sum())
        if after != before:
            raise RuntimeError(
                f"strict market row loss during model materialization market={m} before={before} after={after}"
            )

    quarantine = (
        pd.concat(quarantine_parts, ignore_index=True, sort=False)
        if quarantine_parts
        else pd.DataFrame(columns=[*props.columns, "quarantine_reason"])
    )
    QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
    quarantine.to_csv(QUARANTINE, index=False)
    model.to_csv(PROPS, index=False)
    model.to_csv(COMPACT, index=False)

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
    status["production_compact_artifact"] = str(COMPACT)
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
        "compact_artifact": str(COMPACT),
        "idempotent_reuse": False,
    }
    print("[model_live_props] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    materialize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
