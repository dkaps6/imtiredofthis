#!/usr/bin/env python3
"""Materialize the live sportsbook artifact that modeling is actually allowed to consume.

Provider fetches may contain explicit bookmaker-missing sentinel rows. Those are
useful provider diagnostics but are not players and must never enter PlayerForm,
provider readiness, metrics, or pricing. This boundary adapter quarantines those
rows and emits a canonical compact player schema with a compatibility ``player``
alias for existing model consumers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
STATUS = DATA / "live_odds_status.json"
PROPS = OUTPUTS / "props_raw.csv"
QUARANTINE = DATA / "live_odds_placeholder_rows.csv"


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def materialize() -> dict:
    if not STATUS.exists() or STATUS.stat().st_size <= 0:
        raise RuntimeError("live_odds_status.json missing before model-facing prop materialization")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if not bool(status.get("available")):
        result = {
            "disposition": "MODEL_LIVE_PROPS_NOT_REQUIRED",
            "status": status.get("status", "unknown"),
            "rows": 0,
            "placeholder_rows_quarantined": 0,
        }
        print("[model_live_props] " + json.dumps(result, sort_keys=True))
        return result

    if not PROPS.exists() or PROPS.stat().st_size <= 0:
        raise RuntimeError("live odds status says available but outputs/props_raw.csv is missing/empty")
    props = pd.read_csv(PROPS, low_memory=False)
    if props.empty:
        raise RuntimeError("live odds status says available but outputs/props_raw.csv has zero rows")
    props.columns = [str(c).strip().lower() for c in props.columns]

    missing_flag = pd.to_numeric(
        props.get("bookmaker_missing", pd.Series(0, index=props.index)), errors="coerce"
    ).fillna(0).eq(1)
    placeholders = props.loc[missing_flag].copy()
    actual = props.loc[~missing_flag].copy()
    if actual.empty:
        raise RuntimeError("all compact live prop rows are bookmaker-missing sentinels")

    pcol = next(
        (c for c in ("canonical_player_name", "player_canonical", "player", "book_player_name") if c in actual.columns),
        None,
    )
    if pcol is None:
        raise RuntimeError(f"model-facing live props have no player column; columns={list(actual.columns)}")
    player = _text(actual[pcol])
    if player.eq("").any():
        sample = actual.loc[player.eq(""), [c for c in ("event_id", "market", pcol) if c in actual.columns]].head(20)
        raise RuntimeError(
            "real model-facing prop rows contain blank canonical player identity; "
            f"rows={int(player.eq('').sum())} sample={sample.to_dict('records')}"
        )
    actual["player"] = player

    for col in ("event_id", "market", "team_abbr", "opponent_abbr"):
        if col not in actual.columns:
            raise RuntimeError(f"model-facing live props missing required column: {col}")
        bad = _text(actual[col]).eq("")
        if bad.any():
            sample = actual.loc[bad, [c for c in ("event_id", "market", "player", "team_abbr", "opponent_abbr") if c in actual.columns]].head(20)
            raise RuntimeError(
                f"real model-facing prop rows contain blank {col}; rows={int(bad.sum())} "
                f"sample={sample.to_dict('records')}"
            )

    if actual.duplicated().any():
        raise RuntimeError(f"model-facing compact props contain exact duplicate rows={int(actual.duplicated().sum())}")

    QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
    placeholders.to_csv(QUARANTINE, index=False)
    actual.to_csv(PROPS, index=False)

    status["production_compact_disposition"] = "MODEL_LIVE_PROPS_READY"
    status["production_compact_rows"] = int(len(actual))
    status["placeholder_rows_quarantined"] = int(len(placeholders))
    status["production_compact_unique_players"] = int(actual["player"].nunique())
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "MODEL_LIVE_PROPS_READY",
        "rows": int(len(actual)),
        "unique_players": int(actual["player"].nunique()),
        "placeholder_rows_quarantined": int(len(placeholders)),
        "quarantine": str(QUARANTINE),
    }
    print("[model_live_props] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    materialize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
