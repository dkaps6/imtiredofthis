#!/usr/bin/env python3
"""Reconcile a preserved paid compact sportsbook snapshot to current PlayerForm.

The paid source artifact is immutable evidence.  A canonical replay may rebuild
football availability/PlayerForm later than the paid acquisition, so non-core
sportsbook entities can become absent from the current certified roster.  This
adapter deterministically quarantines only those non-core rows while keeping all
strict yardage/reception markets fail-closed.

It never changes lines/odds and never performs any provider request.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import player_name_key

DATA=Path("data")
OUTPUTS=Path("outputs")
STATUS=DATA/"live_odds_status.json"
FORM=DATA/"player_form.csv"
COMPACT=OUTPUTS/"props_raw_compact.csv"
MODEL_PROPS=OUTPUTS/"props_raw.csv"
QUARANTINE=DATA/"live_odds_placeholder_rows.csv"
AUDIT=DATA/"preserved_live_props_current_roster_reconciliation.json"

STRICT_MARKETS={
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
}


def _read(path:Path)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"required replay artifact missing/empty: {path}")
    df=pd.read_csv(path,low_memory=False)
    if df.empty:
        raise RuntimeError(f"required replay artifact has zero rows: {path}")
    return df


def _roster_keys()->set[tuple[str,str]]:
    f=_read(FORM)
    if not {"team","player"}.issubset(f.columns):
        raise RuntimeError("player_form missing team/player for preserved sportsbook reconciliation")
    return {
        (canon_team(t),player_name_key(p,strip_suffix=True))
        for t,p in zip(f["team"],f["player"])
        if canon_team(t) and player_name_key(p,strip_suffix=True)
    }


def reconcile()->dict:
    status=json.loads(STATUS.read_text(encoding="utf-8"))
    compact=_read(COMPACT)
    if not MODEL_PROPS.exists() or MODEL_PROPS.stat().st_size<=0:
        raise RuntimeError("outputs/props_raw.csv missing before preserved replay reconciliation")

    required={"market","player","team_abbr"}
    missing=required-set(compact.columns)
    if missing:
        raise RuntimeError(f"preserved compact props missing columns: {sorted(missing)}")

    roster=_roster_keys()
    keys=[
        (canon_team(t),player_name_key(p,strip_suffix=True))
        for t,p in zip(compact["team_abbr"],compact["player"])
    ]
    unrostered=pd.Series([k not in roster for k in keys],index=compact.index)
    markets=compact["market"].astype(str)
    strict=markets.isin(STRICT_MARKETS)

    strict_bad=unrostered & strict
    if strict_bad.any():
        sample=compact.loc[strict_bad,["player","team_abbr","market"]].head(30).to_dict("records")
        raise RuntimeError(
            "preserved replay contains strict-market players absent from current PlayerForm; "
            f"rows={int(strict_bad.sum())} sample={sample}"
        )

    removable=unrostered & ~strict
    removed=compact.loc[removable].copy()
    kept=compact.loc[~removable].copy().reset_index(drop=True)

    qcols=list(compact.columns)+["quarantine_reason"]
    if QUARANTINE.exists() and QUARANTINE.stat().st_size>0:
        q=pd.read_csv(QUARANTINE,low_memory=False)
    else:
        q=pd.DataFrame(columns=qcols)
    if not removed.empty:
        add=removed.copy()
        add["quarantine_reason"]="UNROSTERED_NONCORE_PLAYER"
        q=pd.concat([q,add],ignore_index=True,sort=False)

    kept.to_csv(COMPACT,index=False)
    kept.to_csv(MODEL_PROPS,index=False)
    q.to_csv(QUARANTINE,index=False)

    reasons=(
        q["quarantine_reason"].astype(str).value_counts().sort_index().astype(int).to_dict()
        if not q.empty and "quarantine_reason" in q.columns else {}
    )
    market_counts=kept["market"].astype(str).value_counts().sort_index().astype(int).to_dict()

    status["production_compact_disposition"]="MODEL_LIVE_PROPS_READY"
    status["production_compact_rows"]=int(len(kept))
    status["production_compact_unique_players"]=int(kept["player"].nunique())
    status["production_quarantined_rows"]=int(len(q))
    status["production_quarantine_reasons"]=reasons
    status["production_market_rows"]=market_counts
    status["preserved_replay_current_roster_reconciled"]=True
    status["preserved_replay_noncore_rows_quarantined"]=int(len(removed))
    STATUS.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    result={
        "disposition":"PRESERVED_LIVE_PROPS_RECONCILED_TO_CURRENT_ROSTER",
        "input_compact_rows":int(len(compact)),
        "output_compact_rows":int(len(kept)),
        "rows_quarantined":int(len(removed)),
        "quarantined_players":sorted(
            {f"{canon_team(t)}:{p}" for t,p in zip(removed.get("team_abbr",[]),removed.get("player",[]))}
        ),
        "strict_market_rows_removed":0,
        "model_facing_row_set_changed":bool(len(removed)),
        "paid_source_artifact_mutated":False,
        "lines_or_odds_changed":False,
        "provider_requests_used":False,
    }
    AUDIT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("[preserved_live_props_current_roster] "+json.dumps(result,sort_keys=True))
    return result


def main()->int:
    reconcile()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
