#!/usr/bin/env python3
"""Outcome-free qualification evidence for episodic football regime events."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS=["team_change_flag","joint_player_room_transition_flag","target_room_churn_flag","rush_room_churn_flag"]

def build_event_qualification(detail: pd.DataFrame) -> pd.DataFrame:
    x=detail.copy(); x.columns=[str(c).strip().lower() for c in x.columns]
    req={"season","week","position","player_identity_key","known_context_flag",*EVENTS}
    miss=req-set(x.columns)
    if miss: raise RuntimeError(f"missing columns: {sorted(miss)}")
    if x.duplicated(["season","week","player_identity_key"]).any():
        raise RuntimeError("duplicate player-period keys")
    rows=[]
    for event in EVENTS:
        for pos,g0 in x.groupby("position",dropna=False):
            g=g0[g0.known_context_flag.astype(bool)].copy()
            if g.empty: continue
            e=g[event].astype(bool)
            positives=g[e]
            counts=positives.groupby("season").size()
            # Onset: event is true now and was not true in the player's immediately previous
            # observed row in the same season. This detects accidental forward-fill persistence.
            s=g.sort_values(["player_identity_key","season","week"])
            prev=s.groupby(["player_identity_key","season"])[event].shift(1).fillna(False).astype(bool)
            onset=s[event].astype(bool) & ~prev
            positive_n=int(e.sum()); onset_n=int(onset.sum())
            rows.append({
                "event_name":event,"position":pos,"known_rows":len(g),"positive_events":positive_n,
                "prevalence":float(e.mean()),"seasons_with_positive":int((counts>0).sum()),
                "max_season_share":float(counts.max()/positive_n) if positive_n else np.nan,
                "event_onsets":onset_n,"onset_fraction_of_positive":float(onset_n/positive_n) if positive_n else np.nan,
                "support_gate_250":bool(positive_n>=250 and (counts>0).sum()>=3 and (counts.max()/positive_n if positive_n else 1)>0 and (counts.max()/positive_n if positive_n else 1)<=0.50),
                "known_coverage_gate_080":bool(len(g)/len(g0)>=0.80),
            })
    return pd.DataFrame(rows).sort_values(["event_name","position"]).reset_index(drop=True)

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--detail",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    out=build_event_qualification(pd.read_csv(a.detail)); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); print(out.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
