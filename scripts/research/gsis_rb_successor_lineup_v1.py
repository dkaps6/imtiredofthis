#!/usr/bin/env python3
"""Private-safe GSIS RB successor-lineup mechanics for prospective research.

Reads only Lineup Detail identity + Plays from an access-controlled GSIS
snapshot. Public audit output contains aggregate counts only. Exact candidate
rows, when requested, are written to a caller-supplied PRIVATE path and must
never be committed to the public repository.

No target outcomes or sportsbook inputs are consumed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

TEAM_MAP={"ARZ":"ARI","BLT":"BAL","CLV":"CLE","HST":"HOU","LA":"LAR"}
NAME_SUFFIXES={"JR","JR.","SR","SR.","II","III","IV","V"}

def team_key(v:Any)->str:
    x=str(v).strip().upper()
    return TEAM_MAP.get(x,x)

def player_key(v:Any)->str:
    text=re.sub(r"\b(?:Jr|Sr|II|III|IV|V)\.?$","",str(v),flags=re.I).strip()
    return "".join(ch.lower() for ch in text if ch.isalnum())

def split_lineup(v:Any)->list[str]:
    out=[]
    for p in [x.strip() for x in str(v).split(",") if x.strip()]:
        if p.upper() in NAME_SUFFIXES and out:
            out[-1]=f"{out[-1]}, {p}"
        else:
            out.append(p)
    return out

def number(v:Any)->float:
    s=str(v).strip().replace(",","").replace("%","")
    if s in {"","-","—","N/A"}: return float("nan")
    m=re.search(r"-?\d+(?:\.\d+)?",s)
    return float(m.group()) if m else float("nan")

def filter_value(record:dict[str,Any],fid:str)->str:
    for item in record.get("filters",[]):
        if item.get("id")==fid: return str(item.get("value",""))
    raise RuntimeError(f"missing GSIS filter {fid}")

def header_rows(table:dict[str,Any])->tuple[list[str],list[dict[str,Any]]]:
    rows=table.get("rows",[])
    for i,row in enumerate(rows):
        cells=row.get("cells",[])
        if sum(c.get("tag")=="th" for c in cells)>1:
            return [str(c.get("text","")).strip() for c in cells],rows[i+1:]
    raise RuntimeError("GSIS table has no multi-column header")

def load_snapshot(path:Path)->tuple[dict[str,Any],str]:
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    with gzip.open(path,"rt",encoding="utf-8") as f:
        obj=json.load(f)
    return obj,digest

def parse_offense_lineups(payload:dict[str,Any])->list[dict[str,Any]]:
    out=[]
    for record in payload.get("records",[]):
        if record.get("report")!="Lineup Detail" or str(record.get("mode",""))!="Offense":
            continue
        team=team_key(filter_value(record,"select2"))
        for table in record.get("tables",[]):
            headers,rows=header_rows(table)
            required={"Lineup","Plays"}
            missing=required-set(headers)
            if missing: raise RuntimeError(f"Lineup Detail missing fields: {sorted(missing)}")
            ix={h:headers.index(h) for h in required}
            for row in rows:
                cells=[str(c.get("text","")).strip() for c in row.get("cells",[])]
                if len(cells)!=len(headers): continue
                names=split_lineup(cells[ix["Lineup"]])
                plays=number(cells[ix["Plays"]])
                if len(names)!=11 or not math.isfinite(plays) or plays<=0: continue
                keys=frozenset(player_key(n) for n in names if player_key(n))
                if len(keys)!=11:
                    # duplicate/ambiguous normalized identity inside a lineup
                    continue
                out.append({"team":team,"players":keys,"plays":float(plays)})
    return out

def successor_weights(
    rows:list[dict[str,Any]], *,
    team:str,
    unavailable:set[str],
    successors:list[str],
)->dict[str,float]:
    t=team_key(team)
    unavailable={player_key(x) for x in unavailable if player_key(x)}
    successors=[player_key(x) for x in successors if player_key(x)]
    if not unavailable or not successors:
        return {}
    exposure={s:0.0 for s in successors}
    for row in rows:
        if row["team"]!=t: continue
        players=row["players"]
        if any(u in players for u in unavailable):
            continue
        for s in successors:
            if s in players:
                exposure[s]+=float(row["plays"])
    den=sum(exposure.values())
    if den<=0:
        return {}
    return {s:exposure[s]/den for s in successors if exposure[s]>0}

def build_private_candidate(
    rows:list[dict[str,Any]],
    vacancy:pd.DataFrame,
    successor_pool:pd.DataFrame|None=None,
)->tuple[pd.DataFrame,dict[str,Any]]:
    required={
      "target_season","target_week","team","successor_player_clean_key",
      "vacated_rush_share","unavailable_players"
    }
    miss=sorted(required-set(vacancy.columns))
    if miss: raise RuntimeError(f"vacancy state missing columns: {miss}")
    key=["target_season","target_week","team"]
    pool=None
    if successor_pool is not None:
        pool=successor_pool.copy()
        pool.columns=[str(x).strip().lower() for x in pool.columns]
        need_pool={*key,"successor_player_clean_key"}
        missing=need_pool-set(pool.columns)
        if missing:
            raise RuntimeError(f"successor pool missing columns: {sorted(missing)}")
        pool["team"]=pool["team"].map(team_key)
        pool["target_season"]=pd.to_numeric(pool["target_season"],errors="raise").astype(int)
        pool["target_week"]=pd.to_numeric(pool["target_week"],errors="raise").astype(int)
        pool["successor_player_clean_key"]=pool["successor_player_clean_key"].astype(str).map(player_key)
        if pool.duplicated([*key,"successor_player_clean_key"]).any():
            raise RuntimeError("duplicate explicit successor-pool identity")
    private=[]
    event_audits=[]
    for ident,g in vacancy.groupby(key,sort=True):
        season,week,team=ident
        vacated=pd.to_numeric(g["vacated_rush_share"],errors="raise").astype(float)
        if float(vacated.max()-vacated.min())>1e-12:
            raise RuntimeError(f"vacated share inconsistent within event {ident}")
        V=float(vacated.iloc[0])
        unavailable=set()
        for value in g["unavailable_players"].astype(str):
            unavailable.update(
                x.strip()
                for x in re.split(r"[|,]", value)
                if x.strip()
            )
        if pool is None:
            successors=g["successor_player_clean_key"].astype(str).tolist()
        else:
            q=pool.loc[
                pool.target_season.eq(int(season))
                & pool.target_week.eq(int(week))
                & pool.team.eq(team_key(team))
            ]
            successors=q["successor_player_clean_key"].astype(str).tolist()
            snap_successors=set(g["successor_player_clean_key"].astype(str).map(player_key))
            active_successors=set(successors)
            if not snap_successors.issubset(active_successors):
                missing=sorted(snap_successors-active_successors)
                raise RuntimeError(f"Vacancy V1 successors absent from active successor pool: {missing}")
        w=successor_weights(rows,team=team,unavailable=unavailable,successors=successors)
        if not w:
            event_audits.append({"status":"NO_GSIS_SUCCESSOR_EXPOSURE","successors":len(successors)})
            continue
        for s,weight in w.items():
            private.append({
              "target_season":int(season),"target_week":int(week),"team":str(team),
              "successor_player_clean_key":s,
              "vacated_rush_share":V,
              "gsis_successor_weight":float(weight),
              "gsis_transfer_rush_share":float(V*weight),
            })
        gap=abs(sum(V*x for x in w.values())-V)
        event_audits.append({"status":"CANDIDATE_READY","successors":len(w),"conservation_gap":gap})
    out=pd.DataFrame(private)
    ready=[x for x in event_audits if x["status"]=="CANDIDATE_READY"]
    audit={
      "events_seen":len(event_audits),
      "events_candidate_ready":len(ready),
      "events_no_gsis_successor_exposure":sum(x["status"]=="NO_GSIS_SUCCESSOR_EXPOSURE" for x in event_audits),
      "candidate_rows":len(out),
      "max_conservation_gap":max([x.get("conservation_gap",0.0) for x in ready] or [0.0]),
      "target_outcomes_read":False,
      "sportsbook_inputs_read":False,
      "raw_lineups_emitted_publicly":False,
      "successor_pool_mode":"EXPLICIT_ACTIVE_POOL" if pool is not None else "VACANCY_STATE_ONLY_SCHEMA_COMPAT",
    }
    return out,audit

def public_schema_audit(payload:dict[str,Any],digest:str,rows:list[dict[str,Any]])->dict[str,Any]:
    by_team=defaultdict(int)
    plays_by_team=defaultdict(float)
    for r in rows:
        by_team[r["team"]]+=1
        plays_by_team[r["team"]]+=r["plays"]
    return {
      "audit_version":"GSIS_RB_SUCCESSOR_LINEUP_V1_SCHEMA_AUDIT",
      "snapshot_sha256":digest,
      "snapshot_season":payload.get("season"),
      "snapshot_phase":payload.get("phase"),
      "offense_lineup_rows_parsed":len(rows),
      "teams_with_offense_lineups":len(by_team),
      "min_lineup_rows_per_team":min(by_team.values()) if by_team else 0,
      "max_lineup_rows_per_team":max(by_team.values()) if by_team else 0,
      "teams_with_positive_lineup_plays":sum(v>0 for v in plays_by_team.values()),
      "target_outcomes_read":False,
      "sportsbook_inputs_read":False,
      "raw_lineups_emitted_publicly":False,
      "player_identifiers_emitted_publicly":False,
    }

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",type=Path,required=True)
    ap.add_argument("--public-audit",type=Path,required=True)
    ap.add_argument("--vacancy-state",type=Path)
    ap.add_argument("--private-candidate-out",type=Path)
    ap.add_argument("--successor-pool",type=Path)
    a=ap.parse_args()
    payload,digest=load_snapshot(a.snapshot)
    rows=parse_offense_lineups(payload)
    audit=public_schema_audit(payload,digest,rows)
    if a.vacancy_state:
        vacancy=pd.read_csv(a.vacancy_state,low_memory=False)
        successor_pool=pd.read_csv(a.successor_pool,low_memory=False) if a.successor_pool else None
        candidate,cand_audit=build_private_candidate(rows,vacancy,successor_pool=successor_pool)
        audit["candidate_mechanics"]=cand_audit
        if a.private_candidate_out:
            a.private_candidate_out.parent.mkdir(parents=True,exist_ok=True)
            candidate.to_csv(a.private_candidate_out,index=False)
        elif len(candidate):
            raise RuntimeError("candidate rows exist but no private output path was supplied")
    a.public_audit.parent.mkdir(parents=True,exist_ok=True)
    a.public_audit.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(audit,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
