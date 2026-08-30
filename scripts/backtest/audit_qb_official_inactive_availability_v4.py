#!/usr/bin/env python3
"""M78 corrected official-inactive source reconstruction.

Source/data-contract only. No predictive model. This version does not treat the
superseded Run #3 compact snapshot as truth; it rebuilds exact identities from
NFL.com, uses weekly rosters to disambiguate candidate sections, and emits a
new corrected candidate snapshot that must pass strict structural gates before
it can be frozen for M79.
"""
from __future__ import annotations

import argparse, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup, Tag

from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import audit_qb_official_inactive_availability_v3 as v3

SEASONS=(2024,2025)
EXT_POS={"QB","RB","FB","HB","WR","TE","OL","OT","T","LT","RT","OG","G","LG","RG","C","DL","DT","NT","DE","EDGE","OLB","LB","ILB","MLB","CB","DB","S","FS","SS","K","P","LS","KR","PR"}


def _dedupe(xs):
    out=[]; seen=set()
    for x in xs:
        k=m78.norm_space(x)
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out


def candidate_bullets(ul: Tag):
    vals=[]
    for li in ul.find_all("li"):
        t=m78.norm_space(li.get_text(" ",strip=True))
        if not t or t.upper().startswith(("WHERE:","WHEN:","TV:","WATCH:")):
            continue
        vals.append(t)
    return _dedupe(vals)


def flex_identity(raw: str):
    s=m78.norm_space(raw)
    parts=s.split()
    if len(parts)<2:
        return None
    tok=parts[0].upper().rstrip(".:")
    seg=tok.split("/")
    recognized = tok in EXT_POS or (len(seg)>1 and all(x in EXT_POS for x in seg))
    looks_like_unknown_pos = bool(re.fullmatch(r"[A-Z]{1,5}(?:/[A-Z]{1,5})*", parts[0]))
    if recognized or looks_like_unknown_pos:
        pos=tok
        name=m78.norm_space(" ".join(parts[1:]))
        pos_valid=recognized
    else:
        pos="UNK"
        name=s
        pos_valid=False
    name=re.sub(r"\s*[-–—]\s*[A-Z]$","",name)
    key=m78.norm_name(name)
    if len(key)<3:
        return None
    return pos,name,key,pos_valid


def team_sections(soup: BeautifulSoup):
    found=[]; seen=set()
    labels=list(soup.find_all(["h2","h3","h4"]))+list(soup.find_all(True))
    for label in labels:
        team=m78.team_from_heading(label.get_text(" ",strip=True))
        if not team: continue
        ul=label.find_next("ul")
        if ul is None: continue
        key=(team,id(ul))
        if key in seen: continue
        blocked=False
        for el in label.next_elements:
            if el is ul: break
            if isinstance(el,Tag) and el is not label:
                other=m78.team_from_heading(el.get_text(" ",strip=True))
                if other and other!=team:
                    blocked=True; break
        if blocked: continue
        bullets=candidate_bullets(ul)
        if len(bullets)<3: continue
        seen.add(key); found.append((team,ul,bullets))
    return found


def parse_article(url,season,snapshots,roster_idx):
    try: r=m78.request(url)
    except Exception as exc:
        snapshots.append({"kind":"inactive_article_v4","season":season,"url":url,"status":0,"sha256":"","error":f"{type(exc).__name__}:{exc}"})
        return [],[]
    snapshots.append({"kind":"inactive_article_v4","season":season,"url":r.url,"status":r.status_code,"sha256":m78.sha256_bytes(r.content),"error":""})
    soup=BeautifulSoup(r.text,"html.parser")
    h1=soup.find("h1")
    title=m78.norm_space(h1.get_text(" ",strip=True) if h1 else soup.title.get_text(" ",strip=True) if soup.title else "")
    week=m78.extract_week(title)
    if week is None: return [],[]
    records=[]; sections=[]
    for i,(team,ul,bullets) in enumerate(team_sections(soup)):
        sid=f"{season}:{week}:{team}:{hashlib.sha1((r.url+'|'+str(i)).encode()).hexdigest()[:10]}"
        parsed=[flex_identity(x) for x in bullets]
        identity_complete=all(x is not None for x in parsed)
        rr=[]
        for raw,p in zip(bullets,parsed):
            if p is None: continue
            pos,name,key,pos_valid=p
            match=key in roster_idx.get((week,team),set())
            rr.append({"section_id":sid,"season":season,"week":week,"team":team,"raw_bullet":raw,"inactive_name":name,"inactive_name_key":key,"listed_position":pos,"position_valid":pos_valid,"roster_identity_match":match,"article_url":r.url})
        unique_keys={x["inactive_name_key"] for x in rr}
        no_dup=len(unique_keys)==len(rr)==len(bullets)
        bridge=(sum(x["roster_identity_match"] for x in rr)/len(rr)) if rr else 0.0
        posrate=(sum(x["position_valid"] for x in rr)/len(bullets)) if bullets else 0.0
        sections.append({"section_id":sid,"season":season,"week":week,"team":team,"article_url":r.url,"article_title":title,"candidate_bullets":len(bullets),"identity_rows":len(rr),"identity_complete":identity_complete and no_dup,"position_parse_rate":posrate,"roster_bridge_rate":bridge,"reasonable_count":3<=len(bullets)<=8})
        records.extend(rr)
    return records,sections


def choose_sections(sections: pd.DataFrame):
    if sections.empty: return sections
    q=sections.copy()
    q["rank_complete"]=q.identity_complete.astype(int)
    q["rank_reasonable"]=q.reasonable_count.astype(int)
    q=q.sort_values(["season","week","team","rank_complete","rank_reasonable","roster_bridge_rate","position_parse_rate","candidate_bullets"],ascending=[True,True,True,False,False,False,False,False])
    return q.drop_duplicates(["season","week","team"],keep="first").copy()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--canonical',required=True); ap.add_argument('--out-dir',required=True); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    base=m78.require_canonical(Path(args.canonical)); targets=m78.canonical_targets(base)
    snapshots=[]
    roster_idx={s:m78.roster_name_index(m78.load_weekly_rosters(s,snapshots),s) for s in SEASONS}
    allr=[]; alls=[]
    for s in SEASONS:
        for url in m78.discover_article_urls(s,snapshots):
            r,sec=parse_article(url,s,snapshots,roster_idx[s]); allr+=r; alls+=sec
    records=pd.DataFrame(allr); sections=pd.DataFrame(alls)
    selected=choose_sections(sections)
    schedule=v3.load_schedule(snapshots)
    sched= schedule.merge(selected,on=['season','week','team'],how='left',validate='one_to_one')
    for c in ['identity_complete','reasonable_count']:
        sched[c]=sched[c].fillna(False).astype(bool)
    for c in ['candidate_bullets','identity_rows']:
        sched[c]=sched[c].fillna(0).astype(int)
    for c in ['position_parse_rate','roster_bridge_rate']:
        sched[c]=sched[c].fillna(0.0).astype(float)
    chosen_ids=set(selected.section_id.astype(str)) if len(selected) else set()
    chosen_records=records[records.section_id.astype(str).isin(chosen_ids)].copy()

    compact=[]
    for row in selected.itertuples(index=False):
        g=chosen_records[chosen_records.section_id.eq(row.section_id)].drop_duplicates('inactive_name_key')
        toks=sorted(f"{x.inactive_name_key}:{x.listed_position}:{1 if bool(x.roster_identity_match) else 0}" for x in g.itertuples(index=False))
        compact.append({'season':int(row.season),'week':int(row.week),'team':str(row.team),'inactive_tokens':'|'.join(toks),'inactive_count':len(toks)})
    compact=pd.DataFrame(compact).sort_values(['season','week','team']).reset_index(drop=True)
    compact_path=out/'m78_corrected_candidate_teamweek.csv'; compact.to_csv(compact_path,index=False)
    compact_sha=hashlib.sha256(compact_path.read_bytes()).hexdigest()

    gates=[]
    def gate(name,val,thr,passed,scope='historical_m79'): gates.append({'gate':name,'value':float(val),'threshold':thr,'passed':bool(passed),'scope':scope})
    for s in SEASONS:
        sq=sched[sched.season.eq(s)]
        cq=targets[targets.season.eq(s)].merge(selected[['season','week','team','identity_complete']],on=['season','week','team'],how='left')
        cq['identity_complete']=cq.identity_complete.fillna(False)
        gate(f'schedule_team_week_source_coverage_{s}',float(sq.section_id.notna().mean()),'==1.0',bool(sq.section_id.notna().all()))
        gate(f'canonical_identity_complete_coverage_{s}',float(cq.identity_complete.mean()),'==1.0',bool(cq.identity_complete.all()))
        gate(f'selected_reasonable_count_{s}',float(sq.reasonable_count.mean()),'==1.0',bool(sq.reasonable_count.all()))
        pos_num=chosen_records.loc[chosen_records.season.eq(s),'position_valid'].sum(); pos_den=len(chosen_records.loc[chosen_records.season.eq(s)])
        bridge=float(chosen_records.loc[chosen_records.season.eq(s),'roster_identity_match'].mean()) if pos_den else 0.0
        posrate=float(pos_num/pos_den) if pos_den else 0.0
        gate(f'candidate_bullet_position_parse_{s}',posrate,'>=0.95',posrate>=.95)
        gate(f'roster_identity_bridge_{s}',bridge,'>=0.90',bridge>=.90)
        minbridge=float(sq.roster_bridge_rate.min()) if len(sq) else 0.0
        gate(f'min_selected_section_roster_bridge_{s}',minbridge,'>=0.50',minbridge>=.50)
        weeks=int(sq.loc[sq.identity_complete,'week'].nunique())
        gate(f'regular_week_identity_complete_{s}',weeks,'==18',weeks==18)
        for window,wq in sq.groupby('game_window'):
            val=float(wq.identity_complete.mean()) if len(wq) else 0.0
            gate(f'window_identity_complete_{s}_{window}',val,'==1.0',bool(wq.identity_complete.all()))
    live_reach,live_valid,live_teams,live_players,live_detail=v3.live_endpoint_runtime_status(snapshots)
    gate('live_2026_endpoint_reachable',1 if live_reach else 0,'==1',live_reach,'live_runtime')
    gate('live_2026_game_day_payload_validated',1 if live_valid else 0,'==1_before_production_use',live_valid,'live_runtime')
    gdf=pd.DataFrame(gates)
    authorized=bool(gdf.loc[gdf.scope.eq('historical_m79'),'passed'].all())
    interp=pd.DataFrame([{'migration':'M78','status':'CORRECTED_SOURCE_CONTRACT_QUALIFIED' if authorized else 'CORRECTED_SOURCE_CONTRACT_NOT_QUALIFIED','predictive_model_fit':False,'production_actionable':False,'canonical_rows':len(base),'schedule_team_weeks':len(schedule),'corrected_candidate_team_weeks':len(compact),'corrected_candidate_sha256':compact_sha,'historical_m79_authorized':authorized,'superseded_run3_snapshot_is_authority':False,'live_2026_endpoint_reachable':live_reach,'live_2026_payload_validated':live_valid,'next_step':'freeze_corrected_snapshot_then_M79' if authorized else 'repair_source_before_M79'}])
    contract={'as_of_utc':datetime.now(timezone.utc).isoformat(),'canonical_sha256':m78.CANONICAL_SHA256,'sportsbook_used':False,'target_game_performance_used':False,'predictive_model_fit':False,'superseded_run3_snapshot_is_authority':False,'corrected_candidate_sha256':compact_sha,'historical_m79_authorized':authorized,'live_2026_endpoint_reachable':live_reach,'live_2026_payload_validated':live_valid}
    sections.to_csv(out/'m78_v4_all_sections.csv',index=False); selected.to_csv(out/'m78_v4_selected_sections.csv',index=False); chosen_records.to_csv(out/'m78_v4_selected_records.csv',index=False); sched.to_csv(out/'m78_v4_schedule_coverage.csv',index=False); pd.DataFrame(snapshots).to_csv(out/'m78_v4_source_snapshots.csv',index=False); gdf.to_csv(out/'m78_source_gate.csv',index=False); interp.to_csv(out/'m78_interpretation.csv',index=False); (out/'m78_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print('=== M78 V4 INTERPRETATION ==='); print(interp.to_string(index=False)); print('\n=== GATES ==='); print(gdf.to_string(index=False)); print('\nCORRECTED CANDIDATE SHA',compact_sha)
    return 0

if __name__=='__main__': raise SystemExit(main())
