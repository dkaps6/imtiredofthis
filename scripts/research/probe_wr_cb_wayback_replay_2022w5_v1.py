#!/usr/bin/env python3
"""Probe exact Wayback replay modes for one known 2022-W5 availability record.

Metadata/structure only. Does not parse football rows, use outcomes, or save raw
page text. Purpose: determine whether exact timestamp replay is independently
servable when CDX digest/raw id_ access is unavailable.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests

ARTICLE=("https://www.fantasyalarm.com/articles/nfl/wide-receivers/"
         "2022-fantasy-football-wr-cb-match-up-report-week-5-tyreek-hill-to-burn-the-jets-in-week-5/134887")
TS="20221005195142"
EXPECTED=datetime.strptime(TS,"%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0 exact-memento-probe"}

def probe(url):
    try:
        r=requests.get(url,headers=HEADERS,timeout=22,allow_redirects=True)
        final=str(r.url)
        path=urlparse(final).path
        body=r.content
        low=body[:2_000_000].lower()
        return {
          "requested_url":url,"status_code":r.status_code,"final_url":final,
          "final_url_exact_timestamp":f"/web/{TS}" in path,
          "memento_datetime":str(r.headers.get("Memento-Datetime") or ""),
          "content_location":str(r.headers.get("Content-Location") or ""),
          "content_type":str(r.headers.get("Content-Type") or ""),
          "content_length":len(body),
          "body_sha256":hashlib.sha256(body).hexdigest() if r.status_code==200 else "",
          "contains_wr_cb_terms":(
            b"wide receiver" in low and b"cornerback" in low and b"matchup" in low
          ) if r.status_code==200 else False,
        }
    except requests.RequestException as exc:
        return {"requested_url":url,"error":"NETWORK_"+type(exc).__name__}

def run(out_dir):
    modes={
      "identity_raw":f"https://web.archive.org/web/{TS}id_/{ARTICLE}",
      "iframe":f"https://web.archive.org/web/{TS}if_/{ARTICLE}",
      "normal":f"https://web.archive.org/web/{TS}/{ARTICLE}",
      "availability_returned":f"http://web.archive.org/web/{TS}/{ARTICLE}",
    }
    rows={name:probe(url) for name,url in modes.items()}
    exact=[
      name for name,x in rows.items()
      if x.get("status_code")==200
      and x.get("final_url_exact_timestamp")
      and x.get("contains_wr_cb_terms")
    ]
    result={
      "contract":"WR_CB_2022W5_WAYBACK_REPLAY_MODE_PROBE_V1",
      "source_url":ARTICLE,"timestamp":TS,
      "exact_timestamp_content_modes":exact,
      "has_exact_timestamp_content_candidate":bool(exact),
      "raw_page_saved":False,"football_rows_parsed":0,
      "target_game_outcomes":False,"sportsbook_inputs":False,"parameters_fit":0,
      "source_model_gate_cleared":False,"modes":rows,
      "note":"200 exact timestamp is replay-provenance candidate, not source-ready without separate strict parse/identity verification.",
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"wayback_2022w5_replay_mode_probe.json").write_text(
      json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();run(a.out_dir)
