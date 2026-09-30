#!/usr/bin/env python3
"""Exact ONE public 2024-W1 Wayback-body candidate verification; fail closed.

No sportsbook, outcomes, broad archive scan, raw-content publication or model
fitting. The 2024-09-07 Wayback index record was independently observed in
prior frozen run 36739781377; an index hit by itself proves no table contents.
"""
from __future__ import annotations
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scripts.research.acquire_fantasyalarm_wr_cb_archive_v1 import parse_page
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule

ARTICLE = ("https://www.fantasyalarm.com/articles/nfl/wide-receivers/"
           "2024-fantasy-football-wr-cb-matchup-report-week-1-drake-london-looks-to-take-off/163226")
TS = "20240907005714"
CDX_DIGEST = "JE62ZZZYJ4TELVSMS7LVWRPWWTXE53HA"
ARCHIVED_UTC = datetime.strptime(TS, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
REPLAY = f"https://web.archive.org/web/{TS}id_/{ARTICLE}"

def sha1_b32(data: bytes) -> str:
    return base64.b32encode(hashlib.sha1(data).digest()).decode("ascii").rstrip("=")

def pregame_row_timestamp_eligible(source_publish, kickoff, archived=ARCHIVED_UTC) -> bool:
    pub = pd.to_datetime(source_publish, utc=True, errors="coerce")
    game = pd.to_datetime(kickoff, utc=True, errors="coerce")
    if pd.isna(pub) or pd.isna(game):
        return False
    return bool(pub <= pd.Timestamp(archived) < game)


SCRIPT_MARKERS = (
    "cornerback", "wide receiver", "wr vs", "matchup",
    "articlebody", "article_body", "__next_data__", "__initial_state__",
    "apollo", "preloadedstate", "/nfl/players/",
)

def _walk_json_keys(value, out: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            out.add(str(key).lower())
            _walk_json_keys(child, out)
    elif isinstance(value, list):
        for child in value:
            _walk_json_keys(child, out)

def script_structure_inventory(soup: BeautifulSoup) -> list[dict]:
    """Hash/marker-only script inventory; never preserve script body text."""
    items: list[dict] = []
    for idx, node in enumerate(soup.find_all("script")):
        body = node.string if node.string is not None else node.get_text("", strip=False)
        text = str(body or "")
        low = text.lower()
        record = {
            "index": idx,
            "type": str(node.get("type") or ""),
            "id": str(node.get("id") or ""),
            "has_src": bool(node.get("src")),
            "characters": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "markers": [m for m in SCRIPT_MARKERS if m in low],
            "json_parseable": False,
            "json_root_type": "",
            "json_key_count": 0,
            "json_keys_of_interest": [],
        }
        if text.strip():
            try:
                payload = json.loads(text)
            except (ValueError, TypeError):
                payload = None
            if payload is not None:
                record["json_parseable"] = True
                record["json_root_type"] = type(payload).__name__
                keys: set[str] = set()
                _walk_json_keys(payload, keys)
                record["json_key_count"] = len(keys)
                interests = (
                    "articlebody", "article_body", "content", "body", "html",
                    "description", "matchup", "cornerback", "receiver", "players",
                    "props", "pageprops", "data",
                )
                record["json_keys_of_interest"] = [k for k in interests if k in keys]
        items.append(record)
    return items


def _collect_article_bodies(value, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == "articlebody" and isinstance(child, str) and child.strip():
                out.append(child)
            else:
                _collect_article_bodies(child, out)
    elif isinstance(value, list):
        for child in value:
            _collect_article_bodies(child, out)

def archived_article_body_inventory(soup: BeautifulSoup) -> tuple[list[str], list[dict]]:
    """Return in-memory JSON-LD article bodies plus non-text structural evidence.

    Body strings may be processed only in-memory. The caller must never persist
    or print them; only hashes, structure and sanitized factual pairing rows.
    """
    bodies: list[str] = []
    for node in soup.find_all("script", attrs={"type":"application/ld+json"}):
        text = node.string if node.string is not None else node.get_text("", strip=False)
        try:
            payload = json.loads(str(text or ""))
        except (ValueError, TypeError):
            continue
        _collect_article_bodies(payload, bodies)
    inventory = []
    for idx, body in enumerate(bodies):
        bs = BeautifulSoup(body, "html.parser")
        inventory.append({
            "index": idx,
            "characters": len(body),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "html_tag_count": len(bs.find_all()),
            "html_tables": len(bs.find_all("table")),
            "html_rows": len(bs.find_all("tr")),
            "html_divs": len(bs.find_all("div")),
            "html_paragraphs": len(bs.find_all("p")),
            "newline_count": body.count("\n"),
            "contains_wr_cb_terms": (
                "wide receiver" in body.lower()
                and "cornerback" in body.lower()
                and "matchup" in body.lower()
            ),
            "contains_player_links": "/nfl/players/" in body.lower(),
        })
    return bodies, inventory

def _synthetic_page_from_archived_body(body: str, published_at_utc: str) -> str:
    """Supply source publication metadata without changing archived body markup."""
    pub = str(published_at_utc or "")
    return (
        '<html><head><meta property="article:published_time" content="'
        + pub.replace('"', "&quot;")
        + '"></head><body>'
        + body
        + "</body></html>"
    )

def execute(out_dir: Path) -> dict:
    result = {
        "contract":"WR_CB_2024W1_EXACT_ARCHIVED_BODY_CANDIDATE_V1",
        "source_url":ARTICLE, "archive_index_timestamp_utc":ARCHIVED_UTC.isoformat(),
        "index_digest_sha1_base32":CDX_DIGEST,
        "exact_replay_url":REPLAY, "status":"NOT_ATTEMPTED",
        "indexed_record_does_not_prove_content":True,
        "archived_body_digest_verified":False, "verified_pregame_factual_rows":0,
        "raw_body_saved":False, "model_candidates_fit":0, "sportsbook_inputs":False,
        "target_game_outcomes":False, "source_model_gate_cleared":False,
        "factual_pregame_rows":[],
    }
    try:
        response = requests.get(
            REPLAY, headers={"User-Agent":"NFLSourceProvenanceAudit/1.0"}, timeout=22,
            allow_redirects=True,
        )
        result["http_status"] = response.status_code
        result["final_replay_url"] = response.url
        if response.status_code != 200:
            result["status"] = f"BODY_LOOKUP_HTTP_{response.status_code}"
        elif (urlparse(response.url).hostname != "web.archive.org"
              or f"/web/{TS}" not in urlparse(response.url).path):
            result["status"] = "REPLAY_REDIRECTED_AWAY_FROM_EXACT_INDEXED_TIMESTAMP"
        elif len(response.content) > 3500000:
            result["status"] = "BODY_TOO_LARGE_FAIL_CLOSED"
        else:
            raw = response.content
            result["body_sha256"] = hashlib.sha256(raw).hexdigest()
            result["body_sha1_b32"] = sha1_b32(raw)
            if result["body_sha1_b32"] != CDX_DIGEST:
                result["status"] = "REPLAY_BODY_DIGEST_MISMATCH_UNVERIFIED"
            else:
                result["archived_body_digest_verified"] = True
                # Structure-only probe of cryptographically verified exact
                # 2024 replay bytes. Do not persist/copy copyrighted HTML.
                soup = BeautifulSoup(raw, "html.parser")
                plain = soup.get_text(" ", strip=True).lower()
                result["archived_body_structure"] = {
                    "bytes": len(raw),
                    "html_tables": len(soup.find_all("table")),
                    "html_rows": len(soup.find_all("tr")),
                    "wr_cb_alignment_headings": sum(
                        bool("wr" in x.get_text(" ", strip=True).lower()
                             or "wide receiver" in x.get_text(" ", strip=True).lower())
                        for x in soup.find_all(["h2","h3","h4"])
                    ),
                    "script_nodes": len(soup.find_all("script")),
                    "json_ld_scripts": len(soup.find_all("script", attrs={"type":"application/ld+json"})),
                    "body_text_characters": len(plain),
                    "contains_cornerback_term": "cornerback" in plain,
                    "contains_receiver_term": ("wide receiver" in plain or "wr vs" in plain),
                    "contains_matchup_term": "matchup" in plain,
                    "has_next_data_script": soup.find("script",id="__NEXT_DATA__") is not None,
                    "json_like_script_count": sum("json" in str(x.get("type",""))
                                                  for x in soup.find_all("script")),
                }
                bodies, body_inventory = archived_article_body_inventory(soup)
                result["archived_article_body_structure"] = {
                    "candidate_count": len(bodies),
                    "candidates": body_inventory,
                    "raw_article_body_saved": False,
                }
                scripts = script_structure_inventory(soup)
                result["archived_script_structure"] = {
                    "script_count": len(scripts),
                    "scripts_with_matchup_markers": sum(
                        bool(set(x["markers"]) & {"cornerback","wide receiver","wr vs","matchup"})
                        for x in scripts
                    ),
                    "scripts_with_player_link_marker": sum(
                        "/nfl/players/" in x["markers"] for x in scripts
                    ),
                    "json_parseable_scripts": sum(x["json_parseable"] for x in scripts),
                    "json_scripts_with_content_like_keys": sum(
                        bool(set(x["json_keys_of_interest"]) & {
                            "articlebody","article_body","content","body","html","matchup","players"
                        }) for x in scripts
                    ),
                    "scripts": scripts,
                    "raw_script_text_saved": False,
                }
                full_html = raw.decode(response.encoding or "utf-8", errors="replace")
                rows, page = parse_page(
                    full_html, season=2024, week=1, source_url=ARTICLE,
                )
                result["parsed_outer_html_pairing_rows"] = len(rows)
                source_publish = page["published_at_utc"]
                embedded_frames = []
                for body in bodies:
                    candidate_rows, _ = parse_page(
                        _synthetic_page_from_archived_body(body, source_publish),
                        season=2024, week=1, source_url=ARTICLE,
                    )
                    if not candidate_rows.empty:
                        embedded_frames.append(candidate_rows)
                embedded = (
                    pd.concat(embedded_frames, ignore_index=True).drop_duplicates(
                        ["season","week","alignment_bucket","wr_clean_key","cb_clean_key"],
                        keep="last",
                    )
                    if embedded_frames else pd.DataFrame()
                )
                result["parsed_embedded_articlebody_pairing_rows"] = len(embedded)
                if rows.empty and not embedded.empty:
                    rows = embedded
                    result["pairing_parse_source"] = "ARCHIVED_JSON_LD_ARTICLEBODY"
                elif not rows.empty:
                    result["pairing_parse_source"] = "ARCHIVED_OUTER_HTML"
                else:
                    result["pairing_parse_source"] = "NONE"
                result["parsed_factual_pairing_rows"] = len(rows)
                if rows.empty:
                    result["status"] = "MATCHED_BODY_NO_EXPLICIT_PARSEABLE_PAIRINGS"
                else:
                    sched = _load_schedule([2024]).rename(columns={"team":"wr_team"})
                    merged = rows.merge(
                        sched, on=["season","week","wr_team"], how="left",
                        validate="many_to_one",
                    )
                    good = merged.loc[
                        merged.apply(
                            lambda r: pregame_row_timestamp_eligible(
                                source_publish,r["kickoff_utc"]), axis=1)
                        & merged["opponent"].astype(str).eq(
                            merged["scheduled_opponent"].astype(str))
                        & merged["alignment_bucket"].ne("UNKNOWN_ALIGNMENT")
                    ].copy()
                    # Keep only factual row identifiers for auditable source
                    # provenance; NO editorial grade or raw copyrighted HTML.
                    result["factual_pregame_rows"] = good[[
                        "season","week","wr_raw","wr_team","cb_raw",
                        "opponent","alignment_bucket",
                    ]].drop_duplicates().to_dict("records")
                    result["verified_pregame_factual_rows"] = len(
                        result["factual_pregame_rows"]
                    )
                    result["source_publication_utc"] = source_publish
                    result["status"] = (
                        "ARCHIVE_BODY_HASH_MATCH_WITH_STRICT_PREGAME_ROW_CANDIDATES"
                        if result["verified_pregame_factual_rows"]
                        else "ARCHIVE_BODY_HASH_MATCH_NO_ELIGIBLE_PREGAME_ROWS"
                    )
    except requests.RequestException as e:
        result["status"] = "BODY_LOOKUP_NETWORK_" + type(e).__name__
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"wayback_exact_body_verification.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="factual_pregame_rows"},
                     indent=2,sort_keys=True))
    return result

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--out-dir",required=True,type=Path)
    v=a.parse_args()
    execute(v.out_dir)
