#!/usr/bin/env python3
"""Acquire factual WR-CB pairing rows from public FantasyAlarm archive pages.

Research/source-audit only. This deliberately preserves but does NOT authorize
FantasyAlarm's editorial matchup grade for football-model use.

The parser is fail-closed:
- explicit manifest URLs only;
- publication timestamp preserved;
- only tables with an explicit WR and CB field are emitted;
- unlisted players are missing, never zero/no-matchup;
- ambiguous identity rows are quarantined, not guessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NFLResearchBot/1.0; public-source-audit)",
    "Accept-Language": "en-US,en;q=0.9",
}
ALIGNMENTS = {
    "left wr": "LWR_VS_RCB",
    "left wide receiver": "LWR_VS_RCB",
    "right wr": "RWR_VS_LCB",
    "right wide receiver": "RWR_VS_LCB",
    "slot wr": "SWR_VS_SCB",
    "slot wide receiver": "SWR_VS_SCB",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_col(c) -> str:
    if isinstance(c, tuple):
        c = " ".join(str(x) for x in c if str(x).lower() != "nan")
    c = re.sub(r"\s+", " ", str(c)).strip().lower()
    return c


def _publication_time(soup: BeautifulSoup) -> str:
    selectors = [
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "date"}, "content"),
        ("meta", {"name": "publish-date"}, "content"),
    ]
    for tag, attrs, field in selectors:
        node = soup.find(tag, attrs=attrs)
        if node and node.get(field):
            value = str(node.get(field)).strip()
            try:
                ts = pd.to_datetime(value, utc=True, errors="raise")
                return ts.isoformat().replace("+00:00", "Z")
            except Exception:
                pass

    for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(node.get_text(" ", strip=True))
        except Exception:
            continue
        objs: Iterable[dict]
        if isinstance(payload, dict):
            objs = [payload]
        elif isinstance(payload, list):
            objs = [x for x in payload if isinstance(x, dict)]
        else:
            objs = []
        for obj in objs:
            value = obj.get("datePublished")
            if value:
                try:
                    ts = pd.to_datetime(value, utc=True, errors="raise")
                    return ts.isoformat().replace("+00:00", "Z")
                except Exception:
                    pass

    node = soup.find("time")
    if node:
        value = node.get("datetime") or node.get_text(" ", strip=True)
        try:
            ts = pd.to_datetime(value, utc=True, errors="raise")
            return ts.isoformat().replace("+00:00", "Z")
        except Exception:
            pass
    return ""


def _alignment_for_table(table) -> str:
    heading = table.find_previous(["h1", "h2", "h3", "h4", "h5", "strong"])
    text = heading.get_text(" ", strip=True).lower() if heading else ""
    for needle, label in ALIGNMENTS.items():
        if needle in text:
            return label
    # fallback: inspect nearest preceding text block
    prev = table.find_previous(string=re.compile(r"(left|right|slot).*(wr|wide receiver)", re.I))
    text = str(prev).lower() if prev else ""
    for needle, label in ALIGNMENTS.items():
        if needle in text:
            return label
    return "UNKNOWN_ALIGNMENT"


def _find_col(cols: list[str], needles: list[str]) -> str | None:
    for n in needles:
        for c in cols:
            if n == c or n in c:
                return c
    return None


def _canon_name(value) -> tuple[str, str]:
    display, key = canonicalize_player_name_safe(value)
    return str(display or "").strip(), str(key or "").strip()


def _normalize_matchup(value) -> str:
    x = re.sub(r"\s+", " ", str(value or "")).strip()
    return x


def parse_page(html: str, *, season: int, week: int, source_url: str) -> tuple[pd.DataFrame, dict]:
    soup = BeautifulSoup(html, "html.parser")
    published = _publication_time(soup)
    rows: list[dict] = []
    table_audit: list[dict] = []

    for table_index, table in enumerate(soup.find_all("table")):
        try:
            dfs = pd.read_html(str(table))
        except Exception as exc:
            table_audit.append({
                "table_index": table_index,
                "status": "PARSE_ERROR",
                "detail": str(exc)[:240],
            })
            continue
        if not dfs:
            continue
        df = dfs[0].copy()
        df.columns = [_clean_col(c) for c in df.columns]
        cols = list(df.columns)
        wr_col = _find_col(cols, ["wide receiver", "receiver", "wr"])
        cb_col = _find_col(cols, ["cornerback", "right cb", "left cb", "slot cb", "cb"])
        team_col = _find_col(cols, ["team"])
        opp_col = _find_col(cols, ["opp", "opponent"])
        matchup_col = _find_col(cols, ["matchup"])
        alignment = _alignment_for_table(table)

        if not wr_col or not cb_col:
            table_audit.append({
                "table_index": table_index,
                "status": "SKIP_NO_EXPLICIT_WR_CB_COLUMNS",
                "columns": cols,
                "alignment": alignment,
            })
            continue

        emitted = 0
        for _, r in df.iterrows():
            wr_raw = str(r.get(wr_col, "") or "").strip()
            cb_raw = str(r.get(cb_col, "") or "").strip()
            if not wr_raw or not cb_raw:
                continue
            if wr_raw.lower() in {"wide receiver", "wr", "nan", "bye"}:
                continue
            if cb_raw.lower() in {"cornerback", "cb", "nan", "n/a", "bye"}:
                continue

            wr_name, wr_key = _canon_name(wr_raw)
            cb_name, cb_key = _canon_name(cb_raw)
            team = canon_team(str(r.get(team_col, "") or "")) if team_col else ""
            opponent = canon_team(str(r.get(opp_col, "") or "")) if opp_col else ""
            identity_ok = bool(wr_key and cb_key)

            rows.append({
                "season": int(season),
                "week": int(week),
                "source": "fantasyalarm",
                "source_url": source_url,
                "published_at_utc": published,
                "alignment_bucket": alignment,
                "wr_raw": wr_raw,
                "wr": wr_name,
                "wr_clean_key": wr_key,
                "wr_team": team,
                "cb_raw": cb_raw,
                "cb": cb_name,
                "cb_clean_key": cb_key,
                "opponent": opponent,
                "editorial_matchup_raw": _normalize_matchup(r.get(matchup_col, "")) if matchup_col else "",
                "editorial_matchup_model_eligible": False,
                "identity_status": "READY" if identity_ok else "QUARANTINE_IDENTITY",
                "source_table_index": int(table_index),
            })
            emitted += 1

        table_audit.append({
            "table_index": table_index,
            "status": "PARSED_EXPLICIT_WR_CB_TABLE",
            "columns": cols,
            "alignment": alignment,
            "rows_emitted": emitted,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(
            ["season", "week", "alignment_bucket", "wr_clean_key", "cb_clean_key"],
            keep="last",
        ).reset_index(drop=True)

    audit = {
        "season": int(season),
        "week": int(week),
        "source_url": source_url,
        "published_at_utc": published,
        "html_sha256": _sha256_text(html),
        "html_bytes": len(html.encode("utf-8")),
        "tables_seen": len(soup.find_all("table")),
        "rows_emitted": int(len(out)),
        "alignment_counts": (
            out["alignment_bucket"].value_counts(dropna=False).to_dict()
            if not out.empty else {}
        ),
        "identity_quarantine_rows": (
            int(out["identity_status"].ne("READY").sum()) if not out.empty else 0
        ),
        "table_audit": table_audit,
    }
    return out, audit


def fetch_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.text


def run_manifest(manifest: pd.DataFrame, out_dir: Path) -> dict:
    required = {"season", "week", "url"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(f"manifest missing columns: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    page_dir = out_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    audits: list[dict] = []

    for row in manifest.itertuples(index=False):
        season, week, url = int(row.season), int(row.week), str(row.url).strip()
        if not url.startswith("https://www.fantasyalarm.com/"):
            raise RuntimeError(f"unexpected source domain: {url}")

        html = fetch_page(url)
        page_rows, audit = parse_page(html, season=season, week=week, source_url=url)

        html_path = page_dir / f"{season}_wk{week:02d}.html"
        if html_path.exists():
            prior = html_path.read_text(encoding="utf-8")
            if _sha256_text(prior) != _sha256_text(html):
                raise RuntimeError(
                    f"immutable source collision for {season} W{week}: page bytes changed"
                )
        else:
            html_path.write_text(html, encoding="utf-8")

        if not page_rows.empty:
            all_rows.append(page_rows)
        audits.append(audit)

    detail = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    detail.to_csv(out_dir / "fantasyalarm_wr_cb_assignments.csv", index=False)

    audit_rows = []
    for a in audits:
        audit_rows.append({
            k: v for k, v in a.items() if k != "table_audit"
        })
    pd.DataFrame(audit_rows).to_csv(out_dir / "fantasyalarm_wr_cb_page_audit.csv", index=False)
    (out_dir / "fantasyalarm_wr_cb_full_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "contract": "WR_CB_FREE_HISTORICAL_ARCHIVE_SOURCE_V1",
        "pages": int(len(manifest)),
        "assignment_rows": int(len(detail)),
        "seasons": sorted(detail["season"].dropna().astype(int).unique().tolist()) if len(detail) else [],
        "weeks": int(detail[["season", "week"]].drop_duplicates().shape[0]) if len(detail) else 0,
        "ready_rows": int(detail["identity_status"].eq("READY").sum()) if len(detail) else 0,
        "unknown_alignment_rows": int(detail["alignment_bucket"].eq("UNKNOWN_ALIGNMENT").sum()) if len(detail) else 0,
        "editorial_matchup_used_as_model_feature": False,
        "sportsbook_input_used": False,
    }
    (out_dir / "fantasyalarm_wr_cb_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    manifest = pd.read_csv(a.manifest)
    summary = run_manifest(manifest, a.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
