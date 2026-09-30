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


def _norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


TEAM_ABBRS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ",
    "PHI","PIT","SEA","SF","TB","TEN","WAS",
}
MATCHUP_LABELS = {
    "safe","moderate","risky","upgrade","neutral","downgrade",
    "great","good","average","bad","poor","n/a","#n/a",
}


def _alignment_token(value: str) -> str | None:
    x = _norm_text(value).lower().replace(".", "")
    if re.search(r"left (?:wr|wide receiver)\s+vs\s+right (?:cb|cornerback)", x):
        return "LWR_VS_RCB"
    if re.search(r"right (?:wr|wide receiver)\s+vs\s+left (?:cb|cornerback)", x):
        return "RWR_VS_LCB"
    if re.search(r"slot (?:wr|wide receiver)\s+vs\s+slot (?:cb|cornerback)", x):
        return "SWR_VS_SCB"
    return None


def _is_team(value: str) -> bool:
    return _norm_text(value).upper() in TEAM_ABBRS.union({"WFT","WSH"})


def _is_matchup_label(value: str) -> bool:
    return _norm_text(value).lower() in MATCHUP_LABELS


def _source_player_id(node) -> str:
    """Return FantasyAlarm's stable numeric player ID from one source cell/node."""
    if node is None or not hasattr(node, "find"):
        return ""
    a = node.find("a", href=True)
    if not a:
        return ""
    m = re.search(r"/nfl/players/(\d+)(?:/|$)", str(a.get("href") or ""))
    return m.group(1) if m else ""


def _emit_text_pair(
    rows: list[dict],
    *,
    season: int,
    week: int,
    source_url: str,
    published: str,
    alignment: str,
    wr_raw: str,
    team_raw: str,
    cb_raw: str,
    opp_raw: str,
    matchup_raw: str,
    source_layout: str,
    wr_source_player_id: str = "",
    cb_source_player_id: str = "",
) -> None:
    wr_raw = _norm_text(wr_raw)
    cb_raw = _norm_text(cb_raw)
    team_raw = _norm_text(team_raw).upper()
    opp_raw = _norm_text(opp_raw).upper()
    matchup_raw = _normalize_matchup(matchup_raw)

    # Bye / no-assignment rows remain missing rather than becoming a fake pair.
    if not wr_raw or not cb_raw:
        return
    if wr_raw.lower() in {"wide receiver","wr","n/a","#n/a","bye"}:
        return
    if cb_raw.lower() in {"cornerback","cb","n/a","#n/a","bye"}:
        return
    if not _is_team(team_raw) or not _is_team(opp_raw):
        return

    wr_name, wr_key = _canon_name(wr_raw)
    cb_name, cb_key = _canon_name(cb_raw)
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
        "wr_team": canon_team(team_raw),
        "wr_source_player_id": str(wr_source_player_id or "").strip(),
        "cb_raw": cb_raw,
        "cb": cb_name,
        "cb_clean_key": cb_key,
        "opponent": canon_team(opp_raw),
        "cb_source_player_id": str(cb_source_player_id or "").strip(),
        "editorial_matchup_raw": matchup_raw,
        "editorial_matchup_model_eligible": False,
        "identity_status": "READY" if wr_key and cb_key else "QUARANTINE_IDENTITY",
        "source_table_index": -1,
        "source_layout": source_layout,
    })


def _parse_text_cards(
    soup: BeautifulSoup,
    *,
    season: int,
    week: int,
    source_url: str,
    published: str,
) -> list[dict]:
    """Parse explicit card/grid layouts used by older FantasyAlarm articles.

    2023-24 commonly render one six-field WR/team/price/CB/opp/grade row.
    2025 commonly renders a WR three-field card followed by a CB three-field
    card. Only explicit labeled structure is used; prose is never mined.
    """
    tokens = [_norm_text(x) for x in soup.stripped_strings]
    tokens = [x for x in tokens if x]
    rows: list[dict] = []
    alignment = "UNKNOWN_ALIGNMENT"
    i = 0
    n = len(tokens)

    while i < n:
        maybe_alignment = _alignment_token(tokens[i])
        if maybe_alignment:
            alignment = maybe_alignment
            i += 1
            continue

        # Historical 2022-24 layouts print the combined header once, then
        # emit many six-field rows beneath it. Recover each explicit row by
        # anchoring on the standalone matchup label and validating the two NFL
        # team abbreviations immediately around the WR/CB cells. Narrative prose
        # cannot satisfy this shape.
        if alignment != "UNKNOWN_ALIGNMENT" and _is_matchup_label(tokens[i]):
            emitted = False
            if i >= 5 and _is_team(tokens[i - 4]) and _is_team(tokens[i - 1]):
                _emit_text_pair(
                    rows,
                    season=season, week=week, source_url=source_url,
                    published=published, alignment=alignment,
                    wr_raw=tokens[i - 5], team_raw=tokens[i - 4],
                    cb_raw=tokens[i - 2], opp_raw=tokens[i - 1],
                    matchup_raw=tokens[i],
                    source_layout="TEXT_COMBINED_SIX_FIELD_STREAM",
                )
                emitted = True
            elif i >= 4 and _is_team(tokens[i - 3]) and _is_team(tokens[i - 1]):
                # Some rows have an empty salary cell; stripped_strings removes
                # the blank and leaves a five-value row.
                _emit_text_pair(
                    rows,
                    season=season, week=week, source_url=source_url,
                    published=published, alignment=alignment,
                    wr_raw=tokens[i - 4], team_raw=tokens[i - 3],
                    cb_raw=tokens[i - 2], opp_raw=tokens[i - 1],
                    matchup_raw=tokens[i],
                    source_layout="TEXT_COMBINED_FIVE_FIELD_STREAM",
                )
                emitted = True
            if emitted:
                i += 1
                continue

        if tokens[i].lower() != "wide receiver":
            i += 1
            continue
        if i + 2 >= n or tokens[i + 1].lower() != "team":
            i += 1
            continue

        # Look for a combined Cornerback/Opp/Matchup header before row values.
        corner_header = next(
            (j for j in range(i + 2, min(i + 7, n))
             if tokens[j].lower() == "cornerback"),
            None,
        )

        if (
            corner_header is not None
            and corner_header + 2 < n
            and tokens[corner_header + 1].lower() in {"opp","opponent"}
            and tokens[corner_header + 2].lower() == "matchup"
        ):
            value_start = corner_header + 3
            window = tokens[value_start:min(value_start + 10, n)]
            team_idx = next((k for k,v in enumerate(window[:4]) if _is_team(v)), None)
            grade_idx = next((k for k,v in enumerate(window) if _is_matchup_label(v)), None)
            if team_idx is not None and grade_idx is not None:
                wr_idx = team_idx - 1
                opp_idx = grade_idx - 1
                cb_idx = grade_idx - 2
                if wr_idx >= 0 and cb_idx > team_idx and opp_idx >= 0 and _is_team(window[opp_idx]):
                    _emit_text_pair(
                        rows,
                        season=season, week=week, source_url=source_url,
                        published=published, alignment=alignment,
                        wr_raw=window[wr_idx], team_raw=window[team_idx],
                        cb_raw=window[cb_idx], opp_raw=window[opp_idx],
                        matchup_raw=window[grade_idx],
                        source_layout="TEXT_COMBINED_SIX_FIELD",
                    )
                    i = value_start + grade_idx + 1
                    continue

        # Split-card layout: WR header/value, then explicit CB header/value.
        wr_value_start = i + 3
        if wr_value_start + 1 < n:
            wr_raw = tokens[wr_value_start]
            team_raw = tokens[wr_value_start + 1]
            if _is_team(team_raw):
                cb_head = next(
                    (j for j in range(wr_value_start + 2, min(wr_value_start + 7, n))
                     if tokens[j].lower() == "cornerback"),
                    None,
                )
                if (
                    cb_head is not None
                    and cb_head + 5 < n
                    and tokens[cb_head + 1].lower() in {"opp","opponent"}
                    and tokens[cb_head + 2].lower() == "matchup"
                ):
                    cb_raw = tokens[cb_head + 3]
                    opp_raw = tokens[cb_head + 4]
                    grade = tokens[cb_head + 5]
                    if _is_team(opp_raw) and _is_matchup_label(grade):
                        _emit_text_pair(
                            rows,
                            season=season, week=week, source_url=source_url,
                            published=published, alignment=alignment,
                            wr_raw=wr_raw, team_raw=team_raw,
                            cb_raw=cb_raw, opp_raw=opp_raw,
                            matchup_raw=grade,
                            source_layout="TEXT_SPLIT_WR_CB_CARDS",
                        )
                        i = cb_head + 6
                        continue
        i += 1

    return rows


def _source_team(value: str) -> str:
    x = _norm_text(value).upper()
    if x in {"WFT", "WSH"}:
        return "WAS"
    return canon_team(x)


def _split_embedded_team(value: str) -> tuple[str, str] | None:
    x = _norm_text(value)
    m = re.match(r"^(.+?)\s+([A-Z]{2,3})$", x)
    if not m:
        return None
    name, team = m.group(1).strip(), m.group(2).strip().upper()
    if team not in TEAM_ABBRS and team not in {"WFT","WSH"}:
        return None
    return name, _source_team(team)


def _parse_inline_2026_pairs(
    soup: BeautifulSoup,
    *,
    season: int,
    week: int,
    source_url: str,
    published: str,
) -> list[dict]:
    """Parse 2026 inline cards: WR (TEAM) / vs. CB (TEAM) • Matchup: grade."""
    lines = [_norm_text(x) for x in soup.get_text("\n", strip=True).splitlines()]
    lines = [x for x in lines if x]
    rows: list[dict] = []
    alignment = "UNKNOWN_ALIGNMENT"
    pending: tuple[str,str] | None = None

    wr_re = re.compile(r"^(.+?)\s*\(([A-Z]{2,3})\)\s*$")
    pair_re = re.compile(
        r"^vs\.?\s*(.+?)\s*\(([A-Z]{2,3})\)"
        r".*?Matchup\s*:\s*(Safe|Moderate|Risky|Upgrade|Neutral|Downgrade)\b",
        re.I,
    )

    team_only_re = re.compile(r"^\(([A-Z]{2,3})\)\s*$")
    opp_match_re = re.compile(r"^\(([A-Z]{2,3})\)\s*[•·]\s*Matchup\s*:\s*(.*)$", re.I)

    for idx, line in enumerate(lines):
        # Actual 2026 server HTML splits one pairing into:
        # WR name / (TEAM) / vs. / CB name / (OPP) • Matchup: / grade.
        if idx + 5 < len(lines):
            tm = team_only_re.match(lines[idx + 1])
            om = opp_match_re.match(lines[idx + 4])
            if (
                tm and _is_team(tm.group(1))
                and lines[idx + 2].lower() in {"vs.", "vs"}
                and om and _is_team(om.group(1))
            ):
                grade = _norm_text(om.group(2)) or _norm_text(lines[idx + 5])
                if _is_matchup_label(grade):
                    _emit_text_pair(
                        rows,
                        season=season, week=week, source_url=source_url,
                        published=published, alignment=alignment,
                        wr_raw=line, team_raw=tm.group(1),
                        cb_raw=lines[idx + 3], opp_raw=om.group(1),
                        matchup_raw=grade,
                        source_layout="TEXT_2026_SPLIT_NODES",
                    )

        # New 2026 headings include parenthetical abbreviations.
        low = line.lower()
        if (
            ("left wide receiver" in low or "left wr" in low)
            and ("right cornerback" in low or "right cb" in low)
            and "vs" in low
        ):
            alignment = "LWR_VS_RCB"
            pending = None
            continue
        if (
            ("right wide receiver" in low or "right wr" in low)
            and ("left cornerback" in low or "left cb" in low)
            and "vs" in low
        ):
            alignment = "RWR_VS_LCB"
            pending = None
            continue
        if (
            ("slot wide receiver" in low or "slot wr" in low)
            and ("slot cornerback" in low or "slot cb" in low)
            and "vs" in low
        ):
            alignment = "SWR_VS_SCB"
            pending = None
            continue

        m = wr_re.match(line)
        if m and _is_team(m.group(2)):
            # Only arm a WR candidate when the next bounded text contains an
            # explicit vs/Matchup card; this prevents unrelated NAME (TEAM)
            # article prose from being treated as a matchup.
            pending = (m.group(1).strip(), m.group(2).strip().upper())
            continue

        if pending and line.lower().startswith("vs"):
            chunk = line
            # Some DOM versions split the CB name/team/Matchup across adjacent
            # text nodes. Join only a tiny bounded window.
            for extra in range(1, 4):
                mm = pair_re.match(chunk)
                if mm:
                    break
                if idx + extra < len(lines):
                    chunk += " " + lines[idx + extra]
            mm = pair_re.match(chunk)
            if mm and _is_team(mm.group(2)):
                _emit_text_pair(
                    rows,
                    season=season, week=week, source_url=source_url,
                    published=published, alignment=alignment,
                    wr_raw=pending[0], team_raw=pending[1],
                    cb_raw=mm.group(1).strip(), opp_raw=mm.group(2).strip(),
                    matchup_raw=mm.group(3).strip(),
                    source_layout="TEXT_2026_INLINE_PAIR",
                )
            pending = None

    return rows


def parse_page(html: str, *, season: int, week: int, source_url: str) -> tuple[pd.DataFrame, dict]:
    soup = BeautifulSoup(html, "html.parser")
    published = _publication_time(soup)
    rows: list[dict] = []
    table_audit: list[dict] = []
    structured_table_rows = 0

    for table_index, table in enumerate(soup.find_all("table")):
        # 2022-24 articles use real HTML rows with a stable six-field factual
        # schema, but often without semantic TH elements. Parse the actual TD
        # cells rather than soup.stripped_strings so suffixes/hyphenated names
        # cannot be split into fake players such as "Jr.", "ton", or "-Ikhine".
        current_alignment = "UNKNOWN_ALIGNMENT"
        structured_emitted = 0
        for tr in table.find_all("tr"):
            cell_nodes = tr.find_all(["td", "th"])
            cells = [_norm_text(x.get_text(" ", strip=True)) for x in cell_nodes]
            if len(cells) == 1:
                maybe_alignment = _alignment_token(cells[0])
                if maybe_alignment:
                    current_alignment = maybe_alignment
                continue
            if len(cells) < 6 or current_alignment == "UNKNOWN_ALIGNMENT":
                continue
            # Exact historical row shape:
            # WR | TEAM | salary | CB | OPP | editorial matchup.
            if not _is_team(cells[1]) or not _is_team(cells[4]) or not _is_matchup_label(cells[5]):
                continue
            before = len(rows)
            _emit_text_pair(
                rows,
                season=season, week=week, source_url=source_url,
                published=published, alignment=current_alignment,
                wr_raw=cells[0], team_raw=cells[1],
                cb_raw=cells[3], opp_raw=cells[4],
                matchup_raw=cells[5],
                source_layout="HTML_TABLE_EXPLICIT_SIX_FIELD",
                wr_source_player_id=_source_player_id(cell_nodes[0]),
                cb_source_player_id=_source_player_id(cell_nodes[3]),
            )
            if len(rows) > before:
                structured_emitted += 1
                structured_table_rows += 1
        if structured_emitted:
            table_audit.append({
                "table_index": table_index,
                "status": "PARSED_STRUCTURED_SIX_FIELD_WR_CB_ROWS",
                "alignment": current_alignment,
                "rows_emitted": structured_emitted,
            })
            # This historical table is fully parsed from its own cells; do not
            # re-parse it through pandas/text token streams.
            continue
        # 2025 pages use alternating three-cell WR and CB cards. Recover the
        # pair from neighboring table rows so player names and provider IDs stay
        # attached to their exact cells; narrative prose is never parsed.
        split_alignment = "UNKNOWN_ALIGNMENT"
        split_emitted = 0
        split_trs = table.find_all("tr")
        i = 0
        while i < len(split_trs):
            nodes = split_trs[i].find_all(["td", "th"])
            vals = [_norm_text(x.get_text(" ", strip=True)) for x in nodes]
            if len(vals) == 1:
                maybe_alignment = _alignment_token(vals[0])
                if maybe_alignment:
                    split_alignment = maybe_alignment
                i += 1
                continue
            if (
                split_alignment != "UNKNOWN_ALIGNMENT"
                and len(vals) >= 2
                and vals[0].lower() == "wide receiver"
                and vals[1].lower() == "team"
                and i + 3 < len(split_trs)
            ):
                wr_nodes = split_trs[i + 1].find_all(["td", "th"])
                cb_head_nodes = split_trs[i + 2].find_all(["td", "th"])
                cb_nodes = split_trs[i + 3].find_all(["td", "th"])
                wr_vals = [_norm_text(x.get_text(" ", strip=True)) for x in wr_nodes]
                cb_head = [_norm_text(x.get_text(" ", strip=True)).lower() for x in cb_head_nodes]
                cb_vals = [_norm_text(x.get_text(" ", strip=True)) for x in cb_nodes]
                if (
                    len(wr_vals) >= 2
                    and len(cb_head) >= 3
                    and cb_head[0] == "cornerback"
                    and cb_head[1] in {"opp", "opponent"}
                    and cb_head[2] == "matchup"
                    and len(cb_vals) >= 3
                    and _is_team(wr_vals[1])
                    and _is_team(cb_vals[1])
                    and _is_matchup_label(cb_vals[2])
                ):
                    before = len(rows)
                    _emit_text_pair(
                        rows,
                        season=season, week=week, source_url=source_url,
                        published=published, alignment=split_alignment,
                        wr_raw=wr_vals[0], team_raw=wr_vals[1],
                        cb_raw=cb_vals[0], opp_raw=cb_vals[1],
                        matchup_raw=cb_vals[2],
                        source_layout="HTML_TABLE_SPLIT_WR_CB_CARDS",
                        wr_source_player_id=_source_player_id(wr_nodes[0]) if wr_nodes else "",
                        cb_source_player_id=_source_player_id(cb_nodes[0]) if cb_nodes else "",
                    )
                    if len(rows) > before:
                        split_emitted += 1
                        structured_table_rows += 1
                    i += 4
                    continue
            i += 1
        if split_emitted:
            table_audit.append({
                "table_index": table_index,
                "status": "PARSED_STRUCTURED_SPLIT_WR_CB_ROWS",
                "alignment": split_alignment,
                "rows_emitted": split_emitted,
            })
            continue

        # 2021 archive tables use TD cells for the first-row labels rather than
        # semantic TH headers. Parse those explicit cells directly before
        # falling back to pandas table inference.
        trs = table.find_all("tr")
        if trs:
            header_cells = [_norm_text(x.get_text(" ", strip=True)).lower()
                            for x in trs[0].find_all(["td","th"])]
            legacy_alignment = None
            if len(header_cells) >= 2:
                if header_cells[0] == "left wr" and header_cells[1] == "right cb":
                    legacy_alignment = "LWR_VS_RCB"
                elif header_cells[0] == "right wr" and header_cells[1] == "left cb":
                    legacy_alignment = "RWR_VS_LCB"
                elif header_cells[0] == "slot wr" and header_cells[1] == "slot cb":
                    legacy_alignment = "SWR_VS_SCB"
            if legacy_alignment:
                emitted = 0
                for tr in trs[1:]:
                    legacy_nodes = tr.find_all(["td","th"])
                    cells = [_norm_text(x.get_text(" ", strip=True))
                             for x in legacy_nodes]
                    if len(cells) < 2:
                        continue
                    wr_pair = _split_embedded_team(cells[0])
                    cb_pair = _split_embedded_team(cells[1])
                    if not wr_pair or not cb_pair:
                        continue
                    _emit_text_pair(
                        rows,
                        season=season, week=week, source_url=source_url,
                        published=published, alignment=legacy_alignment,
                        wr_raw=wr_pair[0], team_raw=wr_pair[1],
                        cb_raw=cb_pair[0], opp_raw=cb_pair[1],
                        matchup_raw="",
                        source_layout="HTML_TABLE_2021_RAW_TD",
                        wr_source_player_id=_source_player_id(legacy_nodes[0]),
                        cb_source_player_id=_source_player_id(legacy_nodes[1]),
                    )
                    emitted += 1
                table_audit.append({
                    "table_index": table_index,
                    "status": "PARSED_LEGACY_RAW_TD_WR_CB_TABLE",
                    "columns": header_cells,
                    "alignment": legacy_alignment,
                    "rows_emitted": emitted,
                })
                continue

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

        # 2021 legacy tables use columns such as "Left WR" / "Right CB" and
        # embed team abbreviation inside each cell ("Christian Kirk ARZ").
        legacy_wr_col = next((c for c in cols if c in {"left wr","right wr","slot wr"}), None)
        legacy_cb_col = next((c for c in cols if c in {"left cb","right cb","slot cb"}), None)
        if legacy_wr_col and legacy_cb_col:
            if legacy_wr_col == "left wr" and legacy_cb_col == "right cb":
                legacy_alignment = "LWR_VS_RCB"
            elif legacy_wr_col == "right wr" and legacy_cb_col == "left cb":
                legacy_alignment = "RWR_VS_LCB"
            elif legacy_wr_col == "slot wr" and legacy_cb_col == "slot cb":
                legacy_alignment = "SWR_VS_SCB"
            else:
                legacy_alignment = "UNKNOWN_ALIGNMENT"
            emitted = 0
            for _, r in df.iterrows():
                wr_pair = _split_embedded_team(r.get(legacy_wr_col, ""))
                cb_pair = _split_embedded_team(r.get(legacy_cb_col, ""))
                if not wr_pair or not cb_pair:
                    continue
                _emit_text_pair(
                    rows,
                    season=season, week=week, source_url=source_url,
                    published=published, alignment=legacy_alignment,
                    wr_raw=wr_pair[0], team_raw=wr_pair[1],
                    cb_raw=cb_pair[0], opp_raw=cb_pair[1],
                    matchup_raw="",
                    source_layout="HTML_TABLE_2021_EMBEDDED_TEAM",
                )
                emitted += 1
            table_audit.append({
                "table_index": table_index,
                "status": "PARSED_LEGACY_EXPLICIT_WR_CB_TABLE",
                "columns": cols,
                "alignment": legacy_alignment,
                "rows_emitted": emitted,
            })
            continue

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
        if not team_col or not opp_col:
            table_audit.append({
                "table_index": table_index,
                "status": "SKIP_NO_EXPLICIT_TEAM_OPP_COLUMNS",
                "columns": cols,
                "alignment": alignment,
            })
            continue

        emitted = 0
        invalid_team_rows = 0
        for _, r in df.iterrows():
            wr_raw = str(r.get(wr_col, "") or "").strip()
            cb_raw = str(r.get(cb_col, "") or "").strip()
            if not wr_raw or not cb_raw:
                continue
            if wr_raw.lower() in {"wide receiver", "wr", "nan", "bye"}:
                continue
            if cb_raw.lower() in {"cornerback", "cb", "nan", "n/a", "bye"}:
                continue

            team_raw = _norm_text(r.get(team_col, "")).upper()
            opp_raw = _norm_text(r.get(opp_col, "")).upper()
            # A factual matchup row must contain actual NFL team abbreviations
            # in BOTH team fields. Newer FantasyAlarm DOM tables can visually
            # resemble the old schema while shifting CB names into the Opp cell;
            # fail closed instead of manufacturing a player-as-team matchup.
            if not _is_team(team_raw) or not _is_team(opp_raw):
                invalid_team_rows += 1
                continue

            wr_name, wr_key = _canon_name(wr_raw)
            cb_name, cb_key = _canon_name(cb_raw)
            team = _source_team(team_raw)
            opponent = _source_team(opp_raw)
            if not team or not opponent or team == opponent:
                invalid_team_rows += 1
                continue
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
                "source_layout": "HTML_TABLE",
            })
            emitted += 1

        table_audit.append({
            "table_index": table_index,
            "status": "PARSED_EXPLICIT_WR_CB_TABLE",
            "columns": cols,
            "alignment": alignment,
            "rows_emitted": emitted,
            "invalid_team_rows_skipped": invalid_team_rows,
        })

    # Text-card parsing is needed for 2025 split-card pages, but must not
    # duplicate 2022-24 pages already recovered from exact HTML table cells.
    text_rows = []
    if structured_table_rows == 0:
        text_rows = _parse_text_cards(
            soup,
            season=season,
            week=week,
            source_url=source_url,
            published=published,
        )
    if text_rows:
        rows.extend(text_rows)

    inline_rows = _parse_inline_2026_pairs(
        soup,
        season=season,
        week=week,
        source_url=source_url,
        published=published,
    )
    if inline_rows:
        rows.extend(inline_rows)

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
        "layout_counts": (
            out["source_layout"].value_counts(dropna=False).to_dict()
            if not out.empty and "source_layout" in out.columns else {}
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
