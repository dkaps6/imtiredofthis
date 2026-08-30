#!/usr/bin/env python3
"""M78 parser hardening for NFL.com plain-text team labels.

Some 2024 inactive articles render one team label as plain text instead of an
H2/H3/H4 heading (observed for NYG W1, HOU W3, PHI W11, LAC W12). The source
records are present; this wrapper recovers only those missing sections without
changing the M78 source gate or any predictive logic.
"""
from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from scripts.backtest import audit_qb_official_inactive_availability as m78

_original_parse = m78.parse_inactive_article


def _fallback_plain_team_sections(url: str, season: int, existing: list[dict]) -> list[dict]:
    existing_teams = {str(r["team"]) for r in existing}
    try:
        response = m78.request(url)
    except Exception:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    h1 = soup.find("h1")
    title = m78.norm_space(h1.get_text(" ", strip=True) if h1 else soup.title.get_text(" ", strip=True) if soup.title else "")
    week = m78.extract_week(title)
    if week is None:
        return []

    recovered: list[dict] = []
    recovered_teams: set[str] = set()
    for tag in soup.find_all(True):
        text = m78.norm_space(tag.get_text(" ", strip=True))
        # Parent containers include much more than a team name and therefore
        # will not pass the exact alias matcher.
        team = m78.team_from_heading(text)
        if not team or team in existing_teams or team in recovered_teams:
            continue
        ul = tag.find_next("ul")
        if ul is None:
            continue

        # A matchup-display team name can precede the opponent/team inactive
        # heading. Reject the candidate if another recognized team label occurs
        # before the next list.
        blocked = False
        for el in tag.next_elements:
            if el is ul:
                break
            if isinstance(el, Tag) and el is not tag:
                other = m78.team_from_heading(el.get_text(" ", strip=True))
                if other and other != team:
                    blocked = True
                    break
        if blocked:
            continue

        rows = []
        for li in ul.find_all("li"):
            parsed = m78.parse_player_bullet(li.get_text(" ", strip=True))
            if parsed:
                rows.append(parsed)
        # Official inactive lists contain several players; this prevents a
        # stray single bullet elsewhere on the article from qualifying.
        if len(rows) < 3:
            continue

        recovered_teams.add(team)
        for pos, name in rows:
            recovered.append({
                "season": season,
                "week": week,
                "team": team,
                "inactive_name": name,
                "inactive_name_key": m78.norm_name(name),
                "listed_position": pos,
                "article_title": title,
                "article_url": response.url,
            })
    return recovered


def parse_inactive_article(url: str, season: int, snapshots: list[dict]):
    records, meta = _original_parse(url, season, snapshots)
    if not meta or meta.get("week") is None:
        return records, meta
    # Legitimate NFL matchups always contribute two team lists. An odd number
    # of parsed sections is the signature of the plain-label formatting defect.
    if int(meta.get("team_sections", 0)) % 2 == 0:
        return records, meta
    recovered = _fallback_plain_team_sections(url, season, records)
    if not recovered:
        return records, meta
    records = records + recovered
    meta = dict(meta)
    meta["team_sections"] = int(meta.get("team_sections", 0)) + len({r["team"] for r in recovered})
    meta["players"] = int(meta.get("players", 0)) + len(recovered)
    return records, meta


m78.parse_inactive_article = parse_inactive_article

if __name__ == "__main__":
    raise SystemExit(m78.main())
