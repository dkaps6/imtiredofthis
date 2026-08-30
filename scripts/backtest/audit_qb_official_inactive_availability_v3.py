#!/usr/bin/env python3
"""Migration 78 hardened source contract for official game-day inactives.

This is still a SOURCE/DATA-CONTRACT migration. It fits zero predictive models.
The hardened audit addresses four review risks before M79 is authorized:

1. every canonical team-week must have an exact, complete inactive section;
2. every source player bullet is included in the position-parse denominator;
3. coverage is checked inside each actual schedule window, not just season-wide;
4. the current NFL Inactives URL is treated as reachability only unless a real
   parseable team/player payload is present. Live 2026 production remains a
   separate pre-kickoff runtime gate and is not falsely certified in August.

M79, if authorized, consumes only the SHA-pinned frozen M78 snapshot plus
strictly-prior football information. No sportsbook or target-game outcomes are
used here.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

from scripts.backtest import audit_qb_official_inactive_availability as m78

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SEASONS = (2024, 2025)


def _candidate_bullets(ul: Tag) -> list[str]:
    out: list[str] = []
    for li in ul.find_all("li"):
        text = m78.norm_space(li.get_text(" ", strip=True))
        if not text:
            continue
        if text.upper().startswith(("WHERE:", "WHEN:", "TV:", "WATCH:")):
            continue
        out.append(text)
    return out


def _labels_before_ul(label: Tag, ul: Tag) -> list[str]:
    teams: list[str] = []
    for el in label.next_elements:
        if el is ul:
            break
        if isinstance(el, Tag) and el is not label:
            team = m78.team_from_heading(el.get_text(" ", strip=True))
            if team:
                teams.append(team)
    return teams


def _team_section_candidates(soup: BeautifulSoup) -> list[tuple[str, Tag, Tag]]:
    """Return candidate (team,label,ul) sections, including NFL plain-text labels."""
    found: list[tuple[str, Tag, Tag]] = []
    seen: set[tuple[str, int]] = set()

    # Headings are preferred, but some archived NFL pages render one team label
    # as ordinary text. Scanning exact team-label tags covers both schemas.
    labels = list(soup.find_all(["h2", "h3", "h4"])) + list(soup.find_all(True))
    for label in labels:
        team = m78.team_from_heading(label.get_text(" ", strip=True))
        if not team:
            continue
        ul = label.find_next("ul")
        if ul is None:
            continue
        key = (team, id(ul))
        if key in seen:
            continue
        # Reject matchup-display labels when another team label occurs before
        # the list. The list must belong to this label, not the next opponent.
        intervening = [t for t in _labels_before_ul(label, ul) if t != team]
        if intervening:
            continue
        candidates = _candidate_bullets(ul)
        if len(candidates) < 3:
            continue
        seen.add(key)
        found.append((team, label, ul))
    return found


def parse_article_hardened(url: str, season: int, snapshots: list[dict]) -> tuple[list[dict], list[dict], dict | None]:
    try:
        r = m78.request(url)
    except Exception as exc:
        snapshots.append({
            "kind": "inactive_article_hardened", "season": season, "url": url,
            "status": 0, "sha256": "", "error": f"{type(exc).__name__}:{exc}",
        })
        return [], [], None

    snapshots.append({
        "kind": "inactive_article_hardened", "season": season, "url": r.url,
        "status": r.status_code, "sha256": m78.sha256_bytes(r.content), "error": "",
    })
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    title = m78.norm_space(h1.get_text(" ", strip=True) if h1 else soup.title.get_text(" ", strip=True) if soup.title else "")
    week = m78.extract_week(title)
    if week is None:
        return [], [], {"season": season, "week": None, "url": r.url, "title": title}

    records: list[dict] = []
    sections: list[dict] = []
    for team, _label, ul in _team_section_candidates(soup):
        bullets = _candidate_bullets(ul)
        parsed = [m78.parse_player_bullet(x) for x in bullets]
        parsed_ok = [x for x in parsed if x is not None]
        complete = bool(len(bullets) >= 3 and len(parsed_ok) == len(bullets))
        section_id = f"{season}:{week}:{team}:{len(sections)}"
        sections.append({
            "section_id": section_id,
            "season": season,
            "week": week,
            "team": team,
            "article_url": r.url,
            "article_title": title,
            "candidate_bullets": len(bullets),
            "parsed_bullets": len(parsed_ok),
            "position_parse_rate": (len(parsed_ok) / len(bullets)) if bullets else 0.0,
            "section_complete": complete,
        })
        for pos, name in parsed_ok:
            records.append({
                "section_id": section_id,
                "season": season,
                "week": week,
                "team": team,
                "inactive_name": name,
                "inactive_name_key": m78.norm_name(name),
                "listed_position": pos,
                "article_url": r.url,
                "section_complete": complete,
            })
    return records, sections, {"season": season, "week": week, "url": r.url, "title": title}


def load_frozen() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        p = Path(f"/tmp/m78_official_inactives_teamweek_{season}.csv")
        if not p.exists():
            raise RuntimeError(f"verified frozen snapshot missing from workflow: {p}")
        q = pd.read_csv(p, low_memory=False)
        q["season"] = pd.to_numeric(q.season).astype(int)
        q["week"] = pd.to_numeric(q.week).astype(int)
        q["team"] = q.team.map(m78.canon)
        frames.append(q)
    frozen = pd.concat(frames, ignore_index=True)
    if frozen.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate frozen team-week")
    return frozen


def load_schedule(snapshots: list[dict]) -> pd.DataFrame:
    r = m78.request(SCHEDULE_URL, timeout=90)
    snapshots.append({
        "kind": "schedule", "season": 0, "url": r.url, "status": r.status_code,
        "sha256": m78.sha256_bytes(r.content), "error": "",
    })
    df = pd.read_csv(BytesIO(r.content), low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    needed = {"season", "week", "home_team", "away_team", "gameday"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"schedule schema missing {sorted(needed - set(df.columns))}")
    if "game_type" in df.columns:
        df = df.loc[df.game_type.astype(str).str.upper().eq("REG")].copy()
    df = df.loc[pd.to_numeric(df.season, errors="coerce").isin(SEASONS)].copy()
    df["season"] = pd.to_numeric(df.season).astype(int)
    df["week"] = pd.to_numeric(df.week).astype(int)
    df["home_team"] = df.home_team.map(m78.canon)
    df["away_team"] = df.away_team.map(m78.canon)
    day = pd.to_datetime(df.gameday, errors="coerce").dt.day_name().fillna("Unknown")
    times = df["gametime"].fillna("").astype(str) if "gametime" in df.columns else pd.Series("", index=df.index)

    def window(d: str, t: str) -> str:
        if d == "Sunday":
            try:
                hh, mm = [int(x) for x in t.split(":")[:2]]
                minutes = hh * 60 + mm
                if minutes < 15 * 60 + 30:
                    return "SUNDAY_EARLY"
                if minutes < 19 * 60:
                    return "SUNDAY_LATE"
                return "SUNDAY_NIGHT"
            except Exception:
                return "SUNDAY_UNKNOWN_TIME"
        return d.upper()

    df["game_window"] = [window(d, t) for d, t in zip(day, times)]
    home = df[["season", "week", "home_team", "game_window", "gameday"]].rename(columns={"home_team": "team"})
    away = df[["season", "week", "away_team", "game_window", "gameday"]].rename(columns={"away_team": "team"})
    teamweeks = pd.concat([home, away], ignore_index=True)
    if teamweeks.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("schedule duplicate team-week")
    return teamweeks


def live_endpoint_runtime_status(snapshots: list[dict]) -> tuple[bool, bool, int, int, str]:
    """Reachability is distinct from a validated game-day payload."""
    try:
        r = m78.request(m78.LIVE_INACTIVES_URL)
        snapshots.append({
            "kind": "live_2026_endpoint_hardened", "season": 2026, "url": r.url,
            "status": r.status_code, "sha256": m78.sha256_bytes(r.content), "error": "",
        })
        soup = BeautifulSoup(r.text, "html.parser")
        sections = _team_section_candidates(soup)
        teams = set()
        players = 0
        for team, _label, ul in sections:
            bullets = _candidate_bullets(ul)
            parsed = [m78.parse_player_bullet(x) for x in bullets]
            ok = [x for x in parsed if x is not None]
            if len(ok) >= 3 and len(ok) == len(bullets):
                teams.add(team)
                players += len(ok)
        payload_valid = bool(len(teams) >= 2 and players >= 6)
        detail = "parseable_game_day_payload_present" if payload_valid else "endpoint_reachable_but_no_validated_game_day_payload_yet"
        return True, payload_valid, len(teams), players, detail
    except Exception as exc:
        snapshots.append({
            "kind": "live_2026_endpoint_hardened", "season": 2026,
            "url": m78.LIVE_INACTIVES_URL, "status": 0, "sha256": "",
            "error": f"{type(exc).__name__}:{exc}",
        })
        return False, False, 0, 0, f"{type(exc).__name__}:{exc}"


def _best_section_matches(records: pd.DataFrame, sections: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Select a complete section whose exact identities/count match the frozen row."""
    rows = []
    if sections.empty:
        return pd.DataFrame(columns=["season", "week", "team", "source_section_complete", "source_exact_identity_match", "source_candidate_bullets", "source_parsed_bullets"])

    for fr in frozen.itertuples(index=False):
        key = (int(fr.season), int(fr.week), str(fr.team))
        sq = sections.loc[
            sections.season.eq(key[0]) & sections.week.eq(key[1]) & sections.team.eq(key[2])
        ]
        best = None
        for sec in sq.itertuples(index=False):
            rq = records.loc[records.section_id.eq(sec.section_id)]
            names = rq.inactive_name_key.dropna().astype(str).tolist()
            unique_names = set(names)
            count_match = len(names) == int(fr.inactive_count) and len(unique_names) == len(names)
            token_text = str(fr.inactive_tokens)
            names_match = count_match and all(name and name in token_text for name in unique_names)
            exact = bool(sec.section_complete and names_match)
            candidate = {
                "season": key[0], "week": key[1], "team": key[2],
                "source_section_complete": bool(sec.section_complete),
                "source_exact_identity_match": exact,
                "source_candidate_bullets": int(sec.candidate_bullets),
                "source_parsed_bullets": int(sec.parsed_bullets),
            }
            if best is None or (candidate["source_exact_identity_match"], candidate["source_parsed_bullets"]) > (best["source_exact_identity_match"], best["source_parsed_bullets"]):
                best = candidate
        if best is None:
            best = {
                "season": key[0], "week": key[1], "team": key[2],
                "source_section_complete": False,
                "source_exact_identity_match": False,
                "source_candidate_bullets": 0,
                "source_parsed_bullets": 0,
            }
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = m78.require_canonical(Path(args.canonical))
    targets = m78.canonical_targets(base)
    frozen = load_frozen()
    snapshots: list[dict] = []

    all_records: list[dict] = []
    all_sections: list[dict] = []
    article_rows: list[dict] = []
    discovered: dict[int, list[str]] = {}
    for season in SEASONS:
        urls = m78.discover_article_urls(season, snapshots)
        discovered[season] = urls
        for url in urls:
            recs, secs, meta = parse_article_hardened(url, season, snapshots)
            all_records.extend(recs)
            all_sections.extend(secs)
            if meta:
                article_rows.append(meta)

    records = pd.DataFrame(all_records)
    sections = pd.DataFrame(all_sections)
    if records.empty:
        records = pd.DataFrame(columns=["section_id", "season", "week", "team", "inactive_name", "inactive_name_key", "listed_position", "article_url", "section_complete"])
    if sections.empty:
        sections = pd.DataFrame(columns=["section_id", "season", "week", "team", "article_url", "article_title", "candidate_bullets", "parsed_bullets", "position_parse_rate", "section_complete"])

    source_match = _best_section_matches(records, sections, frozen)
    coverage = targets.merge(source_match, on=["season", "week", "team"], how="left")
    for c in ["source_section_complete", "source_exact_identity_match"]:
        coverage[c] = coverage[c].fillna(False).astype(bool)
    for c in ["source_candidate_bullets", "source_parsed_bullets"]:
        coverage[c] = coverage[c].fillna(0).astype(int)

    # Identity bridge is measured from all exact source records against weekly
    # rosters, but only after source completeness is established.
    roster_indexes = {
        season: m78.roster_name_index(m78.load_weekly_rosters(season, snapshots), season)
        for season in SEASONS
    }
    if len(records):
        records["roster_identity_match"] = [
            row.inactive_name_key in roster_indexes.get(int(row.season), {}).get((int(row.week), str(row.team)), set())
            for row in records.itertuples(index=False)
        ]
    else:
        records["roster_identity_match"] = pd.Series(dtype=bool)

    schedule = load_schedule(snapshots)
    coverage = coverage.merge(schedule, on=["season", "week", "team"], how="left", validate="one_to_one")
    schedule_map_ok = coverage.game_window.notna()

    live_reachable, live_payload_validated, live_teams, live_players, live_detail = live_endpoint_runtime_status(snapshots)

    gate_rows: list[dict] = []
    def gate(name: str, value: float, threshold: str, passed: bool, scope: str = "historical_m79"):
        gate_rows.append({"gate": name, "value": float(value), "threshold": threshold, "passed": bool(passed), "scope": scope})

    for season in SEASONS:
        cq = coverage.loc[coverage.season.eq(season)]
        fq = frozen.loc[frozen.season.eq(season)]
        exact_cov = float(cq.source_exact_identity_match.mean()) if len(cq) else 0.0
        sched_cov = float(cq.game_window.notna().mean()) if len(cq) else 0.0

        # Correct denominator: every candidate player bullet in the selected
        # source sections, including bullets whose position failed to parse.
        cand = int(cq.source_candidate_bullets.sum())
        parsed = int(cq.source_parsed_bullets.sum())
        pos_rate = (parsed / cand) if cand else 0.0

        rq = records.loc[records.season.eq(season)]
        bridge = float(rq.roster_identity_match.mean()) if len(rq) else 0.0
        weeks = int(cq.loc[cq.source_exact_identity_match, "week"].nunique())

        gate(f"frozen_team_week_rows_{season}", len(fq), "==544", len(fq) == 544)
        gate(f"canonical_exact_identity_coverage_{season}", exact_cov, "==1.0", exact_cov == 1.0)
        gate(f"schedule_mapping_coverage_{season}", sched_cov, "==1.0", sched_cov == 1.0)
        gate(f"candidate_bullet_position_parse_{season}", pos_rate, ">=0.95", pos_rate >= 0.95)
        gate(f"roster_identity_bridge_{season}", bridge, ">=0.90", bridge >= 0.90)
        gate(f"regular_week_exact_identity_coverage_{season}", weeks, "==18", weeks == 18)

        # Explicit window gates prevent an entire Monday/Thursday/special class
        # from disappearing while a season-wide percentage still looks healthy.
        for window_name, wq in cq.groupby("game_window", dropna=False):
            win_cov = float(wq.source_exact_identity_match.mean()) if len(wq) else 0.0
            gate(f"window_exact_identity_{season}_{window_name}", win_cov, "==1.0", win_cov == 1.0)

    # Live state is deliberately not part of historical M79 authorization.
    # It must be rechecked before a 2026 game can use the feature in production.
    gate("live_2026_endpoint_reachable", 1.0 if live_reachable else 0.0, "==1", live_reachable, scope="live_runtime")
    gate("live_2026_game_day_payload_validated", 1.0 if live_payload_validated else 0.0, "==1_before_production_use", live_payload_validated, scope="live_runtime")

    gates = pd.DataFrame(gate_rows)
    historical = gates.loc[gates.scope.eq("historical_m79")]
    m79_authorized = bool(len(historical) and historical.passed.all())
    status = "QUALIFIED_FROZEN_OFFICIAL_INACTIVE_IDENTITY_FOR_M79" if m79_authorized else "HISTORICAL_SOURCE_CONTRACT_NOT_QUALIFIED"
    next_step = "M79_one_frozen_official_inactive_predictive_test" if m79_authorized else "repair_historical_source_contract_before_M79"

    interpretation = pd.DataFrame([{
        "migration": "M78",
        "status": status,
        "production_actionable": False,
        "predictive_model_fit": False,
        "canonical_rows": len(base),
        "canonical_target_team_weeks": len(targets),
        "frozen_team_weeks": len(frozen),
        "m79_authorized": m79_authorized,
        "live_2026_endpoint_reachable": live_reachable,
        "live_2026_payload_validated": live_payload_validated,
        "live_2026_parsed_teams": live_teams,
        "live_2026_parsed_players": live_players,
        "live_2026_detail": live_detail,
        "next_step": next_step,
    }])

    contract = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_sha256": m78.CANONICAL_SHA256,
        "sportsbook_used": False,
        "target_game_performance_used": False,
        "predictive_model_fit": False,
        "historical_snapshot_frozen_and_sha_verified_by_workflow": True,
        "historical_m79_authorized": m79_authorized,
        "historical_gate_requires_exact_canonical_team_week_identity": True,
        "historical_gate_requires_every_observed_schedule_window": True,
        "position_parse_denominator_is_all_candidate_source_bullets": True,
        "live_2026_endpoint_reachable": live_reachable,
        "live_2026_payload_validated": live_payload_validated,
        "live_2026_production_rule": "must_validate_real_game_day_payload_and_snapshot_before_that_game_kickoff",
    }

    records.to_csv(out_dir / "m78_hardened_source_records.csv", index=False)
    sections.to_csv(out_dir / "m78_hardened_source_sections.csv", index=False)
    coverage.to_csv(out_dir / "m78_hardened_canonical_coverage.csv", index=False)
    pd.DataFrame(article_rows).to_csv(out_dir / "m78_hardened_article_manifest.csv", index=False)
    pd.DataFrame(snapshots).to_csv(out_dir / "m78_hardened_source_snapshots.csv", index=False)
    gates.to_csv(out_dir / "m78_source_gate.csv", index=False)
    interpretation.to_csv(out_dir / "m78_interpretation.csv", index=False)
    (out_dir / "m78_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    print("=== M78 HARDENED INTERPRETATION ===")
    print(interpretation.to_string(index=False))
    print("\n=== M78 HARDENED GATES ===")
    print(gates.to_string(index=False))
    print("\n=== WINDOW COVERAGE ===")
    print(coverage.groupby(["season", "game_window"], dropna=False).source_exact_identity_match.agg(["count", "sum", "mean"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
