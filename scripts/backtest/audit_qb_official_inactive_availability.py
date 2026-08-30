#!/usr/bin/env python3
"""Migration 78: official game-day inactive identity acquisition frontier.

M78 is a source/data-contract audit only. It fits no predictive model.
It reconstructs the NFL's official game-day inactive lists for the 2024 and
2025 regular seasons from archived NFL.com inactive-report articles, checks
coverage against the frozen canonical-v3 QB cohort, bridges inactive names to
weekly rosters, and verifies that the live NFL Inactives endpoint remains a
usable 2026 acquisition path.

Important semantic boundary:
- Historical NFL.com articles may be archived/updated after a game, but the
  inactive designation itself is a fixed pre-kickoff fact announced on game
  day. M78 uses only that identity/status fact; no postgame outcome is read.
- Future 2026 production acquisition must snapshot the live NFL Inactives page
  before kickoff. Historical article timestamps are not treated as features.
- No sportsbook information and no target-game performance fields are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from scripts._opponent_map import canon_team

CANONICAL_SHA256 = "c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
EXPECTED_ROWS = 884
EXPECTED_SEASONS = {2024: 444, 2025: 440}
UA = "imtiredofthis-m78-official-inactive-audit/1.0"
LIVE_INACTIVES_URL = "https://www.nfl.com/inactives/"

TEAM_ALIASES = {
    "ARI": ["ARIZONA CARDINALS", "CARDINALS"],
    "ATL": ["ATLANTA FALCONS", "FALCONS"],
    "BAL": ["BALTIMORE RAVENS", "RAVENS"],
    "BUF": ["BUFFALO BILLS", "BILLS"],
    "CAR": ["CAROLINA PANTHERS", "PANTHERS"],
    "CHI": ["CHICAGO BEARS", "BEARS"],
    "CIN": ["CINCINNATI BENGALS", "BENGALS"],
    "CLE": ["CLEVELAND BROWNS", "BROWNS"],
    "DAL": ["DALLAS COWBOYS", "COWBOYS"],
    "DEN": ["DENVER BRONCOS", "BRONCOS"],
    "DET": ["DETROIT LIONS", "LIONS"],
    "GB": ["GREEN BAY PACKERS", "PACKERS"],
    "HOU": ["HOUSTON TEXANS", "TEXANS"],
    "IND": ["INDIANAPOLIS COLTS", "COLTS"],
    "JAX": ["JACKSONVILLE JAGUARS", "JAGUARS"],
    "KC": ["KANSAS CITY CHIEFS", "CHIEFS"],
    "LV": ["LAS VEGAS RAIDERS", "RAIDERS"],
    "LAC": ["LOS ANGELES CHARGERS", "L.A. CHARGERS", "CHARGERS"],
    "LAR": ["LOS ANGELES RAMS", "L.A. RAMS", "RAMS"],
    "MIA": ["MIAMI DOLPHINS", "DOLPHINS"],
    "MIN": ["MINNESOTA VIKINGS", "VIKINGS"],
    "NE": ["NEW ENGLAND PATRIOTS", "PATRIOTS"],
    "NO": ["NEW ORLEANS SAINTS", "SAINTS"],
    "NYG": ["NEW YORK GIANTS", "N.Y. GIANTS", "GIANTS"],
    "NYJ": ["NEW YORK JETS", "N.Y. JETS", "JETS"],
    "PHI": ["PHILADELPHIA EAGLES", "EAGLES"],
    "PIT": ["PITTSBURGH STEELERS", "STEELERS"],
    "SEA": ["SEATTLE SEAHAWKS", "SEAHAWKS"],
    "SF": ["SAN FRANCISCO 49ERS", "49ERS", "NINERS"],
    "TB": ["TAMPA BAY BUCCANEERS", "BUCCANEERS", "BUCS"],
    "TEN": ["TENNESSEE TITANS", "TITANS"],
    "WAS": ["WASHINGTON COMMANDERS", "COMMANDERS"],
}

POSITION_TOKENS = {
    "QB", "RB", "FB", "HB", "WR", "TE", "OL", "OT", "T", "LT", "RT",
    "OG", "G", "LG", "RG", "C", "DL", "DT", "NT", "DE", "EDGE", "OLB",
    "LB", "ILB", "CB", "DB", "S", "FS", "SS", "K", "P", "LS",
}

SEASON_MONTHS = {
    2024: [(2024, 9), (2024, 10), (2024, 11), (2024, 12), (2025, 1)],
    2025: [(2025, 9), (2025, 10), (2025, 11), (2025, 12), (2026, 1)],
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def norm_space(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def ascii_fold(s: object) -> str:
    x = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in x if not unicodedata.combining(c))


def norm_heading(s: object) -> str:
    s = ascii_fold(s).upper()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^A-Z0-9.' -]+", " ", s)
    return norm_space(s)


def norm_name(s: object) -> str:
    s = ascii_fold(s).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("’", "'")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def team_from_heading(text: object) -> str | None:
    h = norm_heading(text)
    # Remove records occasionally rendered next to team names.
    h = re.sub(r"\b\d{1,2}-\d{1,2}(?:-\d)?\b", " ", h)
    h = norm_space(h)
    for team, aliases in TEAM_ALIASES.items():
        if h in aliases:
            return team
    return None


def canon(v: object) -> str:
    try:
        return canon_team(v)
    except Exception:
        return str(v).strip().upper()


def require_canonical(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != CANONICAL_SHA256:
        raise RuntimeError(f"canonical-v3 SHA drift: {digest}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"canonical row drift: {len(df)}")
    counts = {int(k): int(v) for k, v in df.season.value_counts().to_dict().items()}
    if counts != EXPECTED_SEASONS:
        raise RuntimeError(f"canonical season drift: {counts}")
    forbidden = [c for c in df.columns if any(t in c for t in ("market", "spread", "sportsbook", "moneyline", "game_total"))]
    if forbidden:
        raise RuntimeError(f"market boundary violated: {forbidden}")
    df["team"] = df.team.map(canon)
    df["opponent"] = df.opponent.map(canon)
    df["season"] = pd.to_numeric(df.season).astype(int)
    df["week"] = pd.to_numeric(df.week).astype(int)
    return df


def request(url: str, timeout: int = 45) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=timeout)
    r.raise_for_status()
    return r


def discover_article_urls(season: int, snapshots: list[dict]) -> list[str]:
    urls: set[str] = set()
    for year, month in SEASON_MONTHS[season]:
        sitemap = f"https://www.nfl.com/sitemap/html/articles/{year}/{month}"
        try:
            r = request(sitemap)
            snapshots.append({
                "kind": "sitemap", "season": season, "url": r.url,
                "status": r.status_code, "sha256": sha256_bytes(r.content), "error": "",
            })
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                txt = norm_space(a.get_text(" ", strip=True))
                href = urljoin("https://www.nfl.com", a.get("href", ""))
                if "inactive" not in txt.lower() and "inactive" not in href.lower():
                    continue
                if "/news/" not in href:
                    continue
                parsed = urlparse(href)
                path = parsed.path
                amp = f"https://amp.nfl.com{path}"
                urls.add(amp)
        except Exception as exc:
            snapshots.append({
                "kind": "sitemap", "season": season, "url": sitemap,
                "status": 0, "sha256": "", "error": f"{type(exc).__name__}:{exc}",
            })
    return sorted(urls)


def extract_week(title: str) -> int | None:
    m = re.search(r"\bWeek\s+(\d{1,2})\b", title, flags=re.I)
    if not m:
        return None
    w = int(m.group(1))
    return w if 1 <= w <= 18 else None


def parse_player_bullet(text: object) -> tuple[str, str] | None:
    s = norm_space(text)
    if not s:
        return None
    if s.upper().startswith(("WHERE:", "WHEN:", "TV:", "WATCH:")):
        return None
    s = re.sub(r"\s*\([^)]*(?:QB|quarterback|inactive|injury|third)[^)]*\)\s*$", "", s, flags=re.I)
    parts = s.split()
    if len(parts) < 2:
        return None
    pos = parts[0].upper().rstrip(".:")
    if pos not in POSITION_TOKENS:
        return None
    name = norm_space(" ".join(parts[1:]))
    name = re.sub(r"\s*[-–—]\s*[A-Z]$", "", name)
    if len(norm_name(name)) < 3:
        return None
    return pos, name


def parse_inactive_article(url: str, season: int, snapshots: list[dict]) -> tuple[list[dict], dict | None]:
    try:
        r = request(url)
    except Exception as exc:
        snapshots.append({
            "kind": "inactive_article", "season": season, "url": url,
            "status": 0, "sha256": "", "error": f"{type(exc).__name__}:{exc}",
        })
        return [], None

    snapshots.append({
        "kind": "inactive_article", "season": season, "url": r.url,
        "status": r.status_code, "sha256": sha256_bytes(r.content), "error": "",
    })
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    title = norm_space(h1.get_text(" ", strip=True) if h1 else soup.title.get_text(" ", strip=True) if soup.title else "")
    week = extract_week(title)
    if week is None:
        return [], {"season": season, "week": None, "title": title, "url": r.url, "team_sections": 0, "players": 0}

    records: list[dict] = []
    team_sections = 0
    seen_section: set[tuple[str, int]] = set()
    headings = soup.find_all(["h2", "h3", "h4"])
    for h in headings:
        team = team_from_heading(h.get_text(" ", strip=True))
        if not team:
            continue
        key = (team, id(h))
        if key in seen_section:
            continue
        seen_section.add(key)
        section_rows = []
        for el in h.next_elements:
            if el is h:
                continue
            if isinstance(el, Tag) and el.name in {"h2", "h3", "h4"}:
                break
            if isinstance(el, Tag) and el.name == "li":
                parsed = parse_player_bullet(el.get_text(" ", strip=True))
                if parsed:
                    section_rows.append(parsed)
        if not section_rows:
            continue
        team_sections += 1
        for pos, name in section_rows:
            records.append({
                "season": season, "week": week, "team": team,
                "inactive_name": name, "inactive_name_key": norm_name(name),
                "listed_position": pos, "article_title": title, "article_url": r.url,
            })

    meta = {"season": season, "week": week, "title": title, "url": r.url, "team_sections": team_sections, "players": len(records)}
    return records, meta


def release_urls(season: int) -> list[str]:
    root = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters"
    return [f"{root}/roster_weekly_{season}.parquet", f"{root}/roster_weekly_{season}.csv"]


def load_weekly_rosters(season: int, snapshots: list[dict]) -> pd.DataFrame:
    errors = []
    for url in release_urls(season):
        try:
            r = request(url, timeout=90)
            snapshots.append({
                "kind": "weekly_roster", "season": season, "url": r.url,
                "status": r.status_code, "sha256": sha256_bytes(r.content), "error": "",
            })
            from io import BytesIO
            if url.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(r.content))
            else:
                df = pd.read_csv(BytesIO(r.content), low_memory=False)
            df.columns = [str(c).lower() for c in df.columns]
            if "season" in df.columns:
                df = df.loc[pd.to_numeric(df.season, errors="coerce").eq(season)].copy()
            return df
        except Exception as exc:
            errors.append(f"{url}:{type(exc).__name__}:{exc}")
    snapshots.append({
        "kind": "weekly_roster", "season": season, "url": "|".join(release_urls(season)),
        "status": 0, "sha256": "", "error": " || ".join(errors),
    })
    return pd.DataFrame()


def roster_name_index(df: pd.DataFrame, season: int) -> dict[tuple[int, str], set[str]]:
    if df.empty:
        return {}
    team_col = next((c for c in ["team", "club_code"] if c in df.columns), None)
    week_col = "week" if "week" in df.columns else None
    if not team_col or not week_col:
        return {}
    name_cols = [c for c in ["full_name", "player_name", "football_name", "display_name"] if c in df.columns]
    if not name_cols and {"first_name", "last_name"}.issubset(df.columns):
        df = df.copy()
        df["_full"] = df.first_name.fillna("").astype(str) + " " + df.last_name.fillna("").astype(str)
        name_cols = ["_full"]
    if not name_cols:
        return {}
    out: dict[tuple[int, str], set[str]] = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        try:
            week = int(float(d[week_col]))
        except Exception:
            continue
        team = canon(d[team_col])
        key = (week, team)
        names = out.setdefault(key, set())
        for c in name_cols:
            v = d.get(c)
            k = norm_name(v)
            if k:
                names.add(k)
    return out


def live_endpoint_check(snapshots: list[dict]) -> tuple[bool, str]:
    try:
        r = request(LIVE_INACTIVES_URL)
        snapshots.append({
            "kind": "live_2026_endpoint", "season": 2026, "url": r.url,
            "status": r.status_code, "sha256": sha256_bytes(r.content), "error": "",
        })
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True).lower()
        semantic = "inactive" in text
        return bool(r.status_code == 200 and semantic), "http_200_and_inactive_semantics" if semantic else "http_200_but_semantics_missing"
    except Exception as exc:
        snapshots.append({
            "kind": "live_2026_endpoint", "season": 2026, "url": LIVE_INACTIVES_URL,
            "status": 0, "sha256": "", "error": f"{type(exc).__name__}:{exc}",
        })
        return False, f"{type(exc).__name__}:{exc}"


def canonical_targets(base: pd.DataFrame) -> pd.DataFrame:
    a = base[["season", "week", "team"]].copy()
    b = base[["season", "week", "opponent"]].rename(columns={"opponent": "team"})
    return pd.concat([a, b], ignore_index=True).drop_duplicates(["season", "week", "team"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = require_canonical(Path(args.canonical))
    targets = canonical_targets(base)
    snapshots: list[dict] = []

    all_records: list[dict] = []
    article_meta: list[dict] = []
    discovered: dict[int, list[str]] = {}
    for season in [2024, 2025]:
        urls = discover_article_urls(season, snapshots)
        discovered[season] = urls
        for url in urls:
            recs, meta = parse_inactive_article(url, season, snapshots)
            all_records.extend(recs)
            if meta:
                article_meta.append(meta)

    inactive = pd.DataFrame(all_records)
    if inactive.empty:
        inactive = pd.DataFrame(columns=["season", "week", "team", "inactive_name", "inactive_name_key", "listed_position", "article_title", "article_url"])
    else:
        inactive = inactive.drop_duplicates(["season", "week", "team", "inactive_name_key"]).sort_values(["season", "week", "team", "inactive_name_key"]).reset_index(drop=True)

    roster_indexes = {}
    for season in [2024, 2025]:
        roster_indexes[season] = roster_name_index(load_weekly_rosters(season, snapshots), season)

    if len(inactive):
        inactive["roster_identity_match"] = [
            row.inactive_name_key in roster_indexes.get(int(row.season), {}).get((int(row.week), str(row.team)), set())
            for row in inactive.itertuples(index=False)
        ]
    else:
        inactive["roster_identity_match"] = pd.Series(dtype=bool)

    team_week_counts = inactive.groupby(["season", "week", "team"]).size().rename("inactive_count").reset_index() if len(inactive) else pd.DataFrame(columns=["season", "week", "team", "inactive_count"])
    coverage = targets.merge(team_week_counts, on=["season", "week", "team"], how="left")
    coverage["inactive_team_week_found"] = coverage.inactive_count.notna()
    coverage["inactive_count"] = coverage.inactive_count.fillna(0).astype(int)

    live_ok, live_detail = live_endpoint_check(snapshots)
    meta_df = pd.DataFrame(article_meta)

    gate_rows = []
    def gate(name: str, value: float, threshold: str, passed: bool):
        gate_rows.append({"gate": name, "value": float(value), "threshold": threshold, "passed": bool(passed)})

    for season in [2024, 2025]:
        q = coverage.loc[coverage.season.eq(season)]
        cov = float(q.inactive_team_week_found.mean()) if len(q) else 0.0
        iq = inactive.loc[inactive.season.eq(season)]
        bridge = float(iq.roster_identity_match.mean()) if len(iq) else 0.0
        pos = float(iq.listed_position.notna().mean()) if len(iq) else 0.0
        article_count = int(meta_df.loc[(meta_df.season.eq(season)) & meta_df.week.notna() & meta_df.team_sections.gt(0), "url"].nunique()) if len(meta_df) else 0
        weeks = int(meta_df.loc[(meta_df.season.eq(season)) & meta_df.team_sections.gt(0), "week"].nunique()) if len(meta_df) else 0
        gate(f"canonical_team_week_coverage_{season}", cov, ">=0.90", cov >= 0.90)
        gate(f"roster_identity_bridge_{season}", bridge, ">=0.90", bridge >= 0.90)
        gate(f"listed_position_parse_{season}", pos, ">=0.95", pos >= 0.95)
        gate(f"regular_week_discovery_{season}", weeks, ">=17", weeks >= 17)
        gate(f"official_inactive_article_count_{season}", article_count, ">=18", article_count >= 18)
    gate("live_2026_nfl_inactives_endpoint", 1.0 if live_ok else 0.0, "==1", live_ok)

    gates = pd.DataFrame(gate_rows)
    authorized = bool(len(gates) and gates.passed.all())
    status = "QUALIFIED_OFFICIAL_INACTIVE_IDENTITY_LAYER" if authorized else "SOURCE_CONTRACT_NOT_YET_QUALIFIED"
    next_step = "M79_official_game_day_inactive_identity_test" if authorized else "repair_or_replace_official_inactive_acquisition_before_predictive_test"

    interpretation = pd.DataFrame([{
        "migration": "M78",
        "status": status,
        "production_actionable": False,
        "predictive_model_fit": False,
        "canonical_rows": len(base),
        "target_team_weeks": len(targets),
        "inactive_player_rows": len(inactive),
        "discovered_article_urls_2024": len(discovered.get(2024, [])),
        "discovered_article_urls_2025": len(discovered.get(2025, [])),
        "live_2026_endpoint_ok": live_ok,
        "next_step": next_step,
        "historical_semantics": "retrospective_archive_of_fixed_pre_kickoff_official_inactive_fact",
        "future_2026_semantics": "snapshot_live_nfl_inactives_before_kickoff",
    }])

    no_retest = pd.DataFrame([
        {"family": "depth_chart_personnel_discontinuity", "prior": "M77", "status": "DO_NOT_RETEST_WITH_DIFFERENT_MODEL", "reopen_only_if": "materially_new_information_beyond_depth_role_continuity"},
        {"family": "generic_injury_burden", "prior": "M67", "status": "DO_NOT_RETEST", "reopen_only_if": "exact_player_level_game_day_availability_or_practice_timeline"},
        {"family": "official_game_day_inactive_identity", "prior": "NEW_M78", "status": "SOURCE_AUDIT_ONLY", "reopen_only_if": "M78_qualifies_then_one_frozen_M79_predictive_test"},
    ])

    inactive.to_csv(out_dir / "m78_official_inactives.csv", index=False)
    coverage.to_csv(out_dir / "m78_canonical_team_week_coverage.csv", index=False)
    pd.DataFrame(article_meta).to_csv(out_dir / "m78_article_manifest.csv", index=False)
    pd.DataFrame(snapshots).to_csv(out_dir / "m78_source_snapshots.csv", index=False)
    gates.to_csv(out_dir / "m78_source_gate.csv", index=False)
    interpretation.to_csv(out_dir / "m78_interpretation.csv", index=False)
    no_retest.to_csv(out_dir / "m78_no_retest_ledger.csv", index=False)
    (out_dir / "m78_contract.json").write_text(json.dumps({
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_sha256": CANONICAL_SHA256,
        "sportsbook_used": False,
        "target_game_performance_used": False,
        "predictive_model_fit": False,
        "historical_source": "NFL.com official inactive-report archive",
        "live_2026_source": LIVE_INACTIVES_URL,
        "historical_article_timestamp_used_as_feature": False,
        "official_inactive_identity_is_fixed_pre_kickoff_fact": True,
        "m79_authorized": authorized,
    }, indent=2) + "\n")

    print("=== M78 INTERPRETATION ===")
    print(interpretation.to_string(index=False))
    print("\n=== M78 SOURCE GATE ===")
    print(gates.to_string(index=False))
    print("\n=== COVERAGE BY SEASON ===")
    print(coverage.groupby("season").inactive_team_week_found.agg(["count", "sum", "mean"]).to_string())
    if len(inactive):
        print("\n=== INACTIVE IDENTITY BRIDGE BY SEASON ===")
        print(inactive.groupby("season").roster_identity_match.agg(["count", "sum", "mean"]).to_string())
    print("\n=== LIVE ENDPOINT ===")
    print(live_ok, live_detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
