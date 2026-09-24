#!/usr/bin/env python3
"""V1B gold-sample retrieval feasibility for the frozen QB public-intent source family.

Operational only. No football outcomes, residuals, sportsbook fields, or predictive
modeling. Retrieval uses the already-frozen generic query families and compares
results to the manually reviewed V1 gold prefix only for automation quality.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup

TEAM = {
    "ARI": ("Arizona Cardinals", "azcardinals.com"),
    "ATL": ("Atlanta Falcons", "atlantafalcons.com"),
    "BAL": ("Baltimore Ravens", "baltimoreravens.com"),
    "BUF": ("Buffalo Bills", "buffalobills.com"),
    "CAR": ("Carolina Panthers", "panthers.com"),
    "CHI": ("Chicago Bears", "chicagobears.com"),
    "CIN": ("Cincinnati Bengals", "bengals.com"),
    "CLE": ("Cleveland Browns", "clevelandbrowns.com"),
    "DAL": ("Dallas Cowboys", "dallascowboys.com"),
    "DEN": ("Denver Broncos", "denverbroncos.com"),
    "DET": ("Detroit Lions", "detroitlions.com"),
    "GB": ("Green Bay Packers", "packers.com"),
    "HOU": ("Houston Texans", "houstontexans.com"),
    "IND": ("Indianapolis Colts", "colts.com"),
    "JAX": ("Jacksonville Jaguars", "jaguars.com"),
    "KC": ("Kansas City Chiefs", "chiefs.com"),
    "LV": ("Las Vegas Raiders", "raiders.com"),
    "LAC": ("Los Angeles Chargers", "chargers.com"),
    "LAR": ("Los Angeles Rams", "therams.com"),
    "MIA": ("Miami Dolphins", "miamidolphins.com"),
    "MIN": ("Minnesota Vikings", "vikings.com"),
    "NE": ("New England Patriots", "patriots.com"),
    "NO": ("New Orleans Saints", "neworleanssaints.com"),
    "NYG": ("New York Giants", "giants.com"),
    "NYJ": ("New York Jets", "newyorkjets.com"),
    "PHI": ("Philadelphia Eagles", "philadelphiaeagles.com"),
    "PIT": ("Pittsburgh Steelers", "steelers.com"),
    "SEA": ("Seattle Seahawks", "seahawks.com"),
    "SF": ("San Francisco 49ers", "49ers.com"),
    "TB": ("Tampa Bay Buccaneers", "buccaneers.com"),
    "TEN": ("Tennessee Titans", "tennesseetitans.com"),
    "WAS": ("Washington Commanders", "commanders.com"),
}

OFFICIAL_CONCEPTS = [
    "{team} {opp} coach press conference week {week} {season}",
    "{team} {opp} offensive coordinator press conference week {week} {season}",
    "{team} game preview {opp} week {week} {season}",
    "{team} run pass game plan {opp} {season}",
    "{team} offense approach {opp} {season}",
]
LOCAL_CONCEPTS = [
    "{team} {opp} coach said offense game plan week {week} {season}",
    "{team} {opp} run game passing game plan {season}",
    "{team} offensive coordinator {opp} plan {season}",
]
INTENT_PATTERNS = {
    "RUN_EMPHASIS": [
        r"establish (?:the )?run", r"run the ball", r"running game", r"more touches",
        r"lean on (?:the )?run", r"stick with (?:the )?run",
    ],
    "PASS_EMPHASIS": [
        r"passing game", r"throw the ball", r"let (?:it|the ball) fly",
        r"attack.*through the air", r"downfield",
    ],
    "EARLY_DOWN_AGGRESSION": [r"early down", r"first down", r"aggressive"],
    "TEMPO_CHANGE": [r"tempo", r"no[- ]huddle", r"up[- ]tempo"],
    "PROTECTION_DRIVEN_PLAN": [r"protection", r"protect .*quarterback", r"offensive line"],
    "DEFENSIVE_MATCHUP_PLAN": [r"matchup", r"their front", r"their coverage", r"man coverage", r"zone coverage"],
    "PERSONNEL_AVAILABILITY_PLAN": [r"without ", r"returning ", r"availability", r"injur", r"replacement"],
    "OTHER_EXPLICIT_OFFENSIVE_INTENT": [r"game plan", r"plan for", r"plan to", r"want to", r"need to", r"get .* involved"],
}

UA = "Mozilla/5.0 (compatible; NFLPublicIntentResearch/1.0; +https://github.com/dkaps6/imtiredofthis)"
TIMEOUT = 15


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    p = urlparse(u)
    host = p.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+", "/", p.path).rstrip("/")
    return f"https://{host}{path}"


def unwrap_ddg(href: str) -> str:
    if not href:
        return ""
    if "duckduckgo.com/l/?" in href:
        qs = parse_qs(urlparse(href).query)
        if qs.get("uddg"):
            return unquote(qs["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


def search_ddg(query: str, limit: int = 5) -> tuple[list[dict], str]:
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        status = f"http_{r.status_code}"
        r.raise_for_status()
    except Exception as e:
        return [], f"{type(e).__name__}:{e}"
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    seen = set()
    for a in soup.select("a.result__a"):
        href = unwrap_ddg(a.get("href", ""))
        nu = norm_url(href)
        if not nu or nu in seen:
            continue
        seen.add(nu)
        rows.append({"url": href, "norm_url": nu, "title": " ".join(a.stripped_strings)})
        if len(rows) >= limit:
            break
    return rows, status


def parse_iso(s: str) -> datetime | None:
    try:
        x = str(s).strip().replace("Z", "+00:00")
        d = datetime.fromisoformat(x)
        if d.tzinfo is None:
            return None
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def page_meta(url: str) -> dict:
    out = {
        "fetch_ok": False, "http_status": "", "final_url": "", "title": "",
        "publication_time": "", "timestamp_method": "", "text": "", "error": "",
    }
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)
        out["http_status"] = str(r.status_code)
        out["final_url"] = r.url
        r.raise_for_status()
        out["fetch_ok"] = True
        soup = BeautifulSoup(r.text, "html.parser")
        out["title"] = soup.title.get_text(" ", strip=True) if soup.title else ""
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            txt = script.string or script.get_text(" ", strip=True)
            for m in re.finditer(r'"datePublished"\s*:\s*"([^"]+)"', txt):
                if parse_iso(m.group(1)):
                    out["publication_time"] = m.group(1)
                    out["timestamp_method"] = "jsonld_datePublished"
                    break
            if out["publication_time"]:
                break
        if not out["publication_time"]:
            for prop in ["article:published_time", "og:published_time"]:
                tag = soup.find("meta", attrs={"property": prop})
                val = tag.get("content", "") if tag else ""
                if val and parse_iso(val):
                    out["publication_time"] = val
                    out["timestamp_method"] = prop
                    break
        if not out["publication_time"]:
            for t in soup.find_all("time"):
                val = t.get("datetime", "")
                if val and parse_iso(val):
                    out["publication_time"] = val
                    out["timestamp_method"] = "time_datetime"
                    break
        for bad in soup(["script", "style", "noscript", "svg"]):
            bad.decompose()
        out["text"] = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:120000]
    except Exception as e:
        out["error"] = f"{type(e).__name__}:{e}"
    return out


def intent_tags(text: str) -> list[str]:
    s = str(text or "").lower()
    tags = []
    for tag, pats in INTENT_PATTERNS.items():
        if any(re.search(p, s, flags=re.I) for p in pats):
            tags.append(tag)
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_csv(args.gold)
    candidate_rows = []
    row_results = []

    for row in gold:
        season, week = int(row["season"]), int(row["week"])
        team, opp = row["team"].strip().upper(), row["opponent"].strip().upper()
        team_name, official_domain = TEAM[team]
        opp_name = TEAM[opp][0]
        gold_url = norm_url(row.get("locator", ""))
        gold_source = row.get("source_class", "").strip().upper()

        queries = []
        for tmpl in OFFICIAL_CONCEPTS:
            q = tmpl.format(team=team_name, opp=opp_name, week=week, season=season)
            queries.append(("OFFICIAL", q + f" site:{official_domain}"))
        if gold_source != "OFFICIAL":
            for tmpl in LOCAL_CONCEPTS:
                queries.append(("LOCAL_FALLBACK", tmpl.format(team=team_name, opp=opp_name, week=week, season=season)))

        discovered = []
        search_errors = []
        for phase, q in queries:
            results, status = search_ddg(q, limit=5)
            if not results:
                search_errors.append(f"{phase}:{status}")
            for rank, item in enumerate(results, start=1):
                rec = {
                    "season": season, "week": week, "team": team, "opponent": opp,
                    "phase": phase, "query": q, "rank": rank,
                    "url": item["url"], "norm_url": item["norm_url"], "title": item["title"],
                }
                candidate_rows.append(rec)
                discovered.append(rec)
            time.sleep(args.sleep)

        dedup = {}
        for x in discovered:
            dedup.setdefault(x["norm_url"], x)
        discovered = list(dedup.values())
        canonical_found = gold_url in dedup

        official_candidates = [x for x in discovered if urlparse(x["url"]).netloc.lower().removeprefix("www.").endswith(official_domain)]
        fetch_pool = official_candidates[:5]
        if gold_source != "OFFICIAL":
            fetch_pool += [x for x in discovered if x["phase"] == "LOCAL_FALLBACK"][:5]

        page_rows = []
        for cand in fetch_pool:
            meta = page_meta(cand["url"])
            tags = intent_tags(meta["text"])
            pub = parse_iso(meta["publication_time"])
            ko = parse_iso(row["kickoff"])
            ts_safe = bool(pub and ko and pub < ko)
            opp_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", opp_name) if len(t) >= 4]
            text_low = (meta["title"] + " " + meta["text"][:40000]).lower()
            relevant = any(tok in text_low for tok in opp_tokens)
            page_rows.append({
                **cand,
                "fetch_ok": meta["fetch_ok"],
                "http_status": meta["http_status"],
                "final_url": meta["final_url"],
                "publication_time": meta["publication_time"],
                "timestamp_method": meta["timestamp_method"],
                "timestamp_safe": ts_safe,
                "target_relevant": relevant,
                "intent_tags": ";".join(tags),
                "intent_candidate": bool(tags and relevant and ts_safe),
                "error": meta["error"],
            })
        auto = [x for x in page_rows if x["intent_candidate"]]
        eligible_recalled = bool(auto) or canonical_found
        row_results.append({
            "season": season, "week": week, "team": team, "opponent": opp,
            "gold_source_class": gold_source, "gold_locator": row.get("locator", ""),
            "queries_attempted": len(queries), "unique_candidates": len(discovered),
            "canonical_found": canonical_found,
            "official_candidate_count": len(official_candidates),
            "auto_review_ready_count": len(auto),
            "eligible_candidate_recalled": eligible_recalled,
            "search_errors": "|".join(search_errors),
        })
        pd = __import__("pandas")
        pd.DataFrame(page_rows).to_csv(args.out_dir / f"pages_{season}_{week}_{team}.csv", index=False)

    import pandas as pd
    cand_df = pd.DataFrame(candidate_rows)
    rows_df = pd.DataFrame(row_results)
    cand_df.to_csv(args.out_dir / "candidate_discovery.csv", index=False)
    rows_df.to_csv(args.out_dir / "gold_row_results.csv", index=False)

    n = len(rows_df)
    canonical_rate = float(rows_df.canonical_found.mean()) if n else 0.0
    recall_rate = float(rows_df.eligible_candidate_recalled.mean()) if n else 0.0
    payload = {
        "study": "QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1B_GOLD_RETRIEVAL",
        "gold_rows": int(n),
        "eligible_candidate_recall": recall_rate,
        "canonical_source_retrieval": canonical_rate,
        "transport": "duckduckgo_html",
        "football_outcomes_read": 0,
        "model_residuals_read": 0,
        "sportsbook_fields_used": 0,
        "predictive_models_fit": 0,
        "production_changes": 0,
        "early_stop_recall_below_80pct": recall_rate < 0.80,
        "primary_recall_gate_ge_90pct": recall_rate >= 0.90,
        "primary_canonical_gate_ge_75pct": canonical_rate >= 0.75,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
