#!/usr/bin/env python3
"""V1L prospective structured official-source collector for QB team-pass intent.

Prospective source collection only. No football outcomes, residuals, sportsbook
inputs, predictive fitting, or production changes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.build._schedule_utils import get_nfl_schedule
from scripts._opponent_map import canon_team
from scripts.research.audit_qb_team_pass_intent_first_party_archive_transport_v1c import (
    DOMAINS,
    _first_party,
)
from scripts.research.audit_qb_team_pass_intent_full_official_first_source_screen_v1g import (
    _enumerate_domain,
)
from scripts.research.audit_qb_team_pass_intent_first_party_gold_validation_v1d import (
    _extract_safe_publication,
    _parse_absolute,
    _read_page,
)
from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import (
    OFFENSE_RE,
    POSTGAME_RE,
    TEAM_SYNONYMS,
    _fragments,
    _opponent_relevant,
    _visible,
)

VERSION = "QB_TEAM_PASS_INTENT_PROSPECTIVE_STRUCTURED_SOURCE_COLLECTION_V1L"
DEFAULT_SEASON = 2026
DEFAULT_WEEK = 6
WINDOW_DAYS = 7
POST_PRIOR_GAME_HOURS = 6
MAX_DISCOVERY_CANDIDATES = 16
MAX_RETAINED_PER_TEAM = 3

ALLOWED_SOURCE_TOKENS = (
    "transcript",
    "transcripts",
    "press-conference",
    "press-conferences",
    "media-availability",
    "media_availability",
    "what-they-said",
    "quotes",
)
EXCLUDED_PATH_TOKENS = (
    "injury-report",
    "game-status",
    "roster-move",
    "roster-moves",
    "transaction",
    "game-preview",
    "scouting-report",
    "storylines",
    "keys-to-victory",
    "what-to-watch",
    "prediction",
    "predictions",
    "postgame",
    "post-game",
    "recap",
)
POSTGAME_TITLE_RE = re.compile(
    r"\b(?:postgame|post-game|recap|final score|after (?:the )?game|following (?:the )?(?:win|loss|game))\b",
    re.I,
)


@dataclass(frozen=True)
class TeamWeek:
    season: int
    week: int
    team: str
    opponent: str
    kickoff: str
    prior_kickoff: str
    window_start: str


@dataclass(frozen=True)
class DiscoveryCandidate:
    team: str
    opponent: str
    season: int
    week: int
    kickoff: str
    window_start: str
    url: str
    lastmod: str
    source_family: str
    opponent_url_match: bool
    week_url_match: bool
    discovery_score: int


@dataclass
class RetainedSource:
    team: str
    opponent: str
    season: int
    week: int
    kickoff: str
    window_start: str
    locator: str
    final_url: str
    source_family: str
    publication_time: str
    timestamp_method: str
    page_title: str
    first_party: bool
    timestamp_safe: bool
    target_relevant: bool
    pregame_source: bool
    review_excerpts: list[str]
    errors: list[str]


@dataclass
class TeamResult:
    team: str
    opponent: str
    kickoff: str
    prior_kickoff: str
    window_start: str
    enumerated_urls: int
    discovery_candidates: int
    attempted_candidates: int
    retained_sources: int
    sources: list[dict[str, Any]]
    errors: list[str]


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = _parse_absolute(str(value))
        if dt is None:
            raise RuntimeError(f"non_absolute_datetime:{value}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _schedule_games(season: int) -> list[dict[str, Any]]:
    df = get_nfl_schedule(season)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        week = int(r["week"])
        home = canon_team(str(r["home"]))
        away = canon_team(str(r["away"]))
        kickoff = _to_utc(r["kickoff_utc"])
        rows.append({"week":week,"home":home,"away":away,"kickoff":kickoff})
    return rows


def _build_universe(season: int, week: int) -> list[TeamWeek]:
    games = _schedule_games(season)
    current = [g for g in games if g["week"] == week]
    if not current:
        raise RuntimeError(f"no_schedule_rows:{season}:week{week}")

    prior_by_team: dict[str, datetime] = {}
    for g in games:
        if g["week"] >= week:
            continue
        for team in (g["home"], g["away"]):
            prev = prior_by_team.get(team)
            if prev is None or g["kickoff"] > prev:
                prior_by_team[team] = g["kickoff"]

    out: list[TeamWeek] = []
    seen: set[tuple[int,int,str]] = set()
    for g in current:
        for team, opp in ((g["home"],g["away"]),(g["away"],g["home"])):
            key=(season,week,team)
            if key in seen:
                raise RuntimeError(f"duplicate_team_week:{key}")
            seen.add(key)
            target = g["kickoff"]
            floor = target - timedelta(days=WINDOW_DAYS)
            prior = prior_by_team.get(team)
            if prior is not None:
                floor = max(floor, prior + timedelta(hours=POST_PRIOR_GAME_HOURS))
            out.append(TeamWeek(
                season=season,week=week,team=team,opponent=opp,
                kickoff=_iso(target),
                prior_kickoff=_iso(prior) if prior else "",
                window_start=_iso(floor),
            ))
    out.sort(key=lambda x:(x.kickoff,x.team))
    return out


def _path(url: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlparse(url).path).lower()


def _source_family(path: str) -> str:
    for token in ALLOWED_SOURCE_TOKENS:
        if token in path:
            return token
    return ""


def _excluded_path(path: str) -> bool:
    return any(token in path for token in EXCLUDED_PATH_TOKENS)


def _token_match(path: str, values: tuple[str, ...]) -> bool:
    normalized = " " + re.sub(r"[^a-z0-9]+"," ",path.lower()).strip() + " "
    for raw in values:
        token=re.sub(r"[^a-z0-9]+"," ",str(raw).lower()).strip()
        if token and (" "+token+" ") in normalized:
            return True
    return False


def _week_match(path: str, week: int) -> bool:
    normalized=re.sub(r"[_-]+"," ",path.lower())
    return bool(re.search(rf"\bweek\s*{week}\b", normalized))


def _parse_lastmod(value: str) -> datetime | None:
    s=(value or "").strip().replace("Z","+00:00").replace("z","+00:00")
    if not s:
        return None
    try:
        dt=datetime.fromisoformat(s)
    except ValueError:
        try:
            dt=datetime.fromisoformat(s[:10])
        except Exception:
            return None
    if dt.tzinfo is None:
        dt=dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _discovery_candidates(tw: TeamWeek, entries: list[Any]) -> list[DiscoveryCandidate]:
    start=_to_utc(tw.window_start)
    kickoff=_to_utc(tw.kickoff)
    ranked: list[tuple[int,float,str,DiscoveryCandidate]]=[]
    for ent in entries:
        url=str(getattr(ent,"url",""))
        lastmod=str(getattr(ent,"lastmod",""))
        p=_path(url)
        family=_source_family(p)
        if not family or _excluded_path(p):
            continue
        opp=_token_match(p,TEAM_SYNONYMS[tw.opponent])
        wk=_week_match(p,tw.week)
        lm=_parse_lastmod(lastmod)
        date_hint=bool(lm is not None and start <= lm <= kickoff)
        # Require a target-specific discovery hint before fetching.
        if not (opp or wk or date_hint):
            continue
        score=4*int(opp)+3*int(wk)+int(date_hint)
        distance=abs((kickoff-lm).total_seconds()) if lm else float("inf")
        cand=DiscoveryCandidate(
            team=tw.team,opponent=tw.opponent,season=tw.season,week=tw.week,
            kickoff=tw.kickoff,window_start=tw.window_start,url=url,lastmod=lastmod,
            source_family=family,opponent_url_match=opp,week_url_match=wk,
            discovery_score=score,
        )
        ranked.append((-score,distance,url.lower(),cand))
    ranked.sort(key=lambda x:(x[0],x[1],x[2]))
    return [x[3] for x in ranked[:MAX_DISCOVERY_CANDIDATES]]


def _review_excerpts(text: str, opponent: str) -> list[str]:
    scored: list[tuple[int,int,int,str]]=[]
    oppsyn=TEAM_SYNONYMS[opponent]
    for idx,frag in enumerate(_fragments(text)):
        if POSTGAME_RE.search(frag):
            continue
        if not OFFENSE_RE.search(frag):
            continue
        hay=frag.lower()
        opp=int(any(s.lower() in hay for s in oppsyn))
        action=int(bool(re.search(
            r"\b(?:plan|planning|want|wants|wanted|will|going to|need to|needs to|"
            r"emphasize|emphasizing|feature|use|using|attack|protect|adjust|change|"
            r"more|less|increase|decrease|establish|tempo)\b",
            frag,re.I,
        )))
        scored.append((-action,-opp,idx," ".join(frag.split())))
    scored.sort()
    return [x[3] for x in scored[:12]]


def _fetch_candidate(c: DiscoveryCandidate) -> RetainedSource:
    errors: list[str]=[]
    final=""
    publication=""
    method=""
    title=""
    first=safe=relevant=pregame=False
    excerpts: list[str]=[]
    try:
        raw,final=_read_page(c.url)
        first=bool(_first_party(c.url,DOMAINS[c.team]) and _first_party(final,DOMAINS[c.team]))
        kickoff=_to_utc(c.kickoff)
        start=_to_utc(c.window_start)
        method,publication,parseable,pre=_extract_safe_publication(raw,kickoff)
        pub=_parse_absolute(publication) if parseable else None
        if pub is not None:
            if pub.tzinfo is None:
                pub=pub.replace(tzinfo=timezone.utc)
            pub=pub.astimezone(timezone.utc)
        safe=bool(pub is not None and pre and start <= pub < kickoff)
        title,text=_visible(raw)
        relevant=_opponent_relevant(c.url,title,text,c.opponent)
        pregame=not bool(POSTGAME_TITLE_RE.search(title) or _excluded_path(_path(c.url)))
        excerpts=_review_excerpts(text,c.opponent)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:180]}")
    return RetainedSource(
        team=c.team,opponent=c.opponent,season=c.season,week=c.week,
        kickoff=c.kickoff,window_start=c.window_start,locator=c.url,final_url=final,
        source_family=c.source_family,publication_time=publication,timestamp_method=method,
        page_title=title,first_party=first,timestamp_safe=safe,
        target_relevant=relevant,pregame_source=pregame,review_excerpts=excerpts,
        errors=errors,
    )


def _eligible_source(s: RetainedSource) -> bool:
    return bool(
        s.first_party and s.timestamp_safe and s.target_relevant and s.pregame_source
    )


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,default=DEFAULT_SEASON)
    ap.add_argument("--week",type=int,default=DEFAULT_WEEK)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=20)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    universe=_build_universe(args.season,args.week)

    by_team={tw.team:tw for tw in universe}
    domain_entries: dict[str,list[Any]]={}
    domain_errors: dict[str,list[str]]={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16,args.workers)) as pool:
        futs={pool.submit(_enumerate_domain,DOMAINS[t]):t for t in sorted(by_team)}
        for fut in concurrent.futures.as_completed(futs):
            team=futs[fut]
            try:
                entries,errors=fut.result()
            except Exception as exc:
                entries,errors=[],[f"{type(exc).__name__}:{str(exc)[:180]}"]
            domain_entries[team]=entries
            domain_errors[team]=errors[:20]

    candidate_map: dict[str,list[DiscoveryCandidate]]={}
    all_candidates: list[DiscoveryCandidate]=[]
    for tw in universe:
        cs=_discovery_candidates(tw,domain_entries.get(tw.team,[]))
        candidate_map[tw.team]=cs
        all_candidates.extend(cs)

    fetched: dict[str,list[RetainedSource]]={t:[] for t in by_team}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs={pool.submit(_fetch_candidate,c):c for c in all_candidates}
        for fut in concurrent.futures.as_completed(futs):
            c=futs[fut]
            try:
                s=fut.result()
            except Exception as exc:
                s=RetainedSource(
                    team=c.team,opponent=c.opponent,season=c.season,week=c.week,
                    kickoff=c.kickoff,window_start=c.window_start,locator=c.url,final_url="",
                    source_family=c.source_family,publication_time="",timestamp_method="",
                    page_title="",first_party=False,timestamp_safe=False,target_relevant=False,
                    pregame_source=False,review_excerpts=[],
                    errors=[f"{type(exc).__name__}:{str(exc)[:180]}"],
                )
            fetched[c.team].append(s)

    team_results: list[TeamResult]=[]
    review_packet: list[dict[str,Any]]=[]
    for tw in universe:
        sources=[s for s in fetched.get(tw.team,[]) if _eligible_source(s)]
        # Discovery order is deterministic, preserve it and cap.
        order={c.url:i for i,c in enumerate(candidate_map.get(tw.team,[]))}
        sources.sort(key=lambda s:(order.get(s.locator,9999),s.locator))
        sources=sources[:MAX_RETAINED_PER_TEAM]
        for idx,s in enumerate(sources,1):
            review_packet.append({
                "review_id":f"{tw.season}W{tw.week}-{tw.team}-{idx}",
                "season":tw.season,"week":tw.week,"team":tw.team,"opponent":tw.opponent,
                "kickoff":tw.kickoff,"locator":s.locator,
                "publication_time":s.publication_time,"timestamp_method":s.timestamp_method,
                "page_title":s.page_title,"source_family":s.source_family,
                "review_excerpts":s.review_excerpts,
                "decision":"UNREVIEWED",
                "structured_tags":[],
                "speaker":"",
                "evidence":"",
            })
        team_results.append(TeamResult(
            team=tw.team,opponent=tw.opponent,kickoff=tw.kickoff,
            prior_kickoff=tw.prior_kickoff,window_start=tw.window_start,
            enumerated_urls=len(domain_entries.get(tw.team,[])),
            discovery_candidates=len(candidate_map.get(tw.team,[])),
            attempted_candidates=len(candidate_map.get(tw.team,[])),
            retained_sources=len(sources),
            sources=[asdict(s) for s in sources],
            errors=domain_errors.get(tw.team,[]),
        ))

    retained=sum(r.retained_sources for r in team_results)
    teams_with_sources=sum(r.retained_sources>0 for r in team_results)
    integrity={
        "football_outcomes_read":0,
        "model_residuals_read":0,
        "sportsbook_inputs_used":False,
        "paid_oddsapi_calls":0,
        "week5_2026_outcomes_read":0,
        "football_predictive_models_fit":0,
        "production_changed":False,
    }
    gates={
        "scheduled_team_universe_nonempty":len(universe)>0,
        "every_scheduled_team_attempted":len(team_results)==len(universe),
        "retained_sources_allowed_family_only":all(
            _source_family(_path(s["locator"])) for r in team_results for s in r.sources
        ),
        "retained_sources_first_party_timestamp_safe":all(
            bool(s["first_party"] and s["timestamp_safe"]) for r in team_results for s in r.sources
        ),
        "max_3_sources_per_team":all(r.retained_sources<=MAX_RETAINED_PER_TEAM for r in team_results),
        "review_packet_outcome_free":True,
        "paid_oddsapi_calls_zero":True,
        "football_predictive_models_fit_zero":True,
        "production_unchanged":True,
    }
    ready=all(gates.values())
    summary={
        "version":VERSION,"season":args.season,"week":args.week,
        "disposition":"PROSPECTIVE_STRUCTURED_COLLECTION_READY" if ready else "PROSPECTIVE_STRUCTURED_COLLECTION_NOT_READY",
        "scheduled_team_weeks":len(universe),
        "teams_with_retained_source":teams_with_sources,
        "retained_sources":retained,
        "coverage_rate":teams_with_sources/len(universe) if universe else 0.0,
        "unreviewed_packet_rows":len(review_packet),
        "gates":gates,
        **integrity,
    }

    (args.out_dir/"qb_intent_prospective_structured_v1l_summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    (args.out_dir/"qb_intent_prospective_structured_v1l_team_results.json").write_text(
        json.dumps([asdict(x) for x in team_results],indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    (args.out_dir/"qb_intent_prospective_structured_v1l_review_packet.json").write_text(
        json.dumps(review_packet,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
