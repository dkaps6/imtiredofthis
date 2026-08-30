#!/usr/bin/env python3
"""M84: audit sources for the Top Weapon Escape Hatch hypothesis.

Source audit only. No QB/receiver target-game outcomes are loaded or scored.
The decisive requirement is an explicit receiver-to-defender/responsibility
bridge with multi-season history AND an in-season path. On-field co-presence or
nearest-defender heuristics are not treated as exact assignments.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SEASONS = (2024, 2025)
ASSIGNMENT_CANDIDATES = [
    "coverage_defender_id", "coverage_defender_name",
    "defender_id", "defender_name",
    "primary_defender_id", "primary_defender_name",
    "target_defender_id", "target_defender_name",
    "nearest_defender_id", "nearest_defender_name",
    "coverage_responsibility_id", "coverage_responsibility_name",
    "primarydefensivecoveragematchupnflid",
    "secondarydefensivecoveragematchupnflid",
    "pff_primarydefensivecoveragematchupnflid",
    "pff_secondarydefensivecoveragematchupnflid",
]
ROUTE_ALIGNMENT_TOKENS = ("route", "align", "split", "slot", "wide")
TARGET_TOKENS = ("receiver", "target")
DEFENDER_TOKENS = ("defender", "coverage", "corner", "cb")

DOC_EVIDENCE = {
    "BDB2025": "https://www.kaggle.com/c/nfl-big-data-bowl-2025/data",
    "BDB2026": "https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data",
    "NGS_COVERAGE_RESPONSIBILITY": "https://www.nfl.com/news/next-gen-stats-new-advanced-metrics-you-need-to-know-for-the-2025-nfl-season",
    "NFLVERSE_PARTICIPATION_DICT": "https://nflreadr.nflverse.com/articles/dictionary_participation.html",
    "NFLVERSE_FTN_DICT": "https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html",
    "NFLVERSE_UPDATE_SCHEDULE": "https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html",
    "PFF_MATCHUPS": "https://www.pff.com/tools/matchups",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "m84-top-weapon-source-audit"})
    with urlopen(req, timeout=180) as r:
        raw = r.read()
        final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final, "bytes": len(raw), "sha256": sha256(raw)
    }


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c).strip().lower() for c in x.columns]; return x


def nonempty_coverage(s: pd.Series) -> float:
    z = s.astype("string").str.strip()
    return float((z.notna() & z.ne("") & z.str.lower().ne("nan") & z.str.lower().ne("none")).mean()) if len(z) else 0.0


def find_columns(cols: list[str], tokens: tuple[str, ...]) -> list[str]:
    return sorted([c for c in cols if any(t in c.lower() for t in tokens)])


def assignment_inventory(df: pd.DataFrame) -> list[dict]:
    x = lower(df); rows = []
    normalized = {c.replace("_", "").lower(): c for c in x.columns}
    for candidate in ASSIGNMENT_CANDIDATES:
        key = candidate.replace("_", "").lower()
        if key in normalized:
            col = normalized[key]
            rows.append({"column": col, "coverage": nonempty_coverage(x[col])})
    return rows


def audit_nflverse(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows, inventory_rows = [], []
    for season in SEASONS:
        part_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
        try:
            p, meta = read_parquet(part_url); p = lower(p)
            assignments = assignment_inventory(p)
            route_cols = find_columns(list(p.columns), ROUTE_ALIGNMENT_TOKENS)
            target_cols = find_columns(list(p.columns), TARGET_TOKENS)
            defender_cols = find_columns(list(p.columns), DEFENDER_TOKENS)
            for kind, cols in [("route_alignment", route_cols), ("target", target_cols), ("defender_coverage", defender_cols)]:
                for c in cols:
                    inventory_rows.append({
                        "source": "nflverse_participation", "season": season,
                        "kind": kind, "column": c, "coverage": nonempty_coverage(p[c])
                    })
            for a in assignments:
                inventory_rows.append({"source": "nflverse_participation", "season": season, "kind": "explicit_assignment", **a})
            source_rows.append({
                "source": "nflverse_participation", "season": season,
                "status": "OK", "rows": len(p), "schema_columns": len(p.columns),
                "route_or_alignment_present": bool(route_cols),
                "explicit_assignment_present": bool(assignments),
                "assignment_columns": ";".join(a["column"] for a in assignments),
                "on_field_defenders_present": bool({"defense_players", "defense_names"} & set(p.columns)),
                "coverage_shell_present": "defense_coverage_type" in p.columns,
                "in_season_public_2026": False,
                "public_free": True,
                **meta,
            })
        except Exception as exc:
            print(f"[m84_source_error] participation {season}: {type(exc).__name__}: {exc}")
            source_rows.append({
                "source": "nflverse_participation", "season": season,
                "status": f"ERROR:{type(exc).__name__}:{exc}", "rows": 0,
                "schema_columns": 0, "route_or_alignment_present": False,
                "explicit_assignment_present": False, "assignment_columns": "",
                "on_field_defenders_present": False, "coverage_shell_present": False,
                "in_season_public_2026": False, "public_free": True,
                "url": part_url, "bytes": 0, "sha256": "",
            })

        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        try:
            f, meta = read_parquet(ftn_url); f = lower(f)
            assignments = assignment_inventory(f)
            route_cols = find_columns(list(f.columns), ROUTE_ALIGNMENT_TOKENS)
            target_cols = find_columns(list(f.columns), TARGET_TOKENS)
            defender_cols = find_columns(list(f.columns), DEFENDER_TOKENS)
            for kind, cols in [("route_alignment", route_cols), ("target", target_cols), ("defender_coverage", defender_cols)]:
                for c in cols:
                    inventory_rows.append({
                        "source": "nflverse_ftn_charting", "season": season,
                        "kind": kind, "column": c, "coverage": nonempty_coverage(f[c])
                    })
            for a in assignments:
                inventory_rows.append({"source": "nflverse_ftn_charting", "season": season, "kind": "explicit_assignment", **a})
            source_rows.append({
                "source": "nflverse_ftn_charting", "season": season,
                "status": "OK", "rows": len(f), "schema_columns": len(f.columns),
                "route_or_alignment_present": bool(route_cols),
                "explicit_assignment_present": bool(assignments),
                "assignment_columns": ";".join(a["column"] for a in assignments),
                "on_field_defenders_present": False,
                "coverage_shell_present": False,
                "in_season_public_2026": True,
                "public_free": True,
                **meta,
            })
        except Exception as exc:
            print(f"[m84_source_error] FTN {season}: {type(exc).__name__}: {exc}")
            source_rows.append({
                "source": "nflverse_ftn_charting", "season": season,
                "status": f"ERROR:{type(exc).__name__}:{exc}", "rows": 0,
                "schema_columns": 0, "route_or_alignment_present": False,
                "explicit_assignment_present": False, "assignment_columns": "",
                "on_field_defenders_present": False, "coverage_shell_present": False,
                "in_season_public_2026": True, "public_free": True,
                "url": ftn_url, "bytes": 0, "sha256": "",
            })

    src = pd.DataFrame(source_rows); inv = pd.DataFrame(inventory_rows)
    src.to_csv(out / "m84_nflverse_source_snapshot.csv", index=False)
    inv.to_csv(out / "m84_nflverse_matchup_column_inventory.csv", index=False)
    return src, inv


def documented_source_matrix(nflverse: pd.DataFrame) -> pd.DataFrame:
    rows = []

    part_ok = (
        not nflverse.empty
        and len(nflverse.loc[nflverse.source.eq("nflverse_participation") & nflverse.status.eq("OK")]) == len(SEASONS)
    )
    part_assignment = bool(
        part_ok and nflverse.loc[nflverse.source.eq("nflverse_participation"), "explicit_assignment_present"].fillna(False).any()
    )
    part_route = bool(
        part_ok and nflverse.loc[nflverse.source.eq("nflverse_participation"), "route_or_alignment_present"].fillna(False).all()
    )
    rows.append({
        "source_path": "nflverse_participation",
        "evidence": DOC_EVIDENCE["NFLVERSE_PARTICIPATION_DICT"],
        "weapon_identity": True,
        "route_alignment": part_route,
        "explicit_responsibility": part_assignment,
        "defender_quality": False,
        "replacement_context": False,
        "historical_2024_2025": part_ok,
        "in_season_2026": False,
        "public_free": True,
        "disposition": "NO_EXPLICIT_RESPONSIBILITY" if part_ok and not part_assignment else "SOURCE_ERROR",
        "note": "route/shell/on-field personnel are useful, but every DB on field is not a coverage assignment; 2023+ participation is postseason-only",
    })

    ftn_ok = (
        not nflverse.empty
        and len(nflverse.loc[nflverse.source.eq("nflverse_ftn_charting") & nflverse.status.eq("OK")]) == len(SEASONS)
    )
    ftn_assignment = bool(
        ftn_ok and nflverse.loc[nflverse.source.eq("nflverse_ftn_charting"), "explicit_assignment_present"].fillna(False).any()
    )
    ftn_route = bool(
        ftn_ok and nflverse.loc[nflverse.source.eq("nflverse_ftn_charting"), "route_or_alignment_present"].fillna(False).any()
    )
    rows.append({
        "source_path": "nflverse_ftn_charting",
        "evidence": DOC_EVIDENCE["NFLVERSE_FTN_DICT"],
        "weapon_identity": False,
        "route_alignment": ftn_route,
        "explicit_responsibility": ftn_assignment,
        "defender_quality": False,
        "replacement_context": False,
        "historical_2024_2025": ftn_ok,
        "in_season_2026": True,
        "public_free": True,
        "disposition": "NO_EXPLICIT_RESPONSIBILITY" if ftn_ok and not ftn_assignment else "SOURCE_ERROR",
        "note": "updates in-season but its public charting subset does not provide the route/responsible-defender bridge",
    })

    rows.append({
        "source_path": "NFL_Big_Data_Bowl_2025",
        "evidence": DOC_EVIDENCE["BDB2025"],
        "weapon_identity": True,
        "route_alignment": True,
        "explicit_responsibility": True,
        "defender_quality": True,
        "replacement_context": False,
        "historical_2024_2025": False,
        "in_season_2026": False,
        "public_free": True,
        "disposition": "EXACT_BUT_LIMITED_COMPETITION_SAMPLE",
        "note": "routeRan + PFF defensive assignment + primary/secondary coverage matchup IDs; first nine weeks of 2022 only",
    })

    rows.append({
        "source_path": "NFL_Big_Data_Bowl_2026",
        "evidence": DOC_EVIDENCE["BDB2026"],
        "weapon_identity": True,
        "route_alignment": True,
        "explicit_responsibility": False,
        "defender_quality": True,
        "replacement_context": False,
        "historical_2024_2025": False,
        "in_season_2026": False,
        "public_free": True,
        "disposition": "HISTORICAL_RESEARCH_ONLY",
        "note": "rich 2023/2024 tracking and targeted-receiver route/coverage context; does not create a live 2026 responsibility feed and nearest defender is not exact responsibility",
    })

    rows.append({
        "source_path": "NFL_NGS_Coverage_Responsibility",
        "evidence": DOC_EVIDENCE["NGS_COVERAGE_RESPONSIBILITY"],
        "weapon_identity": True,
        "route_alignment": True,
        "explicit_responsibility": True,
        "defender_quality": True,
        "replacement_context": True,
        "historical_2024_2025": False,
        "in_season_2026": True,
        "public_free": False,
        "disposition": "PROPRIETARY_OR_NO_PUBLIC_BULK_CONTRACT",
        "note": "NFL/AWS production model identifies matchups/assignments in real time; no public/free reproducible historical bulk + live feed established",
    })

    rows.append({
        "source_path": "PFR_advanced_defense_plus_depth_chart",
        "evidence": DOC_EVIDENCE["NFLVERSE_UPDATE_SCHEDULE"],
        "weapon_identity": False,
        "route_alignment": False,
        "explicit_responsibility": False,
        "defender_quality": True,
        "replacement_context": True,
        "historical_2024_2025": True,
        "in_season_2026": True,
        "public_free": True,
        "disposition": "LIVE_AUXILIARY_ONLY",
        "note": "useful defender quality/replacement context, but no explicit receiver-to-defender exposure bridge; M75/M79 proxies already rejected",
    })

    rows.append({
        "source_path": "public_composite_participation_PFR_depth",
        "evidence": ";".join([DOC_EVIDENCE["NFLVERSE_PARTICIPATION_DICT"], DOC_EVIDENCE["NFLVERSE_UPDATE_SCHEDULE"]]),
        "weapon_identity": True,
        "route_alignment": part_route,
        "explicit_responsibility": False,
        "defender_quality": True,
        "replacement_context": True,
        "historical_2024_2025": part_ok,
        "in_season_2026": False,
        "public_free": True,
        "disposition": "NO_EXPLICIT_RESPONSIBILITY",
        "note": "joinable auxiliary pieces still lack true coverage responsibility and participation does not update in season",
    })
    return pd.DataFrame(rows)


def decide(matrix: pd.DataFrame, nflverse: pd.DataFrame) -> dict:
    req = [
        "weapon_identity", "route_alignment", "explicit_responsibility",
        "defender_quality", "replacement_context", "historical_2024_2025",
        "in_season_2026", "public_free",
    ]
    qualifying = []
    for _, r in matrix.iterrows():
        if all(bool(r.get(c, False)) for c in req):
            qualifying.append(str(r.source_path))

    errors = []
    if not nflverse.empty:
        errors = nflverse.loc[~nflverse.status.astype(str).eq("OK"), ["source", "season", "status"]].to_dict(orient="records")
    exact_limited = matrix.loc[matrix.disposition.eq("EXACT_BUT_LIMITED_COMPETITION_SAMPLE"), "source_path"].astype(str).tolist()
    hist_research = matrix.loc[matrix.disposition.eq("HISTORICAL_RESEARCH_ONLY"), "source_path"].astype(str).tolist()
    ideal_unavailable = matrix.loc[matrix.disposition.eq("PROPRIETARY_OR_NO_PUBLIC_BULK_CONTRACT"), "source_path"].astype(str).tolist()

    if qualifying:
        status = "QUALIFIED_HISTORICAL_AND_LIVE_EXACT"
    elif errors and len(errors) >= len(nflverse):
        status = "SOURCE_ERROR"
    else:
        status = "SOURCE_BLOCKED_EXACT_TOP_WEAPON_MATCHUP"
    return {
        "migration": "M84",
        "status": status,
        "production_actionable": False,
        "qb_outcomes_read": False,
        "receiver_target_game_outcomes_read": False,
        "sportsbook_features_used": False,
        "authoritative_full_stack_mae": 56.749517,
        "qualifying_source_paths": qualifying,
        "exact_but_limited_sources": exact_limited,
        "historical_research_only_sources": hist_research,
        "ideal_exact_but_no_public_bulk_contract": ideal_unavailable,
        "nflverse_source_errors": errors,
        "advance_to_top_weapon_predictive_test": bool(qualifying),
        "same_proxy_retest_allowed": False,
        "next_boundary": (
            "freeze exact source construction before 2024 development; keep later confirmation untouched"
            if qualifying else
            "do not reopen M72/M75 proxies; exact receiver-defender responsibility remains source blocked"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/backtests/m84_top_weapon_source_audit"))
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)

    nflverse, inventory = audit_nflverse(out)
    matrix = documented_source_matrix(nflverse)
    matrix.to_csv(out / "m84_source_qualification_matrix.csv", index=False)
    pd.DataFrame([{"source": k, "url": v} for k, v in DOC_EVIDENCE.items()]).to_csv(out / "m84_documentation_evidence.csv", index=False)
    decision = decide(matrix, nflverse)
    (out / "m84_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([decision]).to_csv(out / "m84_decision.csv", index=False)

    print("[m84_source_matrix]")
    print(matrix.to_string(index=False))
    print("[m84_decision]")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
