"""Unanchored provider-ID collision census for the FantasyAlarm WR-CB archive.

Assigned on issue #535. The existing audit computes provider-ID collisions only
over ANCHORED rows (schedule-consistent, both identities already resolved), but
the bridge is applied to any row carrying a provider ID. So the published 7 WR /
2 CB collision counts are a lower bound measured on a narrower population than
the one the bridge touches. This census measures the full population.

Read-only re-analysis of an artifact that already exists. No acquisition, no
fetch, no model candidate, no parameters fit.

A provider ID mapping to several spellings is usually a stable alias, not a
collision -- so the census separates:
  ALIAS         normalized names agree (suffix/punctuation/casing only)
  ALIAS_LIKELY  same surname and same first initial
  MULTI_PERSON  neither -- a genuine candidate for one ID covering two people
and flags separately whether an ID spans multiple teams within one season-week,
which no alias explanation covers.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

BRIDGE = "FANTASYALARM_STABLE_ID_BRIDGE"
# clean_key arrives with separators already stripped ("michaelpittmanjr"), so a
# \b-anchored suffix pattern never fires. Peel known suffixes off the raw tail.
SUFFIXES = ("iii", "iv", "ii", "jr", "sr", "v")
TEAM_ALIAS = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV",
              "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "CLV": "CLE",
              "BLT": "BAL", "HST": "HOU"}
SIDES = [
    ("wr", "wr_source_player_id", "wr_clean_key", "wr_gsis_id", "wr_identity_method", "wr_team"),
    ("cb", "cb_source_player_id", "cb_clean_key", "cb_gsis_id", "cb_identity_method", "opponent"),
]


def _norm(v: str) -> str:
    s = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", "")
    s = re.sub(r"[^a-z]", "", s)
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) - len(suf) >= 6:
            return s[: -len(suf)]
    return s


def _team(v: str) -> str:
    t = str(v or "").strip().upper()
    return TEAM_ALIAS.get(t, t)


def _classify(keys: list[str]) -> str:
    norms = sorted({_norm(k) for k in keys if _norm(k)})
    if len(norms) <= 1:
        return "ALIAS"
    if all(a == norms[0] or a.startswith(norms[0]) or norms[0].startswith(a)
           for a in norms):
        return "ALIAS_LIKELY"
    return "MULTI_PERSON"


def _s(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[col].astype("string").fillna("").str.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    x = pd.read_csv(args.rows, low_memory=False)
    sched = (x["schedule_match"].astype(str).str.lower().eq("true")
             if "schedule_match" in x.columns else pd.Series(False, index=x.index))
    report: dict = {
        "contract": "WR_CB_PROVIDER_ID_COLLISION_CENSUS_V1",
        "source_rows": int(len(x)),
        "reacquired_archive": False,
        "model_candidates_scored": 0,
        "parameters_fit": 0,
        "sides": {},
    }

    for side, src_col, key_col, gsis_col, method_col, team_col in SIDES:
        src = _s(x, src_col)
        key = _s(x, key_col)
        gsis = _s(x, gsis_col)
        team = _s(x, team_col)
        season = x["season"].astype(str) if "season" in x.columns else pd.Series("", index=x.index)
        week = x["week"].astype(str) if "week" in x.columns else pd.Series("", index=x.index)
        # Pre-bridge identity only: a bridged row's GSIS was assigned BY the
        # bridge, so counting it would make the census confirm itself.
        pre_gsis = gsis.mask(_s(x, method_col).eq(BRIDGE), "")

        has_id = src.str.len().gt(0)
        anchored = sched & has_id & pre_gsis.str.len().gt(0)
        anchored_ids = set(src.loc[anchored])

        by_key: dict[str, set] = defaultdict(set)
        by_gsis: dict[str, set] = defaultdict(set)
        by_team_week: dict[str, set] = defaultdict(set)
        for sid, k, g, t, se, wk in zip(
            src.loc[has_id], key.loc[has_id], pre_gsis.loc[has_id],
            team.loc[has_id], season.loc[has_id], week.loc[has_id]
        ):
            if k:
                by_key[sid].add(k)
            if g:
                by_gsis[sid].add(g)
            if t:
                by_team_week[sid].add((se, wk, _team(t)))

        buckets: dict[str, list] = defaultdict(list)
        for sid, keys in by_key.items():
            if len(keys) > 1:
                buckets[_classify(sorted(keys))].append(sid)

        gsis_collisions = sorted(k for k, v in by_gsis.items() if len(v) > 1)
        unanchored_only_gsis = [k for k in gsis_collisions if k not in anchored_ids]

        # One provider ID appearing for two different teams in the SAME week is
        # not explainable as an alias.
        same_week_multi_team = []
        for sid, tws in by_team_week.items():
            seen: dict = defaultdict(set)
            for se, wk, t in tws:
                seen[(se, wk)].add(t)
            if any(len(v) > 1 for v in seen.values()):
                same_week_multi_team.append(sid)

        multi = sorted(buckets.get("MULTI_PERSON", []))
        report["sides"][side] = {
            "rows_with_provider_id": int(has_id.sum()),
            "distinct_provider_ids": len(by_key),
            "anchored_provider_ids": len(anchored_ids),
            "provider_ids_never_anchored": len(set(by_key) - anchored_ids),
            "ids_with_multiple_name_keys": sum(len(v) for v in buckets.values()),
            "classified_ALIAS": len(buckets.get("ALIAS", [])),
            "classified_ALIAS_LIKELY": len(buckets.get("ALIAS_LIKELY", [])),
            "classified_MULTI_PERSON": len(multi),
            "multi_person_examples": [
                {"provider_id": sid, "name_keys": sorted(by_key[sid])} for sid in multi
            ],
            "gsis_collisions_full_population": len(gsis_collisions),
            "gsis_collisions_invisible_to_anchored_audit": len(unanchored_only_gsis),
            "same_season_week_multi_team_ids": len(same_week_multi_team),
            "same_season_week_multi_team_examples": [
                {"provider_id": sid, "name_keys": sorted(by_key.get(sid, []))}
                for sid in sorted(same_week_multi_team)[:8]
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("<<<CENSUS")
    for side, d in report["sides"].items():
        print(f"[{side}] ids={d['distinct_provider_ids']} anchored={d['anchored_provider_ids']} "
              f"never_anchored={d['provider_ids_never_anchored']} "
              f"multi_name={d['ids_with_multiple_name_keys']} "
              f"ALIAS={d['classified_ALIAS']} ALIAS_LIKELY={d['classified_ALIAS_LIKELY']} "
              f"MULTI_PERSON={d['classified_MULTI_PERSON']} "
              f"gsis_coll_full={d['gsis_collisions_full_population']} "
              f"gsis_coll_hidden={d['gsis_collisions_invisible_to_anchored_audit']} "
              f"same_week_multi_team={d['same_season_week_multi_team_ids']}")
        for e in d["multi_person_examples"]:
            print(f"    MULTI_PERSON {e['provider_id']}: {', '.join(e['name_keys'])}")
        for e in d["same_season_week_multi_team_examples"]:
            print(f"    MULTI_TEAM   {e['provider_id']}: {', '.join(e['name_keys'])}")
    print("CENSUS>>>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
