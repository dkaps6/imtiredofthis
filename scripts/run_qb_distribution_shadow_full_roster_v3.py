#!/usr/bin/env python3
"""Prepare football-only primary-QB identity for the real-slate QB-C2 shadow.

Roster membership and game-starter authority are deliberately separate.
Ourlads provides the sportsbook-independent roster/depth fallback. A versioned,
football-only starter-authority file may supersede that ordering when a newer
official team announcement names the starter for the exact season/week.

The QB distribution shadow also consumes the exact final target-entitlement trace
emitted by the same successful no-credit Full Slate replay. That trace contains
the finite team target probabilities after M38 and TE-R5P and is football-only.
Using the exact replay trace prevents the shadow from reconstructing QB Monte
Carlo under a different RNG path than production.

Sportsbook identity is never used to choose a quarterback or construct football
probabilities. Priced pass-yard rows are compared only after the football starter
and football distribution are established. Production pricing is not modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.utils.player_identity_v3 import player_name_key

FORBIDDEN_ENTITLEMENT_COLUMNS = {
    "line", "source_line", "over_odds", "under_odds", "book", "book_title",
    "vegas_line", "vegas_odds", "market_prob", "edge_pct", "edge_abs", "team_wp",
}


def _role_rank(value: object) -> int:
    text = str(value or "").upper().replace(" ", "")
    if text in {"QB1", "QB01"} or "START" in text or "FIRST" in text:
        return 1
    if text in {"QB2", "QB02"} or "SECOND" in text:
        return 2
    if text in {"QB3", "QB03"} or "THIRD" in text:
        return 3
    if text in {"QB4", "QB04"} or "FOURTH" in text:
        return 4
    return 99


def _key(value: object) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _load_authority(path: str, season: int, week: int) -> pd.DataFrame:
    p = Path(path)
    required = {
        "season", "week", "team", "starter", "authority_type",
        "authority_date", "source_url", "reason",
    }
    if not p.exists():
        return pd.DataFrame(columns=sorted(required))
    frame = pd.read_csv(p, dtype=str).fillna("")
    frame.columns = [str(c).lower() for c in frame.columns]
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"starter authority missing columns: {missing}")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["team"] = frame["team"].astype(str).str.upper().str.strip()
    frame["starter_key"] = frame["starter"].map(_key)
    frame = frame.loc[
        frame["season"].eq(int(season)) & frame["week"].eq(int(week))
    ].copy()
    if frame.duplicated("team").any():
        dup = frame.loc[frame.duplicated("team", keep=False), ["team", "starter"]].to_dict("records")
        raise RuntimeError(f"duplicate starter authority rows: {dup}")
    bad_type = frame.loc[~frame["authority_type"].isin({"official_team_announcement", "official_team_depth_chart"})]
    if not bad_type.empty:
        raise RuntimeError(f"unsupported starter authority type: {bad_type['authority_type'].tolist()}")
    if frame["starter_key"].eq("").any():
        raise RuntimeError("blank canonical starter in starter authority")
    return frame


def _attach_exact_entitlement(universe: pd.DataFrame, path: str) -> tuple[pd.DataFrame, dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        raise RuntimeError(f"exact Full Slate entitlement trace missing/empty: {p}")
    trace = pd.read_csv(p, low_memory=False)
    trace.columns = [str(c).strip().lower() for c in trace.columns]
    required = {"event_id", "team", "player_clean_key", "entitlement_tgt_share"}
    missing = sorted(required - set(trace.columns))
    if missing:
        raise RuntimeError(f"entitlement trace missing columns: {missing}")
    leaked = sorted(FORBIDDEN_ENTITLEMENT_COLUMNS & set(trace.columns))
    if leaked:
        raise RuntimeError(f"sportsbook fields leaked into entitlement trace: {leaked}")

    trace["team"] = trace["team"].astype(str).str.upper().str.strip()
    trace["player_clean_key"] = trace["player_clean_key"].astype(str).str.strip()
    trace["event_id"] = trace["event_id"].astype(str).str.strip()
    trace["entitlement_tgt_share"] = pd.to_numeric(trace["entitlement_tgt_share"], errors="coerce")
    if trace["entitlement_tgt_share"].isna().any() or not np.isfinite(trace["entitlement_tgt_share"].to_numpy(float)).all():
        raise RuntimeError("entitlement trace has non-finite target shares")
    if trace["entitlement_tgt_share"].lt(0).any():
        raise RuntimeError("entitlement trace has negative target shares")

    key_cols = ["event_id", "team", "player_clean_key"]
    if trace.duplicated(key_cols).any():
        sample = trace.loc[trace.duplicated(key_cols, keep=False), key_cols].head(20).to_dict("records")
        raise RuntimeError(f"duplicate exact entitlement identities: {sample}")

    u = universe.copy()
    u["event_id"] = u["event_id"].astype(str).str.strip()
    u["team"] = u["team"].astype(str).str.upper().str.strip()
    u["player_clean_key"] = u["player_clean_key"].astype(str).str.strip()
    if u.duplicated(key_cols).any():
        raise RuntimeError("football universe has duplicate identities before entitlement attach")
    ukeys = set(map(tuple, u[key_cols].itertuples(index=False, name=None)))
    tkeys = set(map(tuple, trace[key_cols].itertuples(index=False, name=None)))
    if ukeys != tkeys:
        raise RuntimeError(
            "exact entitlement identity set != football universe; "
            f"missing_from_trace={sorted(ukeys-tkeys)[:20]} extra_in_trace={sorted(tkeys-ukeys)[:20]}"
        )

    keep = key_cols + ["entitlement_tgt_share"]
    if "baseline_entitlement_tgt_share" in trace.columns:
        trace["baseline_entitlement_tgt_share"] = pd.to_numeric(trace["baseline_entitlement_tgt_share"], errors="coerce")
        if trace["baseline_entitlement_tgt_share"].isna().any():
            raise RuntimeError("entitlement trace has invalid baseline_entitlement_tgt_share")
        keep.append("baseline_entitlement_tgt_share")
    for optional in ("te_r5p_applied", "te_r5p_model_version", "entitlement_version"):
        if optional in trace.columns:
            keep.append(optional)

    u["_shadow_row_order"] = np.arange(len(u))
    drop_existing = [c for c in keep if c not in key_cols and c in u.columns]
    if drop_existing:
        u = u.drop(columns=drop_existing)
    u = u.merge(trace[keep], on=key_cols, how="left", validate="one_to_one")
    u = u.sort_values("_shadow_row_order", kind="stable").drop(columns="_shadow_row_order").reset_index(drop=True)

    team = u.groupby(["event_id", "team"], as_index=False).agg(
        modeled_sum=("entitlement_tgt_share", "sum"), players=("player_clean_key", "nunique")
    )
    max_sum = float(team["modeled_sum"].max())
    min_sum = float(team["modeled_sum"].min())
    if len(team) != 32 or team["team"].nunique() != 32:
        raise RuntimeError(f"exact entitlement must cover 32 team-games, got {len(team)}")
    if max_sum > 0.950000000001 or min_sum < 0.949999999:
        raise RuntimeError(f"exact entitlement team mass drift: min={min_sum} max={max_sum}")
    payload = {
        "source": "EXACT_FULL_SLATE_REPLAY_TARGET_ENTITLEMENT_TRACE",
        "football_players": int(len(u)),
        "team_games": int(len(team)),
        "modeled_sum_min": min_sum,
        "modeled_sum_max": max_sum,
        "sportsbook_inputs_used": 0,
    }
    return u, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--priced", required=True)
    ap.add_argument("--state-context", required=True)
    ap.add_argument("--entitlement-trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--starter-authority", default="config/qb_starter_authority_v1.csv")
    args = ap.parse_args()

    universe = pd.read_csv(args.universe, low_memory=False)
    priced = pd.read_csv(args.priced, low_memory=False)
    universe.columns = [str(c).lower() for c in universe.columns]
    priced.columns = [str(c).lower() for c in priced.columns]
    if "depth_role" not in universe.columns or "position" not in universe.columns:
        raise RuntimeError("football universe missing Ourlads depth_role/position")

    universe, entitlement_audit = _attach_exact_entitlement(universe, args.entitlement_trace)
    Path("data/qb_c2_shadow_entitlement_audit.json").write_text(
        json.dumps(entitlement_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    authority = _load_authority(args.starter_authority, args.season, args.week)
    authority_by_team = authority.set_index("team", drop=False) if not authority.empty else None

    universe["qb_projection_eligible"] = 0
    universe["qb_role_score"] = np.nan
    universe["qb_role_source"] = "not_qb"

    qb_mask = universe["position"].astype(str).str.upper().eq("QB")
    audit_rows = []
    for team, part in universe.loc[qb_mask].groupby("team", sort=True):
        team = str(team).upper().strip()
        ranked = part.copy()
        ranked["_depth_rank"] = ranked["depth_role"].map(_role_rank)
        ranked["_player_key"] = ranked["player"].map(_key)
        ranked = ranked.sort_values(["_depth_rank", "player_clean_key"], kind="mergesort")
        if ranked.empty or int(ranked.iloc[0]["_depth_rank"]) != 1:
            raise RuntimeError(f"team={team} has no football-only QB1 depth role")
        if int((ranked["_depth_rank"] == 1).sum()) != 1:
            sample = ranked.loc[ranked["_depth_rank"].eq(1), ["player", "depth_role"]].to_dict("records")
            raise RuntimeError(f"team={team} ambiguous Ourlads QB1: {sample}")

        ourlads_idx = ranked.index[0]
        primary_idx = ourlads_idx
        source_type = "ourlads_depth_role_fallback"
        source_date = ""
        source_url = ""
        source_reason = "No newer versioned official starter authority row for this season/week"

        if authority_by_team is not None and team in authority_by_team.index:
            ar = authority_by_team.loc[team]
            if isinstance(ar, pd.DataFrame):
                raise RuntimeError(f"duplicate authority rows team={team}")
            wanted_key = str(ar["starter_key"])
            matches = ranked.loc[ranked["_player_key"].eq(wanted_key)]
            if len(matches) != 1:
                sample = ranked[["player", "depth_role"]].to_dict("records")
                raise RuntimeError(
                    f"official starter authority player not uniquely present in football roster "
                    f"team={team} starter={ar['starter']} roster={sample}"
                )
            primary_idx = matches.index[0]
            source_type = str(ar["authority_type"])
            source_date = str(ar["authority_date"])
            source_url = str(ar["source_url"])
            source_reason = str(ar["reason"])

        universe.loc[ranked.index, "qb_role_score"] = -ranked["_depth_rank"].astype(float).to_numpy()
        universe.loc[ranked.index, "qb_role_source"] = "ourlads_depth_role"
        universe.at[primary_idx, "qb_role_score"] = 0.0
        universe.at[primary_idx, "qb_role_source"] = source_type
        universe.at[primary_idx, "qb_projection_eligible"] = 1

        audit_rows.append({
            "team": team,
            "primary_player": str(universe.at[primary_idx, "player"]),
            "primary_player_key": _key(universe.at[primary_idx, "player"]),
            "depth_role": str(universe.at[primary_idx, "depth_role"]),
            "ourlads_qb1_player": str(universe.at[ourlads_idx, "player"]),
            "authority_source": source_type,
            "authority_date": source_date,
            "authority_url": source_url,
            "authority_reason": source_reason,
            "authority_overrode_ourlads": int(primary_idx != ourlads_idx),
            "sportsbook_inputs_used": 0,
        })

    audit = pd.DataFrame(audit_rows)
    if len(audit) != 32 or audit["team"].nunique() != 32:
        raise RuntimeError(f"football-only QB starter authority must cover 32 teams, got {len(audit)}")
    if not audit["sportsbook_inputs_used"].eq(0).all():
        raise RuntimeError("football-only QB starter authority leakage flag")

    pass_rows = priced.loc[priced["market"].astype(str).str.lower().eq("pass_yards")].copy()
    pass_rows["priced_player_key"] = pass_rows["player"].map(_key)
    pass_rows = pass_rows.sort_values(["team", "player", "side"], kind="mergesort").drop_duplicates(
        ["team", "priced_player_key"], keep="first"
    )
    offered = pass_rows.groupby("team", sort=True).agg(
        priced_qb_count=("priced_player_key", "nunique"),
        priced_primary_player=("player", "first"),
        priced_primary_player_key=("priced_player_key", "first"),
    ).reset_index()
    audit = audit.merge(offered, on="team", how="left", validate="one_to_one")
    audit["priced_qb_count"] = pd.to_numeric(audit["priced_qb_count"], errors="coerce").fillna(0).astype(int)
    audit["downstream_identity_match"] = (
        audit["priced_qb_count"].eq(1)
        & audit["priced_primary_player_key"].fillna("").eq(audit["primary_player_key"])
    )
    audit.to_csv("data/qb_c2_shadow_primary_qb_audit.csv", index=False)

    mismatches = audit.loc[~audit["downstream_identity_match"], [
        "team", "primary_player", "authority_source", "priced_primary_player", "priced_qb_count"
    ]].to_dict("records")
    if mismatches:
        result = {
            "disposition": "QB_DISTRIBUTION_FULL_ROSTER_SHADOW_BLOCKED_UNRESOLVED_STARTER_IDENTITY",
            "football_only_qb1_teams": 32,
            "official_authority_overrides": int(audit["authority_overrode_ourlads"].sum()),
            "downstream_identity_mismatch_teams": int(len(mismatches)),
            "mismatches": mismatches,
            "sportsbook_inputs_to_qb_selection": 0,
            "production_pricing_modified": 0,
            "scientific_candidate_rejected": False,
            "reason": "downstream paid pass-yard identity does not match football-only current starter authority",
        }
        Path(args.result).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(3)

    tmp = Path("data/qb_c2_shadow_universe_with_primary_identity.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(tmp, index=False)

    cmd = [
        sys.executable,
        "scripts/run_qb_distribution_shadow_full_roster_v2.py",
        "--universe", str(tmp),
        "--priced", args.priced,
        "--state-context", args.state_context,
        "--out", args.out,
        "--result", args.result,
    ]
    print(
        "[qb_c2_primary_identity] football_only_qb1_teams=32 "
        f"official_authority_overrides={int(audit['authority_overrode_ourlads'].sum())} "
        "exact_production_entitlement=1 sportsbook_inputs_used=0 downstream_identity_match=32"
    )
    return int(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
