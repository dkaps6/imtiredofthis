#!/usr/bin/env python3
"""Player Role Elasticity from Teammate Availability V1 — source audit.

Source-only / anti-retest audit. It uses the frozen M78 official-inactive
team-week snapshot plus strictly-prior player usage/snap history. It does not
attach target-game outcomes and fits no predictive model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_40s_information_frontier as m76
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import test_qb_exact_personnel_discontinuity as m77
from scripts import player_form_v2 as pf

VERSION = "PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_AUDIT"
M78_SHA = "d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609"
SEASONS = (2024, 2025)
HISTORY_SEASONS = (2023, 2024, 2025)

FAMILY_POS = {
    "RB": {"RB", "FB", "HB"},
    "WR": {"WR", "LWR", "RWR", "SWR"},
    "TE": {"TE"},
}
EVENT_MIN = {"RB": 20, "WR": 40, "TE": 15}
SUCCESSOR_MIN = {"RB": 30, "WR": 50, "TE": 20}
TWO_STATE_MIN = {"RB": 20, "WR": 20, "TE": 20}
READY = {
    "RB": "RB_ROLE_ELASTICITY_SOURCE_READY",
    "WR": "WR_ROLE_ELASTICITY_SOURCE_READY",
    "TE": "TE_ROLE_ELASTICITY_SOURCE_READY",
}
BLOCKED = {
    "RB": "RB_ROLE_ELASTICITY_SOURCE_BLOCKED",
    "WR": "WR_ROLE_ELASTICITY_SOURCE_BLOCKED",
    "TE": "TE_ROLE_ELASTICITY_SOURCE_BLOCKED",
}
ACTIVE_ROSTER_STATUS = {"ACT", "INA"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def family(pos: object) -> str:
    p = str(pos or "").upper().strip().replace("-", "").replace(" ", "")
    for fam, vals in FAMILY_POS.items():
        if p in vals:
            return fam
    return ""


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def parse_inactives(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sha256(path) != M78_SHA:
        raise RuntimeError("frozen M78 inactive snapshot SHA drift")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).lower() for c in x.columns]
    need = {"season", "week", "team", "inactive_tokens", "inactive_count"}
    if need - set(x.columns):
        raise RuntimeError(f"M78 snapshot missing {sorted(need-set(x.columns))}")
    if len(x) != 1088:
        raise RuntimeError(f"M78 team-week row drift {len(x)} != 1088")
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("M78 duplicate team-week")
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)

    rows = []
    for r in x.itertuples(index=False):
        toks = str(r.inactive_tokens or "").split("|")
        for tok in toks:
            if not tok.strip():
                continue
            parts = tok.rsplit(":", 2)
            if len(parts) != 3:
                continue
            name_key, pos, matched = parts
            rows.append({
                "season": int(r.season),
                "week": int(r.week),
                "team": str(r.team),
                "inactive_name_key": str(name_key),
                "listed_position": str(pos).upper(),
                "family": family(pos),
                "roster_identity_match": int(str(matched).strip() == "1"),
            })
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("M78 inactive token parse empty")
    return x, d


def normalize_roster(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=[
            "season", "week", "team", "player_id", "player_name_key",
            "position", "family", "status"
        ])
    x = raw.copy()
    x.columns = [str(c).lower() for c in x.columns]
    wc = first_col(x, ["week"])
    tc = first_col(x, ["team", "club_code"])
    ic = first_col(x, ["gsis_id", "player_id"])
    pc = first_col(x, ["position", "depth_chart_position", "ngs_position", "pos"])
    sc = first_col(x, ["status", "status_description_abbr"])
    name_cols = [c for c in ["full_name", "player_name", "display_name", "football_name"] if c in x.columns]
    if not all([wc, tc, pc]) or not name_cols:
        raise RuntimeError(f"weekly roster schema incomplete for {season}: {list(x.columns)}")
    if ic is None:
        x["_pid"] = ""
        ic = "_pid"
    if sc is None:
        x["_status"] = ""
        sc = "_status"
    out = pd.DataFrame({
        "season": int(season),
        "week": num(x[wc]),
        "team": x[tc].map(canon_team),
        "player_id": x[ic].astype("string").fillna("").str.strip(),
        "position": x[pc].astype("string").fillna("").str.upper().str.strip(),
        "status": x[sc].astype("string").fillna("").str.upper().str.strip(),
    })
    raw_name = x[name_cols[0]].astype("string").fillna("").str.strip()
    out["player_name_key"] = raw_name.map(m78.norm_name)
    out["family"] = out["position"].map(family)
    out = out.loc[out["week"].notna() & out["team"].astype(str).ne("") & out["player_name_key"].ne("")].copy()
    out["week"] = out["week"].astype(int)
    if out["status"].ne("").any():
        keep = out["status"].isin(ACTIVE_ROSTER_STATUS)
        if keep.any():
            out = out.loc[keep].copy()
    return out.drop_duplicates(["season", "week", "team", "player_name_key"], keep="first").reset_index(drop=True)


def load_rosters(source_meta: list[dict]) -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        raw = m78.load_weekly_rosters(s, source_meta)
        parts.append(normalize_roster(raw, s))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_usage() -> pd.DataFrame:
    parts = []
    for s in HISTORY_SEASONS:
        raw = pf._load_weekly(s)
        q = pf._normalize_weekly(raw, s)
        q = q[[
            "season", "week", "team", "player_id", "position",
            "targets", "rushes", "tgt_share_game", "rush_share_game"
        ]].copy()
        q["family"] = q["position"].map(family)
        parts.append(q)
    x = pd.concat(parts, ignore_index=True)
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].astype("string").fillna("").str.strip()
    x["ordinal"] = x["season"] * 100 + x["week"]
    return x


def load_snaps(source_meta: list[dict]) -> pd.DataFrame:
    as_of = datetime.now(timezone.utc).isoformat()
    parts = [
        m76.download_table("pfr_snap_counts", s, m76.release_urls("snaps", s), source_meta, as_of)
        for s in HISTORY_SEASONS
    ]
    raw = pd.concat([q for q in parts if not q.empty], ignore_index=True) if any(not q.empty for q in parts) else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", "off_pct", "ordinal"])
    raw, _, _ = m76.build_id_bridge(raw, pd.DataFrame(), source_meta, as_of)
    q = m77.prepare_snap_history(raw)
    if q.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", "off_pct", "ordinal"])
    q["ordinal"] = q["season"].astype(int) * 100 + q["week"].astype(int)
    return q


def teamweek_inactive_maps(inactive_rows: pd.DataFrame):
    all_names = {}
    fam_names = {}
    for key, g in inactive_rows.groupby(["season", "week", "team"], sort=False):
        all_names[key] = set(g["inactive_name_key"].astype(str))
        for fam in FAMILY_POS:
            fam_names[(*key, fam)] = set(g.loc[g["family"].eq(fam), "inactive_name_key"].astype(str))
    return all_names, fam_names


def build_successor_rows(roster: pd.DataFrame, inactive_rows: pd.DataFrame) -> pd.DataFrame:
    all_names, fam_names = teamweek_inactive_maps(inactive_rows)
    events = (
        inactive_rows.loc[inactive_rows["family"].isin(FAMILY_POS)]
        .groupby(["season", "week", "team", "family"], as_index=False)
        .agg(inactive_players=("inactive_name_key", "nunique"))
    )
    rows = []
    roster_groups = {
        key: g for key, g in roster.groupby(["season", "week", "team", "family"], sort=False)
    }
    for e in events.itertuples(index=False):
        key = (int(e.season), int(e.week), str(e.team), str(e.family))
        g = roster_groups.get(key)
        if g is None or g.empty:
            continue
        inactive_any = all_names.get((int(e.season), int(e.week), str(e.team)), set())
        for r in g.itertuples(index=False):
            if str(r.player_name_key) in inactive_any:
                continue
            rows.append({
                "season": int(e.season),
                "week": int(e.week),
                "team": str(e.team),
                "family": str(e.family),
                "inactive_players_in_family": int(e.inactive_players),
                "player_id": str(r.player_id),
                "player_name_key": str(r.player_name_key),
                "position": str(r.position),
                "status": str(r.status),
                "stable_roster_identity": int(bool(str(r.player_id).strip())),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(["season", "week", "team", "family", "player_name_key"]).reset_index(drop=True)


def attach_prior_support(candidates: pd.DataFrame, usage: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    use_groups = {
        (pid, team): g.sort_values("ordinal")
        for (pid, team), g in usage.loc[usage["player_id"].ne("")].groupby(["player_id", "team"], sort=False)
    }
    snap_groups = {
        pid: g.sort_values("ordinal")
        for pid, g in snaps.loc[snaps["player_id"].astype(str).ne("")].groupby("player_id", sort=False)
    }
    prior_usage = []
    prior_snap = []
    chronology_bad = []
    for r in out.itertuples(index=False):
        target = int(r.season) * 100 + int(r.week)
        ug = use_groups.get((str(r.player_id), str(r.team)))
        uq = ug.loc[ug["ordinal"].lt(target)] if ug is not None else pd.DataFrame()
        sg = snap_groups.get(str(r.player_id))
        sq = sg.loc[sg["ordinal"].lt(target)] if sg is not None else pd.DataFrame()
        prior_usage.append(int(not uq.empty))
        if not sq.empty and "off_pct" in sq.columns:
            v = num(sq["off_pct"]).dropna()
            prior_snap.append(int(len(v) > 0))
        else:
            prior_snap.append(0)
        bad = 0
        if ug is not None and not uq.empty and (uq["ordinal"] >= target).any():
            bad += 1
        if sg is not None and not sq.empty and (sq["ordinal"] >= target).any():
            bad += 1
        chronology_bad.append(bad)
    out["strict_prior_same_team_usage"] = prior_usage
    out["strict_prior_snap"] = prior_snap
    out["chronology_violations"] = chronology_bad
    return out


def attach_two_state_support(
    candidates: pd.DataFrame,
    roster: pd.DataFrame,
    inactive_rows: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    _, fam_names = teamweek_inactive_maps(inactive_rows)
    roster_by_player = {
        (str(pid), str(team), str(fam)): g.sort_values(["season", "week"])
        for (pid, team, fam), g in roster.loc[
            roster["family"].isin(FAMILY_POS) & roster["player_id"].astype(str).ne("")
        ].groupby(["player_id", "team", "family"], sort=False)
    }
    out = candidates.copy()
    states = []
    for r in out.itertuples(index=False):
        target = int(r.season) * 100 + int(r.week)
        g = roster_by_player.get((str(r.player_id), str(r.team), str(r.family)))
        seen = set()
        if g is not None:
            for h in g.itertuples(index=False):
                ord_ = int(h.season) * 100 + int(h.week)
                if ord_ >= target:
                    continue
                inact = set(fam_names.get((int(h.season), int(h.week), str(h.team), str(h.family)), set()))
                if str(h.player_name_key) in inact:
                    continue
                competitors = inact - {str(h.player_name_key)}
                seen.add("COMPETITOR_INACTIVE" if competitors else "NO_COMPETITOR_INACTIVE")
        states.append(int(len(seen) >= 2))
    out["two_strict_prior_room_states"] = states
    return out


def season_family_metrics(
    inactive_rows: pd.DataFrame,
    candidates: pd.DataFrame,
    season: int,
    fam: str,
) -> dict:
    iq = inactive_rows.loc[(inactive_rows["season"].eq(season)) & (inactive_rows["family"].eq(fam))]
    cq = candidates.loc[(candidates["season"].eq(season)) & (candidates["family"].eq(fam))] if not candidates.empty else pd.DataFrame()
    event_games = int(iq[["season", "week", "team"]].drop_duplicates().shape[0])
    inactive_bridge = float(iq["roster_identity_match"].mean()) if len(iq) else 0.0
    pos_map = float(iq["family"].eq(fam).mean()) if len(iq) else 0.0
    stable = float(cq["stable_roster_identity"].mean()) if len(cq) else 0.0
    prior_usage = float(cq["strict_prior_same_team_usage"].mean()) if len(cq) else 0.0
    prior_snap = float(cq["strict_prior_snap"].mean()) if len(cq) else 0.0
    successors = int(cq.loc[cq["stable_roster_identity"].eq(1), "player_id"].nunique()) if len(cq) else 0
    two_state_players = int(
        cq.loc[cq["two_strict_prior_room_states"].eq(1) & cq["stable_roster_identity"].eq(1), "player_id"].nunique()
    ) if len(cq) else 0
    chronology = int(cq["chronology_violations"].sum()) if len(cq) else 0
    return {
        "season": season,
        "family": fam,
        "inactive_rows": int(len(iq)),
        "candidate_successor_rows": int(len(cq)),
        "inactive_roster_identity_bridge": inactive_bridge,
        "position_family_mapping": pos_map,
        "candidate_stable_identity": stable,
        "strict_prior_usage_coverage": prior_usage,
        "strict_prior_snap_coverage": prior_snap,
        "event_team_games": event_games,
        "distinct_successors": successors,
        "two_prior_room_state_players": two_state_players,
        "chronology_violations": chronology,
    }


def gate_metrics(m: dict) -> dict:
    fam = m["family"]
    return {
        "inactive_identity_bridge_ge_90": m["inactive_roster_identity_bridge"] >= 0.90,
        "position_family_mapping_ge_95": m["position_family_mapping"] >= 0.95,
        "successor_stable_identity_ge_90": m["candidate_stable_identity"] >= 0.90,
        "strict_prior_usage_ge_90": m["strict_prior_usage_coverage"] >= 0.90,
        "strict_prior_snap_ge_85": m["strict_prior_snap_coverage"] >= 0.85,
        "event_team_games_support": m["event_team_games"] >= EVENT_MIN[fam],
        "distinct_successor_support": m["distinct_successors"] >= SUCCESSOR_MIN[fam],
        "two_prior_room_state_support": m["two_prior_room_state_players"] >= TWO_STATE_MIN[fam],
        "chronology_violations_zero": m["chronology_violations"] == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inactives", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teamweeks, inactive_rows = parse_inactives(args.inactives)
    source_meta: list[dict] = []
    roster = load_rosters(source_meta)
    usage = load_usage()
    snaps = load_snaps(source_meta)

    candidates = build_successor_rows(roster, inactive_rows)
    candidates = attach_prior_support(candidates, usage, snaps)
    candidates = attach_two_state_support(candidates, roster, inactive_rows)

    metric_rows = []
    gates = {}
    pos_disposition = {}
    for fam in ("RB", "WR", "TE"):
        fam_pass = True
        for season in SEASONS:
            m = season_family_metrics(inactive_rows, candidates, season, fam)
            g = gate_metrics(m)
            metric_rows.append({**m, **{f"gate_{k}": bool(v) for k, v in g.items()}})
            fam_pass = fam_pass and all(g.values())
        pos_disposition[fam] = READY[fam] if fam_pass else BLOCKED[fam]
        gates[fam] = fam_pass

    if all(gates.values()):
        overall = "PLAYER_ROLE_ELASTICITY_SOURCE_READY_ALL"
    elif any(gates.values()):
        overall = "PLAYER_ROLE_ELASTICITY_SOURCE_READY_PARTIAL"
    else:
        overall = "PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED"

    result = {
        "version": VERSION,
        "disposition": overall,
        "position_dispositions": pos_disposition,
        "m78_sha256": M78_SHA,
        "m78_teamweek_rows": int(len(teamweeks)),
        "inactive_player_rows": int(len(inactive_rows)),
        "candidate_successor_rows": int(len(candidates)),
        "metric_rows": metric_rows,
        "target_game_outcomes_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "predictive_models_fit": 0,
        "projection_values_changed": 0,
        "production_changed": False,
    }

    pd.DataFrame(source_meta).to_csv(args.out_dir / "player_role_elasticity_source_snapshots.csv", index=False)
    inactive_rows.to_csv(args.out_dir / "player_role_elasticity_inactive_rows.csv", index=False)
    roster.to_csv(args.out_dir / "player_role_elasticity_roster_rows.csv", index=False)
    candidates.to_csv(args.out_dir / "player_role_elasticity_successor_rows.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "player_role_elasticity_source_metrics.csv", index=False)
    (args.out_dir / "player_role_elasticity_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
