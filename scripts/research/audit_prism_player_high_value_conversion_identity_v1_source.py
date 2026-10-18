#!/usr/bin/env python3
"""Source/support audit for PRISM named-player scoring-area TD conversion identity V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research import audit_player_high_value_scoring_opportunity_identity_v1 as hv
from scripts.utils.pbp import get_pbp

VERSION = "PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_V1_SOURCE_AUDIT"
SEASONS = (2022, 2023, 2024, 2025)
TARGET_SEASONS = (2024, 2025)
MIN_PRIOR_OPPS = 3
MIN_SUPPORTED_PLAYER_GAMES = 350
MIN_SUPPORTED_PLAYERS = 40


def num(x):
    return pd.to_numeric(x, errors="coerce")


def first(df, names):
    return next((c for c in names if c in df.columns), None)


def _regular_pbp(season: int) -> pd.DataFrame:
    p = get_pbp(season, min_rows=1000).copy()
    p.columns = [str(c).lower() for c in p.columns]
    if "season_type" in p.columns:
        q = p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if not q.empty:
            p = q
    return p


def _base_events(season: int, roster: pd.DataFrame) -> pd.DataFrame:
    p = _regular_pbp(season)
    wc = first(p, ["week"])
    tc = first(p, ["posteam"])
    yc = first(p, ["yardline_100"])
    gc = first(p, ["game_id", "nflverse_game_id", "old_game_id"])
    pc = first(p, ["play_id"])
    rc = first(p, ["rusher_player_id", "rusher_id", "rusher_gsis_id"])
    xc = first(p, ["receiver_player_id", "receiver_id", "receiver_gsis_id"])
    rtc = first(p, ["rush_touchdown"])
    ptc = first(p, ["pass_touchdown"])
    cc = first(p, ["complete_pass"])
    required = [wc, tc, yc, gc, pc, rc, xc, rtc, ptc]
    if not all(required):
        raise RuntimeError(f"PBP conversion schema missing for {season}: {required}")

    week = num(p[wc])
    team = p[tc].map(canon_team)
    yard = num(p[yc])
    rush = num(p.get("rush_attempt", pd.Series(0, index=p.index))).eq(1)
    pas = num(p.get("pass_attempt", pd.Series(0, index=p.index))).eq(1)
    rpid = p[rc].astype("string").fillna("").str.strip()
    xpid = p[xc].astype("string").fillna("").str.strip()
    receiver_name_col = first(p, ["receiver_player_name", "receiver_name"])
    has_receiver = (
        p[receiver_name_col].astype("string").fillna("").str.strip().ne("")
        if receiver_name_col else xpid.ne("")
    )

    rush_ev = pd.DataFrame({
        "season": season,
        "week": week,
        "team": team,
        "game_id": p[gc].astype(str),
        "play_id": p[pc].astype(str),
        "player_id": rpid,
        "yardline_100": yard,
        "td_raw": num(p[rtc]),
        "complete_pass": np.nan,
    }).loc[rush & yard.le(10)].copy()
    rush_ev["channel"] = "RB_RUSH_INSIDE10"

    rec_ev = pd.DataFrame({
        "season": season,
        "week": week,
        "team": team,
        "game_id": p[gc].astype(str),
        "play_id": p[pc].astype(str),
        "player_id": xpid,
        "yardline_100": yard,
        "td_raw": num(p[ptc]),
        "complete_pass": num(p[cc]) if cc else np.nan,
    }).loc[pas & has_receiver & yard.le(20)].copy()

    r = roster.loc[
        roster.season.eq(season),
        ["season", "week", "team", "player_id", "family"],
    ].drop_duplicates(["season", "week", "team", "player_id"])

    rush_ev = rush_ev.merge(
        r, on=["season", "week", "team", "player_id"], how="left", validate="many_to_one"
    )
    rec_ev = rec_ev.merge(
        r, on=["season", "week", "team", "player_id"], how="left", validate="many_to_one"
    )

    rush_ev = rush_ev.loc[rush_ev.family.eq("RB")].copy()
    rb_rec = rec_ev.loc[rec_ev.family.eq("RB")].copy()
    rb_rec["channel"] = "RB_RECEIVING_REDZONE"
    te_rec = rec_ev.loc[rec_ev.family.eq("TE")].copy()
    te_rec["channel"] = "TE_RECEIVING_REDZONE"
    return pd.concat([rush_ev, rb_rec, te_rec], ignore_index=True)


def add_prior_support(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-games first; target game cannot enter its own prior state."""
    if events.empty:
        return pd.DataFrame()
    g = (
        events.groupby(
            ["channel", "season", "week", "game_id", "team", "player_id"],
            as_index=False,
        )
        .agg(high_value_opps=("play_id", "size"), touchdowns=("td_raw", "sum"))
    )
    g = g.sort_values(
        ["channel", "player_id", "season", "week", "game_id"],
        kind="stable",
    ).reset_index(drop=True)
    g["prior_high_value_opps"] = (
        g.groupby(["channel", "player_id"], sort=False)["high_value_opps"].cumsum()
        - g["high_value_opps"]
    )
    g["prior_touchdowns"] = (
        g.groupby(["channel", "player_id"], sort=False)["touchdowns"].cumsum()
        - g["touchdowns"]
    )
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    roster = hv.load_rosters(meta)
    events = pd.concat([_base_events(s, roster) for s in SEASONS], ignore_index=True)

    rows = []
    channels = ["RB_RUSH_INSIDE10", "RB_RECEIVING_REDZONE", "TE_RECEIVING_REDZONE"]
    for s in SEASONS:
        for ch in channels:
            q = events.loc[(events.season.eq(s)) & events.channel.eq(ch)].copy()
            td = num(q.td_raw)
            finite = td.notna()
            binary = td.loc[finite].isin([0, 1]).all()
            duplicate_rows = int(q.duplicated(["game_id", "play_id", "player_id"]).sum())
            rec_contra = 0
            if "RECEIVING" in ch:
                comp = num(q.complete_pass)
                rec_contra = int(((td.eq(1)) & (~comp.eq(1))).sum())
            rows.append({
                "season": s,
                "channel": ch,
                "events": int(len(q)),
                "distinct_players": int(q.player_id.loc[q.player_id.ne("")].nunique()),
                "tds": int(td.fillna(0).sum()),
                "td_label_coverage": float(finite.mean()) if len(q) else 0.0,
                "player_id_coverage": float(q.player_id.ne("").mean()) if len(q) else 0.0,
                "family_bridge_coverage": float(q.family.notna().mean()) if len(q) else 0.0,
                "duplicate_player_play_rows": duplicate_rows,
                "receiving_td_without_complete_pass": rec_contra,
                "td_label_binary": bool(binary),
                "gate_events_nonzero": bool(len(q) > 0),
                "gate_td_label_coverage_ge_999": bool(len(q) > 0 and finite.mean() >= 0.999),
                "gate_player_id_coverage_ge_99": bool(len(q) > 0 and q.player_id.ne("").mean() >= 0.99),
                "gate_family_bridge_coverage_ge_98": bool(len(q) > 0 and q.family.notna().mean() >= 0.98),
                "gate_duplicate_player_play_zero": bool(duplicate_rows == 0),
                "gate_receiving_td_complete_consistency": bool(rec_contra == 0),
                "gate_td_label_binary": bool(binary),
            })

    metrics = pd.DataFrame(rows)
    games = add_prior_support(events)
    support_rows = []
    readiness = {}
    source_gate_cols = [c for c in metrics.columns if c.startswith("gate_")]

    for ch in channels:
        channel_source_ok = bool(
            len(metrics.loc[metrics.channel.eq(ch)])
            and metrics.loc[metrics.channel.eq(ch), source_gate_cols].all(axis=None)
        )
        target_ok = True
        for s in TARGET_SEASONS:
            q = games.loc[
                games.channel.eq(ch)
                & games.season.eq(s)
                & games.prior_high_value_opps.ge(MIN_PRIOR_OPPS)
            ].copy()
            n_pg = int(len(q))
            n_pl = int(q.player_id.nunique())
            ok = n_pg >= MIN_SUPPORTED_PLAYER_GAMES and n_pl >= MIN_SUPPORTED_PLAYERS
            target_ok = target_ok and ok
            support_rows.append({
                "season": s,
                "channel": ch,
                "min_prior_high_value_opps": MIN_PRIOR_OPPS,
                "supported_player_games": n_pg,
                "supported_players": n_pl,
                "gate_supported_player_games_ge_350": bool(n_pg >= MIN_SUPPORTED_PLAYER_GAMES),
                "gate_supported_players_ge_40": bool(n_pl >= MIN_SUPPORTED_PLAYERS),
                "support_ready": bool(ok),
            })
        readiness[ch] = bool(channel_source_ok and target_ok)

    ready_n = sum(readiness.values())
    if ready_n == len(channels):
        disposition = "PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_READY_ALL"
    elif ready_n:
        disposition = "PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_READY_PARTIAL"
    else:
        disposition = "PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_BLOCKED"

    support = pd.DataFrame(support_rows)
    result = {
        "version": VERSION,
        "disposition": disposition,
        "channel_readiness": readiness,
        "frozen_min_prior_high_value_opps": MIN_PRIOR_OPPS,
        "frozen_min_supported_player_games": MIN_SUPPORTED_PLAYER_GAMES,
        "frozen_min_supported_players": MIN_SUPPORTED_PLAYERS,
        "predictive_candidates_scored": 0,
        "outcomes_2026_read": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
        "metrics": rows,
        "support": support_rows,
    }

    pd.DataFrame(meta).to_csv(a.out_dir / "source_manifest.csv", index=False)
    metrics.to_csv(a.out_dir / "source_metrics.csv", index=False)
    games.to_csv(a.out_dir / "player_game_prior_support.csv", index=False)
    support.to_csv(a.out_dir / "support_metrics.csv", index=False)
    (a.out_dir / "source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
