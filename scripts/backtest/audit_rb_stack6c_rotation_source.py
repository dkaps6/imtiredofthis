#!/usr/bin/env python3
"""RB STACK6C / ND4: secondary-back drive/series rotation source audit.

No outcome model is fit. No sportsbook data is loaded.

Historical participation is used as delayed on-field truth for completed games.
A separate PBP-only rush+target drive proxy is evaluated against that truth so
we can determine whether the useful rotation state has a live-season analogue.
Target-game participation is never emitted as a pregame feature; only strictly
lagged prior-1/prior-3 proxy values are produced.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

JOIN_GATE = 0.95
ALIGN_GATE = 0.95
DRIVE_GATE = 0.95
CORR_GATE = 0.60
TOP_AGREE_GATE = 0.70
PRIOR3_COVERAGE_GATE = 0.75

CORE_PROXY_FEATURES = [
    "touch_opp_share",
    "touch_drive_share",
    "touch_lead_drive_share",
    "opening_drive_touch_share",
    "team_touch_leader_switch_rate",
    "team_touch_hhi",
]


def to_pd(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def lower(x):
    z = x.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def num(x):
    return pd.to_numeric(x, errors="coerce")


def split_list(v):
    s = "" if pd.isna(v) else str(v).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return []
    return [q.strip() for q in re.split(r"[;,]", s) if q.strip()]


def clean_id(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def load_sources(seasons):
    import nflreadpy as nfl

    part = lower(to_pd(nfl.load_participation(seasons=seasons)))
    pbp = lower(to_pd(nfl.load_pbp(seasons=seasons)))
    return part, pbp


def derive_game_fields(x):
    z = x.copy()
    if "season" not in z.columns or "week" not in z.columns:
        gid = next((c for c in ["nflverse_game_id", "game_id", "old_game_id"] if c in z.columns), None)
        if gid:
            p = z[gid].astype("string").str.split("_", expand=True)
            if "season" not in z.columns:
                z["season"] = num(p[0])
            if "week" not in z.columns and p.shape[1] > 1:
                z["week"] = num(p[1])
    return z


def find_join_keys(part, pbp):
    for keys in (
        ["nflverse_game_id", "play_id"],
        ["old_game_id", "play_id"],
        ["game_id", "play_id"],
    ):
        if all(k in part.columns and k in pbp.columns for k in keys):
            return list(keys)
    raise RuntimeError("STACK6C no common participation/PBP game-play key")


def choose_existing(z, candidates):
    return next((c for c in candidates if c in z.columns), None)


def regular_only(z):
    if "season_type" not in z.columns:
        return z.copy()
    reg = z.loc[z.season_type.astype(str).str.upper().eq("REG")].copy()
    return reg if len(reg) else z.copy()


def build_join(part, pbp):
    part = derive_game_fields(part)
    pbp = derive_game_fields(pbp)
    keys = find_join_keys(part, pbp)

    wanted = [
        *keys,
        "season",
        "week",
        "season_type",
        "posteam",
        "possession_team",
        "fixed_drive",
        "drive",
        "rush_attempt",
        "rusher_player_id",
        "rusher_id",
        "receiver_player_id",
        "receiver_id",
    ]
    right = pbp[[c for c in wanted if c in pbp.columns]].drop_duplicates(keys)
    j = part.merge(right, on=keys, how="inner", suffixes=("", "_pbp"), validate="one_to_one")

    # Normalize season/week from whichever side survived without ambiguity.
    for field in ["season", "week", "season_type"]:
        if field not in j.columns and f"{field}_pbp" in j.columns:
            j[field] = j[f"{field}_pbp"]
    j = regular_only(j)

    team_col = choose_existing(j, ["posteam", "posteam_pbp", "possession_team", "possession_team_pbp"])
    if not team_col:
        raise RuntimeError("STACK6C joined data missing possession team")
    j["team"] = j[team_col].map(canon_team)
    j = j.loc[j.team.ne("")].copy()
    j["season"] = num(j.season).astype(int)
    j["week"] = num(j.week).astype(int)

    game_col = keys[0]
    j["game_key"] = j[game_col].astype(str)
    j["play_order"] = num(j["play_id"])

    drive_col = choose_existing(j, ["fixed_drive", "fixed_drive_pbp", "drive", "drive_pbp"])
    if drive_col:
        j["drive_id"] = j[drive_col].astype("string")
        invalid = j["drive_id"].isna() | j["drive_id"].str.lower().isin(["nan", "none", "<na>", ""])
        j.loc[invalid, "drive_id"] = pd.NA
    else:
        j["drive_id"] = pd.NA
    return j, keys


def parse_participation_plays(j):
    player_col = choose_existing(j, ["offense_players"])
    pos_col = choose_existing(j, ["offense_positions"])
    if not player_col or not pos_col:
        raise RuntimeError("STACK6C participation missing offense player/position arrays")

    play_rows = []
    rb_rows = []
    eligible = 0
    aligned = 0

    for _, r in j.iterrows():
        ids = split_list(r[player_col])
        pos = split_list(r[pos_col])
        if ids or pos:
            eligible += 1
        if not ids or len(ids) != len(pos):
            continue
        aligned += 1

        rb_ids = [clean_id(pid) for pid, pp in zip(ids, pos) if str(pp).upper() in {"RB", "FB"} and clean_id(pid)]
        rec = {
            "season": int(r.season),
            "week": int(r.week),
            "team": r.team,
            "game_key": str(r.game_key),
            "drive_id": r.drive_id,
            "play_id": r.play_order,
            "rb_count_onfield": len(rb_ids),
        }
        play_rows.append(rec)
        for pid in rb_ids:
            rb_rows.append({**rec, "player_id": pid})

    plays = pd.DataFrame(play_rows)
    rb = pd.DataFrame(rb_rows)
    if plays.empty or rb.empty:
        raise RuntimeError("STACK6C participation parse produced no RB plays")
    return plays, rb, {
        "eligible_array_plays": eligible,
        "aligned_array_plays": aligned,
        "aligned_rate": aligned / max(eligible, 1),
    }


def add_drive_index(plays):
    z = plays.loc[plays.drive_id.notna()].copy()
    first = (
        z.groupby(["season", "week", "team", "game_key", "drive_id"], as_index=False)
        .agg(drive_first_play=("play_id", "min"), drive_offensive_plays=("play_id", "nunique"))
        .sort_values(["season", "week", "team", "game_key", "drive_first_play", "drive_id"])
    )
    first["drive_index"] = first.groupby(["season", "week", "team", "game_key"]).cumcount() + 1
    first["team_drive_count"] = first.groupby(["season", "week", "team", "game_key"])["drive_id"].transform("nunique")
    return first


def switch_rate(drive_leaders, leader_col):
    rows = []
    keys = ["season", "week", "team", "game_key"]
    for vals, g in drive_leaders.sort_values(keys + ["drive_index"]).groupby(keys, sort=False):
        q = g.loc[g[leader_col].astype(str).ne("")].copy()
        if len(q) >= 2:
            rate = float(q[leader_col].astype(str).ne(q[leader_col].astype(str).shift(1)).iloc[1:].mean())
        else:
            rate = np.nan
        rows.append({**dict(zip(keys, vals)), "switch_rate": rate, "leader_observed_drives": int(len(q))})
    return pd.DataFrame(rows)


def build_onfield_truth(plays, rb):
    drive = add_drive_index(plays)
    rb = rb.merge(
        drive[["season", "week", "team", "game_key", "drive_id", "drive_index", "drive_offensive_plays", "team_drive_count"]],
        on=["season", "week", "team", "game_key", "drive_id"],
        how="inner",
        validate="many_to_one",
    )

    # Play-level co-presence is direct truth from participation arrays.
    rb["multi_rb_play"] = num(rb.rb_count_onfield).ge(2)

    pdv = (
        rb.groupby(["season", "week", "team", "game_key", "drive_id", "drive_index", "player_id"], as_index=False)
        .agg(
            player_drive_onfield_plays=("play_id", "nunique"),
            player_multi_rb_plays=("multi_rb_play", "sum"),
            drive_offensive_plays=("drive_offensive_plays", "first"),
            team_drive_count=("team_drive_count", "first"),
        )
    )
    pdv["drive_onfield_rate"] = num(pdv.player_drive_onfield_plays) / num(pdv.drive_offensive_plays).replace(0, np.nan)
    pdv["drive_top_plays"] = pdv.groupby(["season", "week", "team", "game_key", "drive_id"])["player_drive_onfield_plays"].transform("max")
    pdv["is_drive_top"] = pdv.player_drive_onfield_plays.eq(pdv.drive_top_plays)
    pdv["drive_top_tie_count"] = pdv.groupby(["season", "week", "team", "game_key", "drive_id"])["is_drive_top"].transform("sum")
    pdv["drive_lead_credit"] = np.where(pdv.is_drive_top, 1.0 / num(pdv.drive_top_tie_count).replace(0, np.nan), 0.0)

    keys = ["season", "week", "team", "game_key", "player_id"]
    pg = (
        pdv.groupby(keys, as_index=False)
        .agg(
            drives_present=("drive_id", "nunique"),
            sum_drive_onfield_rate=("drive_onfield_rate", "sum"),
            lead_drive_credit=("drive_lead_credit", "sum"),
            team_drive_count=("team_drive_count", "max"),
            player_onfield_plays=("player_drive_onfield_plays", "sum"),
            player_multi_rb_plays=("player_multi_rb_plays", "sum"),
        )
    )
    pg["drive_presence_share"] = num(pg.drives_present) / num(pg.team_drive_count).replace(0, np.nan)
    pg["mean_drive_onfield_rate"] = num(pg.sum_drive_onfield_rate) / num(pg.team_drive_count).replace(0, np.nan)
    pg["lead_drive_share"] = num(pg.lead_drive_credit) / num(pg.team_drive_count).replace(0, np.nan)
    pg["multi_rb_copresence_share"] = num(pg.player_multi_rb_plays) / num(pg.player_onfield_plays).replace(0, np.nan)
    pg["single_rb_presence_share"] = 1.0 - pg.multi_rb_copresence_share

    total_presence = pg.groupby(["season", "week", "team", "game_key"])["player_onfield_plays"].transform("sum").replace(0, np.nan)
    pg["rb_onfield_presence_share"] = num(pg.player_onfield_plays) / total_presence

    opening = pdv.loc[pdv.drive_index.eq(1), keys + ["drive_onfield_rate"]].copy()
    opening = opening.rename(columns={"drive_onfield_rate": "opening_drive_onfield_rate"})
    opening["opening_drive_present"] = 1
    pg = pg.merge(opening, on=keys, how="left", validate="one_to_one")
    pg["opening_drive_onfield_rate"] = num(pg.opening_drive_onfield_rate).fillna(0.0)
    pg["opening_drive_present"] = num(pg.opening_drive_present).fillna(0).astype(int)

    team_keys = ["season", "week", "team", "game_key"]
    play_team = (
        plays.loc[plays.drive_id.notna()]
        .groupby(team_keys, as_index=False)
        .agg(
            team_offensive_plays=("play_id", "nunique"),
            team_multi_rb_plays=("rb_count_onfield", lambda s: int(num(s).ge(2).sum())),
            team_unique_drives=("drive_id", "nunique"),
        )
    )
    play_team["team_multi_rb_play_rate"] = num(play_team.team_multi_rb_plays) / num(play_team.team_offensive_plays).replace(0, np.nan)

    # Fractional drive-lead credit sums to 1 on every drive even when tied.
    lead_share = pdv.groupby(team_keys + ["player_id"], as_index=False).agg(lead_credit=("drive_lead_credit", "sum"))
    team_drive_counts = drive.groupby(team_keys, as_index=False).agg(team_drive_count=("drive_id", "nunique"))
    lead_share = lead_share.merge(team_drive_counts, on=team_keys, how="left")
    lead_share["lead_share"] = num(lead_share.lead_credit) / num(lead_share.team_drive_count).replace(0, np.nan)
    hhi = lead_share.groupby(team_keys, as_index=False).agg(team_drive_lead_hhi=("lead_share", lambda s: float(np.square(num(s).fillna(0)).sum())))

    presence_hhi = pg.groupby(team_keys, as_index=False).agg(
        team_rb_presence_hhi=("rb_onfield_presence_share", lambda s: float(np.square(num(s).fillna(0)).sum())),
        team_unique_rbs_onfield=("player_id", "nunique"),
    )

    # Unique most-present RB by drive; tied drives are explicitly excluded from switch calculations.
    top = pdv.loc[pdv.is_drive_top].copy()
    unique_top = top.loc[num(top.drive_top_tie_count).eq(1), team_keys + ["drive_id", "drive_index", "player_id"]].rename(columns={"player_id": "onfield_drive_leader"})
    sw = switch_rate(unique_top, "onfield_drive_leader").rename(columns={"switch_rate": "team_onfield_leader_switch_rate", "leader_observed_drives": "onfield_unique_leader_drives"})
    tie = (
        pdv.groupby(team_keys + ["drive_id"], as_index=False)
        .agg(tie_count=("drive_top_tie_count", "max"))
        .groupby(team_keys, as_index=False)
        .agg(team_onfield_leader_tie_rate=("tie_count", lambda s: float(num(s).gt(1).mean())))
    )

    team = play_team.merge(hhi, on=team_keys, how="left").merge(presence_hhi, on=team_keys, how="left").merge(sw, on=team_keys, how="left").merge(tie, on=team_keys, how="left")
    pg = pg.merge(team, on=team_keys, how="left", validate="many_to_one")
    return pg, pdv, drive, unique_top


def prep_pbp_for_touch(pbp):
    z = derive_game_fields(pbp)
    z = regular_only(z)
    team_col = choose_existing(z, ["posteam", "possession_team"])
    if not team_col:
        raise RuntimeError("STACK6C PBP missing possession team")
    z["team"] = z[team_col].map(canon_team)
    z = z.loc[z.team.ne("")].copy()
    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    game_col = choose_existing(z, ["nflverse_game_id", "old_game_id", "game_id"])
    if not game_col:
        raise RuntimeError("STACK6C PBP missing game id")
    z["game_key"] = z[game_col].astype(str)
    z["play_id"] = num(z.play_id)
    drive_col = choose_existing(z, ["fixed_drive", "drive"])
    if not drive_col:
        z["drive_id"] = pd.NA
    else:
        z["drive_id"] = z[drive_col].astype("string")
        invalid = z.drive_id.isna() | z.drive_id.str.lower().isin(["nan", "none", "<na>", ""])
        z.loc[invalid, "drive_id"] = pd.NA
    return z


def build_touch_proxy(pbp, rb_universe):
    z = prep_pbp_for_touch(pbp)
    rusher_col = choose_existing(z, ["rusher_player_id", "rusher_id"])
    receiver_col = choose_existing(z, ["receiver_player_id", "receiver_id"])
    if not rusher_col or not receiver_col:
        raise RuntimeError("STACK6C PBP missing rusher or receiver ids")

    rb_keys = set(zip(rb_universe.season.astype(int), rb_universe.team.astype(str), rb_universe.player_id.astype(str)))
    events = []

    rush_mask = num(z.get("rush_attempt", 0)).fillna(0).eq(1)
    for r in z.loc[rush_mask & z.drive_id.notna(), ["season", "week", "team", "game_key", "drive_id", "play_id", rusher_col]].itertuples(index=False, name=None):
        season, week, team, game_key, drive_id, play_id, pid = r
        pid = clean_id(pid)
        if pid and (int(season), str(team), pid) in rb_keys:
            events.append({"season": int(season), "week": int(week), "team": team, "game_key": str(game_key), "drive_id": drive_id, "play_id": play_id, "player_id": pid, "event_type": "RUSH"})

    # nflfastR receiver id is populated on targeted pass plays. Using the identity
    # itself does not require participation; participation only defines RB position in this audit.
    recv = z.loc[z.drive_id.notna(), ["season", "week", "team", "game_key", "drive_id", "play_id", receiver_col]]
    for r in recv.itertuples(index=False, name=None):
        season, week, team, game_key, drive_id, play_id, pid = r
        pid = clean_id(pid)
        if pid and (int(season), str(team), pid) in rb_keys:
            events.append({"season": int(season), "week": int(week), "team": team, "game_key": str(game_key), "drive_id": drive_id, "play_id": play_id, "player_id": pid, "event_type": "TARGET"})

    ev = pd.DataFrame(events)
    if ev.empty:
        raise RuntimeError("STACK6C touch proxy found no RB rush/target events")
    ev = ev.drop_duplicates(["season", "week", "team", "game_key", "drive_id", "play_id", "player_id", "event_type"])

    # Drive order comes from ordinary PBP, independent of participation arrays.
    drives = (
        z.loc[z.drive_id.notna()]
        .groupby(["season", "week", "team", "game_key", "drive_id"], as_index=False)
        .agg(drive_first_play=("play_id", "min"))
        .sort_values(["season", "week", "team", "game_key", "drive_first_play", "drive_id"])
    )
    drives["drive_index"] = drives.groupby(["season", "week", "team", "game_key"]).cumcount() + 1
    drives["team_drive_count"] = drives.groupby(["season", "week", "team", "game_key"])["drive_id"].transform("nunique")

    ev = ev.merge(drives, on=["season", "week", "team", "game_key", "drive_id"], how="left", validate="many_to_one")
    tdrive = (
        ev.groupby(["season", "week", "team", "game_key", "drive_id", "drive_index", "player_id"], as_index=False)
        .agg(touch_opps=("event_type", "count"), rush_opps=("event_type", lambda s: int((s == "RUSH").sum())), target_opps=("event_type", lambda s: int((s == "TARGET").sum())), team_drive_count=("team_drive_count", "first"))
    )
    tdrive["drive_top_touch"] = tdrive.groupby(["season", "week", "team", "game_key", "drive_id"])["touch_opps"].transform("max")
    tdrive["is_touch_drive_top"] = tdrive.touch_opps.eq(tdrive.drive_top_touch)
    tdrive["touch_top_tie_count"] = tdrive.groupby(["season", "week", "team", "game_key", "drive_id"])["is_touch_drive_top"].transform("sum")
    tdrive["touch_lead_credit"] = np.where(tdrive.is_touch_drive_top, 1.0 / num(tdrive.touch_top_tie_count).replace(0, np.nan), 0.0)

    keys = ["season", "week", "team", "game_key", "player_id"]
    pg = (
        tdrive.groupby(keys, as_index=False)
        .agg(
            touch_opps=("touch_opps", "sum"),
            rush_opps=("rush_opps", "sum"),
            target_opps=("target_opps", "sum"),
            touch_drives=("drive_id", "nunique"),
            touch_lead_credit=("touch_lead_credit", "sum"),
            team_drive_count=("team_drive_count", "max"),
        )
    )
    team_total = pg.groupby(["season", "week", "team", "game_key"])["touch_opps"].transform("sum").replace(0, np.nan)
    pg["touch_opp_share"] = num(pg.touch_opps) / team_total
    pg["touch_drive_share"] = num(pg.touch_drives) / num(pg.team_drive_count).replace(0, np.nan)
    pg["touch_lead_drive_share"] = num(pg.touch_lead_credit) / num(pg.team_drive_count).replace(0, np.nan)

    opening = tdrive.loc[tdrive.drive_index.eq(1)].groupby(keys, as_index=False).agg(opening_drive_touch_opps=("touch_opps", "sum"))
    pg = pg.merge(opening, on=keys, how="left")
    pg["opening_drive_touch_opps"] = num(pg.opening_drive_touch_opps).fillna(0.0)
    opening_team = pg.groupby(["season", "week", "team", "game_key"])["opening_drive_touch_opps"].transform("sum")
    pg["opening_drive_touch_share"] = np.where(opening_team.gt(0), pg.opening_drive_touch_opps / opening_team, 0.0)

    team_keys = ["season", "week", "team", "game_key"]
    pg["team_touch_hhi"] = pg.groupby(team_keys)["touch_opp_share"].transform(lambda s: float(np.square(num(s).fillna(0)).sum()))

    top = tdrive.loc[tdrive.is_touch_drive_top].copy()
    unique = top.loc[num(top.touch_top_tie_count).eq(1), team_keys + ["drive_id", "drive_index", "player_id"]].rename(columns={"player_id": "touch_drive_leader"})
    sw = switch_rate(unique, "touch_drive_leader").rename(columns={"switch_rate": "team_touch_leader_switch_rate", "leader_observed_drives": "touch_unique_leader_drives"})
    tie = (
        tdrive.groupby(team_keys + ["drive_id"], as_index=False)
        .agg(tie_count=("touch_top_tie_count", "max"))
        .groupby(team_keys, as_index=False)
        .agg(team_touch_leader_tie_rate=("tie_count", lambda s: float(num(s).gt(1).mean())))
    )
    pg = pg.merge(sw, on=team_keys, how="left", validate="many_to_one").merge(tie, on=team_keys, how="left", validate="many_to_one")
    return pg, tdrive, drives, unique, ev, z


def merge_proxy_truth(onfield, touch):
    keys = ["season", "week", "team", "game_key", "player_id"]
    touch_cols = [c for c in touch.columns if c not in ["team_drive_count"]]
    q = onfield.merge(touch[touch_cols], on=keys, how="left", validate="one_to_one")

    zero_cols = [
        "touch_opps",
        "rush_opps",
        "target_opps",
        "touch_drives",
        "touch_lead_credit",
        "touch_opp_share",
        "touch_drive_share",
        "touch_lead_drive_share",
        "opening_drive_touch_opps",
        "opening_drive_touch_share",
    ]
    for c in zero_cols:
        if c not in q.columns:
            q[c] = 0.0
        q[c] = num(q[c]).fillna(0.0)

    # Team proxy fields can be copied from any touch-active player on that team-game.
    team_keys = ["season", "week", "team", "game_key"]
    team_proxy = touch.groupby(team_keys, as_index=False).agg(
        team_touch_hhi=("team_touch_hhi", "first"),
        team_touch_leader_switch_rate=("team_touch_leader_switch_rate", "first"),
        team_touch_leader_tie_rate=("team_touch_leader_tie_rate", "first"),
    )
    for c in ["team_touch_hhi", "team_touch_leader_switch_rate", "team_touch_leader_tie_rate"]:
        if c in q.columns:
            q = q.drop(columns=[c])
    q = q.merge(team_proxy, on=team_keys, how="left", validate="many_to_one")
    return q


def corr(a, b):
    q = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(q) < 3 or q.a.nunique() < 2 or q.b.nunique() < 2:
        return np.nan
    return float(q.a.corr(q.b))


def top_identity_agreement(q):
    rows = []
    keys = ["season", "week", "team", "game_key"]
    for vals, g in q.groupby(keys, sort=False):
        if g.empty:
            continue
        gp = g.sort_values(["rb_onfield_presence_share", "player_id"], ascending=[False, True])
        gt = g.sort_values(["touch_opp_share", "player_id"], ascending=[False, True])
        top_presence = str(gp.iloc[0].player_id)
        top_touch = str(gt.iloc[0].player_id)
        presence_tie = int((num(g.rb_onfield_presence_share) == num(g.rb_onfield_presence_share).max()).sum())
        touch_tie = int((num(g.touch_opp_share) == num(g.touch_opp_share).max()).sum())
        rows.append({
            **dict(zip(keys, vals)),
            "top_presence_player": top_presence,
            "top_touch_player": top_touch,
            "top_identity_agree": int(top_presence == top_touch),
            "presence_top_tie_count": presence_tie,
            "touch_top_tie_count": touch_tie,
        })
    out = pd.DataFrame(rows)
    agreement = float(out.top_identity_agree.mean()) if len(out) else np.nan
    return out, agreement


def add_lags(q, team_game_min_week):
    z = q.sort_values(["season", "team", "player_id", "week", "game_key"]).copy()
    z["target_order"] = num(z.season) * 100 + num(z.week)
    gkeys = ["season", "team", "player_id"]

    for c in CORE_PROXY_FEATURES:
        z[f"prior1_{c}"] = z.groupby(gkeys, sort=False)[c].shift(1)
        z[f"prior3_{c}"] = z.groupby(gkeys, sort=False)[c].transform(lambda s: num(s).shift(1).rolling(3, min_periods=1).mean())

    z["feature_source_max_order"] = z.groupby(gkeys, sort=False)["target_order"].shift(1)
    z["strict_prior_safe"] = (num(z.feature_source_max_order).isna() | num(z.feature_source_max_order).lt(num(z.target_order))).astype(int)
    z = z.merge(team_game_min_week, on=["season", "team"], how="left", validate="many_to_one")
    z["has_earlier_team_game"] = num(z.week).gt(num(z.team_first_week))
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    seasons = [int(x) for x in str(a.seasons).split(",")]

    part, pbp = load_sources(seasons)
    j, join_keys = build_join(part, pbp)

    part_keys = part[join_keys].dropna().drop_duplicates()
    pbp_keys = pbp[join_keys].dropna().drop_duplicates()
    play_join_rate = len(part_keys.merge(pbp_keys, on=join_keys, how="inner")) / max(len(part_keys), 1)

    joined_drive_coverage = float(j.drive_id.notna().mean()) if len(j) else 0.0
    plays, rb, parse = parse_participation_plays(j)
    onfield, onfield_drive, onfield_drives, onfield_unique_leaders = build_onfield_truth(plays, rb)

    rb_universe = rb[["season", "team", "player_id"]].drop_duplicates()
    touch, touch_drive, touch_drives, touch_unique_leaders, touch_events, prepared_pbp = build_touch_proxy(pbp, rb_universe)
    q = merge_proxy_truth(onfield, touch)

    presence_touch_corr = corr(q.rb_onfield_presence_share, q.touch_opp_share)
    drive_touch_corr = corr(q.drive_presence_share, q.touch_drive_share)
    top_table, top_agreement = top_identity_agreement(q)

    team_first = (
        onfield_drives.groupby(["season", "team"], as_index=False)
        .agg(team_first_week=("week", "min"))
    )
    lag = add_lags(q, team_first)
    prior3_cov_rows = []
    cov_mask = lag.season.eq(2025) & lag.has_earlier_team_game
    for c in CORE_PROXY_FEATURES:
        pc = f"prior3_{c}"
        prior3_cov_rows.append({
            "feature": pc,
            "eligible_2025_rows": int(cov_mask.sum()),
            "nonnull_rate_2025_with_earlier_team_game": float(num(lag.loc[cov_mask, pc]).notna().mean()) if cov_mask.any() else np.nan,
        })
    prior_cov = pd.DataFrame(prior3_cov_rows)
    min_prior3_coverage = float(prior_cov.nonnull_rate_2025_with_earlier_team_game.min()) if len(prior_cov) else np.nan

    infra = {
        "gate_play_join": int(play_join_rate >= JOIN_GATE),
        "gate_array_alignment": int(parse["aligned_rate"] >= ALIGN_GATE),
        "gate_drive_coverage": int(joined_drive_coverage >= DRIVE_GATE),
        "gate_strict_prior": int(float(lag.strict_prior_safe.mean()) == 1.0),
    }
    proxy = {
        "gate_presence_touch_corr": int(pd.notna(presence_touch_corr) and presence_touch_corr >= CORR_GATE),
        "gate_drive_touch_corr": int(pd.notna(drive_touch_corr) and drive_touch_corr >= CORR_GATE),
        "gate_top_identity_agreement": int(pd.notna(top_agreement) and top_agreement >= TOP_AGREE_GATE),
        "gate_prior3_coverage": int(pd.notna(min_prior3_coverage) and min_prior3_coverage >= PRIOR3_COVERAGE_GATE),
    }
    infra_pass = int(all(infra.values()))
    proxy_pass_count = int(sum(proxy.values()))
    go = int(infra_pass == 1 and proxy_pass_count >= 3)
    disposition = "GO_STACK6C_ROTATION_PROXY_BUILD" if go else "ROTATION_PROXY_INSUFFICIENT_FIND_NEW_LIVE_SOURCE"

    source = pd.DataFrame([{
        "participation_rows": int(len(part)),
        "pbp_rows": int(len(pbp)),
        "joined_rows": int(len(j)),
        "play_join_rate": float(play_join_rate),
        "array_aligned_rate": float(parse["aligned_rate"]),
        "joined_drive_id_coverage": joined_drive_coverage,
        "parsed_offensive_plays": int(len(plays)),
        "rb_player_play_rows": int(len(rb)),
        "rb_player_games": int(len(q)),
        "pbp_rb_touch_events": int(len(touch_events)),
        "presence_vs_touch_opp_share_corr": presence_touch_corr,
        "drive_presence_vs_touch_drive_share_corr": drive_touch_corr,
        "top_rb_identity_agreement": top_agreement,
        "min_core_prior3_coverage_2025": min_prior3_coverage,
        "strict_prior_leakage_pass_rate": float(lag.strict_prior_safe.mean()),
        "historical_participation_live_2026": 0,
        "pbp_touch_proxy_live_capable_after_prior_game": 1,
        "sportsbook_used": 0,
        "outcome_model_fit": 0,
    }])

    gates = pd.DataFrame([{
        **infra,
        **proxy,
        "infrastructure_pass": infra_pass,
        "proxy_evidence_pass_count": proxy_pass_count,
        "source_gate_pass": go,
        "disposition": disposition,
        "production_change": 0,
        "target_game_participation_feature_used": 0,
    }])

    manifest = pd.DataFrame([
        {
            "family": "historical_onfield_drive_rotation_truth",
            "source": "nflverse participation + PBP",
            "historical_timestamp_safe_if_lagged": 1,
            "live_2026": 0,
            "allowed_role": "benchmark delayed truth and retrospective prior-game research only",
        },
        {
            "family": "pbp_touch_drive_rotation_proxy",
            "source": "nflverse PBP rush and target identities",
            "historical_timestamp_safe_if_lagged": 1,
            "live_2026": 1,
            "allowed_role": "completed prior-game proxy candidate; production must use canonical live identity/position bridge",
        },
        {
            "family": "exact_game_day_inactive_state",
            "source": "not qualified in STACK6C",
            "historical_timestamp_safe_if_lagged": 0,
            "live_2026": 0,
            "allowed_role": "separate source audit required; target-game participation forbidden as substitute",
        },
    ])

    outputs = {
        "stack6c_rotation_source_audit.csv": source,
        "stack6c_rotation_source_gates.csv": gates,
        "stack6c_rotation_source_manifest.csv": manifest,
        "stack6c_rotation_prior3_coverage.csv": prior_cov,
        "stack6c_rotation_player_game_comparison.csv": q,
        "stack6c_rotation_player_game_lagged_proxy.csv": lag,
        "stack6c_rotation_top_rb_agreement.csv": top_table,
        "stack6c_rotation_onfield_player_drive.csv": onfield_drive,
        "stack6c_rotation_touch_player_drive.csv": touch_drive,
    }
    for name, df in outputs.items():
        df.to_csv(a.out_dir / name, index=False)

    print("=== STACK6C source audit ===")
    print(source.to_string(index=False))
    print("=== STACK6C gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6C prior3 coverage ===")
    print(prior_cov.to_string(index=False))
    print("=== STACK6C descriptive player-game correlations ===")
    print(
        q[[
            "rb_onfield_presence_share",
            "touch_opp_share",
            "drive_presence_share",
            "touch_drive_share",
            "lead_drive_share",
            "touch_lead_drive_share",
        ]].corr().to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
