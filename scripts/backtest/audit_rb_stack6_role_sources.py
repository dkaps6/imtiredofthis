#!/usr/bin/env python3
"""RB STACK6 source audit: play-level secondary-back role/substitution evidence.

Diagnostic only. This audit never uses sportsbook information and never writes a
projection. It asks whether nflverse participation can be joined to PBP and
converted into player-level RB/FB on-field situational role observations that
can later be shifted strictly to prior games.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RB_POSITIONS = {"RB", "FB", "HB"}
PLAYER_FIELDS = ["offense_players", "offense_names", "offense_positions"]
PARTICIPATION_CANDIDATES = [
    "offense_players", "offense_names", "offense_positions",
    "offense_formation", "offense_personnel", "n_offense_backfield",
    "n_offense_box", "is_motion", "is_no_huddle", "is_rpo",
]
PBP_CANDIDATES = [
    "season", "week", "posteam", "down", "ydstogo", "yardline_100",
    "qtr", "half_seconds_remaining", "game_seconds_remaining",
    "drive", "series", "play_type", "rush_attempt", "rusher_player_id",
    "rusher_player_name", "score_differential", "goal_to_go",
]


def to_pd(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def lower(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def present(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        q = s.astype("string").str.strip()
        return q.notna() & q.ne("") & q.str.lower().ne("nan") & q.str.lower().ne("none")
    return pd.to_numeric(s, errors="coerce").notna()


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def derive_season_week(p: pd.DataFrame) -> pd.DataFrame:
    z = p.copy()
    gid = next((c for c in ["nflverse_game_id", "game_id"] if c in z.columns), None)
    if gid is not None and ("season" not in z.columns or "week" not in z.columns):
        parts = z[gid].astype("string").str.split("_", expand=True)
        if "season" not in z.columns and parts.shape[1] >= 1:
            z["season"] = num(parts[0])
        if "week" not in z.columns and parts.shape[1] >= 2:
            z["week"] = num(parts[1])
    return z


def join_keys(p: pd.DataFrame, b: pd.DataFrame) -> tuple[list[str], str]:
    for keys, label in [
        (["nflverse_game_id", "play_id"], "nflverse_game_id+play_id"),
        (["old_game_id", "play_id"], "old_game_id+play_id"),
        (["game_id", "play_id"], "game_id+play_id"),
    ]:
        if all(c in p.columns and c in b.columns for c in keys):
            return keys, label
    return [], "NONE"


def split_cell(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [str(v).strip() for v in value if str(v).strip() and str(v).strip().lower() not in {"nan", "none"}]
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    # Some parquet readers preserve list-looking strings; strip only the outer brackets.
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    # nflverse participation has historically used semicolon-delimited aligned arrays.
    # Keep fallbacks for pipe/comma serialization without splitting player names on spaces.
    delim = ";" if ";" in s else ("|" if "|" in s else ("," if "," in s else None))
    parts = [s] if delim is None else s.split(delim)
    out = []
    for v in parts:
        v = v.strip().strip("'\"")
        if v and v.lower() not in {"nan", "none"}:
            out.append(v)
    return out


def sample_inventory(frame: pd.DataFrame, source: str, candidates: list[str]) -> pd.DataFrame:
    rows = []
    n = len(frame)
    for c in candidates:
        if c not in frame.columns:
            rows.append({"source": source, "column": c, "exists": 0, "coverage": 0.0, "sample": ""})
            continue
        mask = present(frame[c])
        examples = frame.loc[mask, c].astype(str).head(3).tolist()
        rows.append({
            "source": source,
            "column": c,
            "exists": 1,
            "coverage": float(mask.mean()) if n else np.nan,
            "sample": " || ".join(examples)[:1000],
        })
    return pd.DataFrame(rows)


def build_joined(p: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys, label = join_keys(p, b)
    if not keys:
        raise RuntimeError("STACK6 participation/PBP have no common game+play key")
    left_keys = p[keys].dropna().drop_duplicates()
    right_keys = b[keys].dropna().drop_duplicates()
    matched = left_keys.merge(right_keys, on=keys, how="inner")
    join_rate = len(matched) / max(len(left_keys), 1)

    keep = keys + [c for c in PBP_CANDIDATES if c in b.columns and c not in keys]
    b2 = b[keep].drop_duplicates(keys)
    j = p.merge(b2, on=keys, how="inner", validate="one_to_one", suffixes=("", "_pbp"))
    # Prefer PBP season/week if participation-derived columns collide.
    for c in ["season", "week"]:
        cp = f"{c}_pbp"
        if cp in j.columns:
            j[c] = num(j[cp]).combine_first(num(j.get(c, pd.Series(np.nan, index=j.index))))
    audit = pd.DataFrame([{
        "metric": "play_join",
        "value": float(join_rate),
        "detail": f"{label}; matched={len(matched)}/{len(left_keys)}; joined_rows={len(j)}",
    }])
    return j, audit


def parse_player_play(j: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    play_rows = []
    for rec in j.to_dict("records"):
        ids = split_cell(rec.get("offense_players"))
        names = split_cell(rec.get("offense_names"))
        poss = [x.upper() for x in split_cell(rec.get("offense_positions"))]
        aligned = bool(ids and poss and len(ids) == len(poss))
        rb_idx = [i for i, p in enumerate(poss) if p in RB_POSITIONS] if aligned else []
        rb_ids = [ids[i] for i in rb_idx]
        rusher = str(rec.get("rusher_player_id") or "").strip()
        if rusher.lower() in {"nan", "none"}:
            rusher = ""
        rush = float(pd.to_numeric(pd.Series([rec.get("rush_attempt")]), errors="coerce").iloc[0] or 0) == 1
        play_rows.append({
            "season": rec.get("season"), "week": rec.get("week"),
            "aligned_arrays": int(aligned), "offense_player_count": len(ids),
            "offense_position_count": len(poss), "rb_fb_count": len(rb_idx),
            "rush_attempt": int(rush), "rusher_present": int(bool(rusher)),
            "rusher_in_offense_players": int(bool(rusher) and rusher in ids),
            "rusher_in_rb_fb": int(bool(rusher) and rusher in rb_ids),
        })
        if not aligned:
            continue
        down = pd.to_numeric(pd.Series([rec.get("down")]), errors="coerce").iloc[0]
        ytg = pd.to_numeric(pd.Series([rec.get("ydstogo")]), errors="coerce").iloc[0]
        yl = pd.to_numeric(pd.Series([rec.get("yardline_100")]), errors="coerce").iloc[0]
        half = pd.to_numeric(pd.Series([rec.get("half_seconds_remaining")]), errors="coerce").iloc[0]
        qtr = pd.to_numeric(pd.Series([rec.get("qtr")]), errors="coerce").iloc[0]
        sd = pd.to_numeric(pd.Series([rec.get("score_differential")]), errors="coerce").iloc[0]
        for i in rb_idx:
            pid = ids[i]
            pname = names[i] if i < len(names) else ""
            rows.append({
                "season": int(rec.get("season")) if pd.notna(rec.get("season")) else np.nan,
                "week": int(rec.get("week")) if pd.notna(rec.get("week")) else np.nan,
                "team": rec.get("posteam"),
                "player_id": pid,
                "player_name": pname,
                "position": poss[i],
                "drive": rec.get("drive"),
                "down": down, "ydstogo": ytg, "yardline_100": yl,
                "qtr": qtr, "half_seconds_remaining": half,
                "score_differential": sd,
                "rb_fb_count": len(rb_idx),
                "snap": 1,
                "early_down": int(pd.notna(down) and down in [1, 2]),
                "third_down": int(pd.notna(down) and down == 3),
                "fourth_down": int(pd.notna(down) and down == 4),
                "two_minute": int(pd.notna(half) and half <= 120),
                "short_yardage": int(pd.notna(ytg) and ytg <= 2 and pd.notna(down)),
                "red_zone": int(pd.notna(yl) and yl <= 20),
                "inside_10": int(pd.notna(yl) and yl <= 10),
                "inside_5": int(pd.notna(yl) and yl <= 5),
                "neutral_score": int(pd.notna(sd) and abs(sd) <= 7),
                "single_back_on_field": int(len(rb_idx) == 1),
                "multi_back_on_field": int(len(rb_idx) >= 2),
                "is_rusher": int(bool(rusher) and pid == rusher),
            })
    return pd.DataFrame(rows), pd.DataFrame(play_rows)


def role_game_table(pp: pd.DataFrame) -> pd.DataFrame:
    if pp.empty:
        return pd.DataFrame()
    keys = ["season", "week", "team", "player_id", "player_name", "position"]
    sum_cols = [
        "snap", "early_down", "third_down", "fourth_down", "two_minute",
        "short_yardage", "red_zone", "inside_10", "inside_5", "neutral_score",
        "single_back_on_field", "multi_back_on_field", "is_rusher",
    ]
    g = pp.groupby(keys, dropna=False)[sum_cols].sum().reset_index()
    drives = pp.groupby(keys, dropna=False)["drive"].nunique(dropna=True).reset_index(name="drives_seen")
    g = g.merge(drives, on=keys, how="left")
    denom = g["snap"].replace(0, np.nan)
    for c in sum_cols[1:]:
        g[f"{c}_snap_rate"] = g[c] / denom
    g["rush_per_snap"] = g["is_rusher"] / denom
    return g


def by_season_audit(j: pd.DataFrame, pp: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seasons = sorted(num(j["season"]).dropna().astype(int).unique()) if "season" in j.columns else []
    for s in seasons:
        pj = j.loc[num(j["season"]).eq(s)]
        pl = plays.loc[num(plays["season"]).eq(s)] if not plays.empty else pd.DataFrame()
        rb = pp.loc[num(pp["season"]).eq(s)] if not pp.empty else pd.DataFrame()
        rush_plays = pl.loc[pl["rush_attempt"].eq(1) & pl["rusher_present"].eq(1)] if not pl.empty else pd.DataFrame()
        rows.append({
            "season": s,
            "participation_pbp_rows": len(pj),
            "aligned_array_rate": float(pl["aligned_arrays"].mean()) if len(pl) else np.nan,
            "plays_with_rb_fb_rate": float(pl["rb_fb_count"].gt(0).mean()) if len(pl) else np.nan,
            "rusher_in_offense_players_rate": float(rush_plays["rusher_in_offense_players"].mean()) if len(rush_plays) else np.nan,
            "rusher_in_rb_fb_rate_all_rushes": float(rush_plays["rusher_in_rb_fb"].mean()) if len(rush_plays) else np.nan,
            "rb_fb_player_play_rows": len(rb),
            "rb_fb_player_games": int(rb[["week", "team", "player_id"]].drop_duplicates().shape[0]) if len(rb) else 0,
            "drive_coverage": float(present(pj["drive"]).mean()) if "drive" in pj.columns and len(pj) else 0.0,
            "down_coverage": float(num(pj["down"]).notna().mean()) if "down" in pj.columns and len(pj) else 0.0,
            "ydstogo_coverage": float(num(pj["ydstogo"]).notna().mean()) if "ydstogo" in pj.columns and len(pj) else 0.0,
            "yardline_coverage": float(num(pj["yardline_100"]).notna().mean()) if "yardline_100" in pj.columns and len(pj) else 0.0,
            "clock_coverage": float(num(pj["half_seconds_remaining"]).notna().mean()) if "half_seconds_remaining" in pj.columns and len(pj) else 0.0,
        })
    return pd.DataFrame(rows)


def feature_feasibility(role: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("overall_snap_role", "snap"), ("early_down_role", "early_down_snap_rate"),
        ("third_down_role", "third_down_snap_rate"), ("two_minute_role", "two_minute_snap_rate"),
        ("short_yardage_role", "short_yardage_snap_rate"), ("red_zone_role", "red_zone_snap_rate"),
        ("inside_10_role", "inside_10_snap_rate"), ("inside_5_role", "inside_5_snap_rate"),
        ("single_back_role", "single_back_on_field_snap_rate"), ("multi_back_role", "multi_back_on_field_snap_rate"),
        ("ballcarrier_given_snap", "rush_per_snap"), ("drive_participation", "drives_seen"),
    ]
    rows = []
    for name, col in specs:
        if col not in role.columns:
            rows.append({"feature_family": name, "column": col, "coverage": 0.0, "status": "NO_GO"})
            continue
        cov = float(num(role[col]).notna().mean()) if len(role) else 0.0
        rows.append({"feature_family": name, "column": col, "coverage": cov,
                     "status": "GO_PRIOR_GAME_SHIFT_ONLY" if cov >= 0.90 else "PARTIAL"})
    rows += [
        {"feature_family": "target_week_active_inactive_from_participation", "column": "target_week_participation", "coverage": np.nan,
         "status": "PROHIBITED_POSTGAME_LEAKAGE_USE_PREGAME_SOURCE"},
        {"feature_family": "offensive_line_availability", "column": "pregame_ol_availability", "coverage": np.nan,
         "status": "REQUIRES_SEPARATE_TIMESTAMP_SAFE_SOURCE"},
    ]
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    seasons = [int(x) for x in a.seasons.split(",") if x.strip()]

    import nflreadpy as nfl
    p = derive_season_week(lower(to_pd(nfl.load_participation(seasons=seasons))))
    b = lower(to_pd(nfl.load_pbp(seasons=seasons)))

    inv = pd.concat([
        sample_inventory(p, "participation", PARTICIPATION_CANDIDATES),
        sample_inventory(b, "pbp", PBP_CANDIDATES),
    ], ignore_index=True)
    j, join_audit = build_joined(p, b)
    pp, plays = parse_player_play(j)
    role = role_game_table(pp)
    season = by_season_audit(j, pp, plays)
    feas = feature_feasibility(role)

    join_rate = float(join_audit.iloc[0]["value"])
    aligned = float(plays["aligned_arrays"].mean()) if len(plays) else 0.0
    rusher_presence = float(plays.loc[plays["rush_attempt"].eq(1) & plays["rusher_present"].eq(1), "rusher_in_offense_players"].mean()) if len(plays) else 0.0
    descriptors_ok = bool((season[["down_coverage", "ydstogo_coverage", "yardline_coverage", "clock_coverage"]].min().min() >= 0.90)) if len(season) else False
    go = join_rate >= 0.95 and aligned >= 0.90 and rusher_presence >= 0.95 and descriptors_ok and len(role) > 0
    disposition = pd.DataFrame([{
        "disposition": "GO_STACK6_PRIOR_GAME_SITUATIONAL_ROLE_BUILD" if go else "STACK6_SOURCE_PARTIAL_OR_NO_GO",
        "play_join_rate": join_rate,
        "aligned_array_rate": aligned,
        "rusher_in_offense_players_rate": rusher_presence,
        "descriptors_ok": int(descriptors_ok),
        "rb_fb_player_games": int(len(role)),
        "sportsbook_upstream": 0,
        "target_game_participation_upstream": 0,
        "next": "BUILD_STRICTLY_SHIFTED_ROLE_FEATURES_AND_ALLOCATION_ABLATION" if go else "PRESERVE_AVAILABLE_FAMILIES_AND_AUDIT_MISSING_SOURCE",
    }])

    inv.to_csv(a.out_dir / "stack6_source_inventory.csv", index=False)
    join_audit.to_csv(a.out_dir / "stack6_join_audit.csv", index=False)
    season.to_csv(a.out_dir / "stack6_source_by_season.csv", index=False)
    feas.to_csv(a.out_dir / "stack6_feature_feasibility.csv", index=False)
    role.to_csv(a.out_dir / "stack6_target_game_role_observations_DIAGNOSTIC_ONLY.csv", index=False)
    plays.head(500).to_csv(a.out_dir / "stack6_parsing_examples.csv", index=False)
    disposition.to_csv(a.out_dir / "stack6_source_disposition.csv", index=False)

    print("=== STACK6 SOURCE DISPOSITION ==="); print(disposition.to_string(index=False))
    print("\n=== BY SEASON ==="); print(season.to_string(index=False))
    print("\n=== FEATURE FEASIBILITY ==="); print(feas.to_string(index=False))
    print("\n=== INVENTORY ==="); print(inv.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
