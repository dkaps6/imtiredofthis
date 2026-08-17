#!/usr/bin/env python3
"""Deterministic metrics assembly for Full Slate.

One row represents one sportsbook player/market/line/book offer with current
player form, team environment, opponent defense, and optional enrichments.
All joins use canonical player/team keys and the authoritative runtime NFL week.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_season, resolve_slate_date, resolve_week
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
OUTPUTS = Path("outputs")


def _canon_name(value) -> tuple[str, str]:
    try:
        name, key = canonicalize_player_name_safe(value)
    except Exception:
        name, key = "", ""
    raw = "" if value is None else str(value).strip()
    name = (name or raw).strip()
    key = (key or "").strip() or "".join(ch.lower() for ch in name if ch.isalnum())
    return name, key


def _read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"Required input missing/empty: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Unable to read required input {path}: {exc}") from exc
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"Required input has 0 rows: {path}")
    return df


def _normalize_props(season: int, week: int) -> pd.DataFrame:
    raw = _read(OUTPUTS / "props_raw.csv", required=True).copy()
    if "player" not in raw.columns:
        for c in ("player_name", "name", "participant"):
            if c in raw.columns:
                raw["player"] = raw[c]
                break
    if "player" not in raw.columns:
        raise RuntimeError("props_raw.csv has no player column")
    if "market" not in raw.columns:
        for c in ("market_key", "key"):
            if c in raw.columns:
                raw["market"] = raw[c]
                break
    if "market" not in raw.columns or "line" not in raw.columns:
        raise RuntimeError("props_raw.csv must contain market and line")

    canon = raw["player"].map(_canon_name)
    raw["player"] = canon.map(lambda t: t[0])
    raw["player_clean_key"] = canon.map(lambda t: t[1])
    raw["season"] = int(season)
    raw["week"] = int(week)

    for col in ("team", "team_abbr", "player_team_abbr", "opponent", "opponent_abbr", "opponent_team_abbr"):
        if col in raw.columns:
            raw[col] = raw[col].map(canon_team)
    if "team_abbr" not in raw.columns:
        raw["team_abbr"] = raw.get("team", pd.Series(pd.NA, index=raw.index))
    if "team" not in raw.columns:
        raw["team"] = raw["team_abbr"]
    if "opponent_abbr" not in raw.columns:
        raw["opponent_abbr"] = raw.get("opponent", pd.Series(pd.NA, index=raw.index))
    if "opponent" not in raw.columns:
        raw["opponent"] = raw["opponent_abbr"]

    raw["line"] = pd.to_numeric(raw["line"], errors="coerce")
    raw = raw.loc[raw["line"].notna() | raw["market"].astype(str).str.contains("anytime", case=False, na=False)].copy()

    # Convert side rows into one offer row per book/line.  If raw already carries
    # wide odds, this is a no-op after the combine step.
    key_candidates = ["event_id", "player", "player_clean_key", "market", "line", "book", "book_title"]
    keys = [c for c in key_candidates if c in raw.columns]
    if "side" in raw.columns and "price_american" in raw.columns:
        side = raw[keys + ["side", "price_american"]].copy()
        side["side"] = side["side"].astype(str).str.upper().replace({"YES": "OVER", "NO": "UNDER"})
        pvt = side.pivot_table(index=keys, columns="side", values="price_american", aggfunc="first").reset_index()
        pvt.columns = ["over_odds" if c == "OVER" else "under_odds" if c == "UNDER" else c for c in pvt.columns]
        raw_base_cols = keys + [c for c in ("team", "team_abbr", "opponent", "opponent_abbr", "commence_time") if c in raw.columns]
        base = raw[raw_base_cols].drop_duplicates(keys, keep="first")
        raw = base.merge(pvt, on=keys, how="left")
    else:
        if "over_odds" not in raw.columns:
            raw["over_odds"] = np.nan
        if "under_odds" not in raw.columns:
            raw["under_odds"] = np.nan
        raw = raw.drop_duplicates(keys, keep="first")

    raw["season"] = int(season)
    raw["week"] = int(week)
    return raw.reset_index(drop=True)


def _backfill_prop_identity(props: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    out = props.copy()
    enriched = _read(DATA / "props_enriched.csv")
    if not enriched.empty:
        if "player_canonical" in enriched.columns:
            canon = enriched["player_canonical"].map(_canon_name)
        elif "player_name_raw" in enriched.columns:
            canon = enriched["player_name_raw"].map(_canon_name)
        else:
            canon = pd.Series([("", "")] * len(enriched), index=enriched.index)
        enriched["player_clean_key"] = canon.map(lambda t: t[1])
        for c in ("player_team_abbr", "opponent_team_abbr"):
            if c in enriched.columns:
                enriched[c] = enriched[c].map(canon_team)
        join = [c for c in ("event_id", "player_clean_key") if c in out.columns and c in enriched.columns]
        if join:
            cols = join + [c for c in ("player_team_abbr", "opponent_team_abbr", "kickoff_ts") if c in enriched.columns]
            e = enriched[cols].drop_duplicates(join, keep="last")
            out = out.merge(e, on=join, how="left")
            if "player_team_abbr" in out.columns:
                out["team"] = out["team"].replace("", pd.NA).combine_first(out["player_team_abbr"])
                out["team_abbr"] = out["team_abbr"].replace("", pd.NA).combine_first(out["player_team_abbr"])
            if "opponent_team_abbr" in out.columns:
                out["opponent"] = out["opponent"].replace("", pd.NA).combine_first(out["opponent_team_abbr"])
                out["opponent_abbr"] = out["opponent_abbr"].replace("", pd.NA).combine_first(out["opponent_team_abbr"])

    opp = _read(DATA / "opponent_map_from_props.csv")
    if not opp.empty:
        for c in ("season", "week"):
            if c in opp.columns:
                opp[c] = pd.to_numeric(opp[c], errors="coerce").astype("Int64")
        if "player_clean_key" not in opp.columns and "player" in opp.columns:
            opp["player_clean_key"] = opp["player"].map(_canon_name).map(lambda t: t[1])
        for c in ("team", "opponent"):
            if c in opp.columns:
                opp[c] = opp[c].map(canon_team)
        scoped = opp.loc[(opp.get("season") == int(season)) & (opp.get("week") == int(week))].copy() if {"season", "week"}.issubset(opp.columns) else opp.copy()
        join = [c for c in ("event_id", "player_clean_key") if c in out.columns and c in scoped.columns]
        if not join and "player_clean_key" in scoped.columns:
            join = ["player_clean_key"]
        if join:
            cols = join + [c for c in ("team", "opponent") if c in scoped.columns]
            right = scoped[cols].drop_duplicates(join, keep="last").rename(columns={"team": "team_map", "opponent": "opponent_map"})
            out = out.merge(right, on=join, how="left")
            if "team_map" in out.columns:
                out["team"] = out["team"].replace("", pd.NA).combine_first(out["team_map"])
                out["team_abbr"] = out["team_abbr"].replace("", pd.NA).combine_first(out["team_map"])
            if "opponent_map" in out.columns:
                out["opponent"] = out["opponent"].replace("", pd.NA).combine_first(out["opponent_map"])
                out["opponent_abbr"] = out["opponent_abbr"].replace("", pd.NA).combine_first(out["opponent_map"])

    schedule = _read(DATA / "team_week_map.csv", required=True)
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce").astype("Int64")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce").astype("Int64")
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule["opponent"].map(canon_team)
    schedule = schedule.loc[(schedule["season"] == int(season)) & (schedule["week"] == int(week)), ["team", "opponent"]].drop_duplicates("team")
    out = out.merge(schedule.rename(columns={"opponent": "opponent_schedule"}), on="team", how="left")
    out["opponent"] = out["opponent"].replace("", pd.NA).combine_first(out["opponent_schedule"])
    out["opponent_abbr"] = out["opponent_abbr"].replace("", pd.NA).combine_first(out["opponent_schedule"])
    out["team_abbr"] = out["team"]
    return out


def _join_player_form(base: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    pf = _read(DATA / "player_form_consensus.csv", required=True)
    for c in ("season", "week"):
        if c in pf.columns:
            pf[c] = pd.to_numeric(pf[c], errors="coerce").astype("Int64")
    if "player_clean_key" not in pf.columns:
        pf["player_clean_key"] = pf["player"].map(_canon_name).map(lambda t: t[1])
    pf["team"] = pf["team"].map(canon_team)
    scoped = pf.loc[(pf["season"] == int(season)) & (pf["week"] == int(week))].copy() if {"season", "week"}.issubset(pf.columns) else pf.copy()
    if scoped.empty:
        raise RuntimeError(f"player_form_consensus has no rows for season={season} week={week}")
    # Avoid duplicate identity columns from the player form; props/schedule owns
    # the current event identity.
    skip = {"player", "team", "opponent", "team_abbr", "opponent_abbr", "season", "week"}
    features = [c for c in scoped.columns if c not in skip and c != "player_clean_key"]
    right = scoped[["player_clean_key", "team", *features]].drop_duplicates(["player_clean_key", "team"], keep="last")
    merged = base.merge(right, on=["player_clean_key", "team"], how="left", indicator="_pf_merge")
    unmatched = merged.loc[merged["_pf_merge"].eq("left_only"), [c for c in ("player", "team", "market", "line") if c in merged.columns]].drop_duplicates()
    merged.drop(columns=["_pf_merge"], inplace=True)
    if not unmatched.empty:
        path = DATA / "_debug" / "metrics_unmatched_player_form.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(path, index=False)
        raise RuntimeError(f"{len(unmatched)} prop identities failed PlayerForm join; see {path}")
    return merged


def _join_team_context(base: pd.DataFrame, season: int) -> pd.DataFrame:
    tf = _read(DATA / "team_form.csv", required=True)
    tf["team"] = tf["team"].map(canon_team)
    if "season" in tf.columns:
        s = pd.to_numeric(tf["season"], errors="coerce")
        tf = tf.loc[s.eq(int(season))].copy()
    if tf.empty:
        raise RuntimeError(f"team_form has no rows for season={season}")
    tf = tf.drop_duplicates("team", keep="last")

    identity = {"team", "season", "week", "opponent"}
    off_features = [c for c in tf.columns if c not in identity]
    offense = tf[["team", *off_features]].copy()
    out = base.merge(offense, on="team", how="left")

    defense = tf[["team", *off_features]].rename(columns={"team": "opponent"})
    defense = defense.rename(columns={c: f"{c}_opp" for c in off_features})
    out = out.merge(defense, on="opponent", how="left")
    return out


def _join_optional(base: pd.DataFrame, week: int) -> pd.DataFrame:
    out = base
    injuries = _read(DATA / "injuries.csv")
    if not injuries.empty and {"player", "team", "status"}.issubset(injuries.columns):
        injuries["player_clean_key"] = injuries["player"].map(_canon_name).map(lambda t: t[1])
        injuries["team"] = injuries["team"].map(canon_team)
        if "week" in injuries.columns:
            w = pd.to_numeric(injuries["week"], errors="coerce")
            injuries = injuries.loc[w.eq(int(week))].copy()
        inj = injuries[["player_clean_key", "team", "status"]].drop_duplicates(["player_clean_key", "team"], keep="last").rename(columns={"status": "injury_status"})
        out = out.merge(inj, on=["player_clean_key", "team"], how="left")

    qb = _read(DATA / "qb_run_metrics.csv")
    if not qb.empty and {"player", "week"}.issubset(qb.columns):
        qb["player_clean_key"] = qb["player"].map(_canon_name).map(lambda t: t[1])
        w = pd.to_numeric(qb["week"], errors="coerce")
        qb = qb.loc[w.lt(int(week))].copy()
        if not qb.empty:
            qb = qb.sort_values("week").drop_duplicates("player_clean_key", keep="last")
            qcols = [c for c in ("scramble_rate", "scrambles", "dropbacks", "designed_run_rate", "designed_runs", "snaps") if c in qb.columns]
            out = out.merge(qb[["player_clean_key", *qcols]], on="player_clean_key", how="left")

    weather = _read(DATA / "weather_week.csv")
    if not weather.empty:
        home = "home_team" if "home_team" in weather.columns else "home" if "home" in weather.columns else None
        away = "away_team" if "away_team" in weather.columns else "away" if "away" in weather.columns else None
        if home and away:
            h = weather.copy(); a = weather.copy()
            h["team"] = h[home].map(canon_team); h["opponent"] = h[away].map(canon_team)
            a["team"] = a[away].map(canon_team); a["opponent"] = a[home].map(canon_team)
            wlong = pd.concat([h, a], ignore_index=True)
            ren = {}
            for src, dst in (("wind_mph_mean", "wind_mph"), ("wind_mph", "wind_mph"), ("temp_f_mean", "temp_f"), ("temp_f", "temp_f"), ("precip_prob_max", "precip"), ("precip", "precip")):
                if src in wlong.columns and dst not in ren.values():
                    ren[src] = dst
            wlong = wlong.rename(columns=ren)
            cols = [c for c in ("team", "opponent", "wind_mph", "temp_f", "precip") if c in wlong.columns]
            if {"team", "opponent"}.issubset(cols):
                out = out.merge(wlong[cols].drop_duplicates(["team", "opponent"], keep="last"), on=["team", "opponent"], how="left", suffixes=("", "_weather"))
    return out


def build(season: int, week: int) -> pd.DataFrame:
    props = _normalize_props(season, week)
    props = _backfill_prop_identity(props, season, week)
    missing_team = props["team"].isna() | props["team"].astype(str).str.strip().eq("")
    missing_opp = props["opponent"].isna() | props["opponent"].astype(str).str.strip().eq("")
    if missing_team.any() or missing_opp.any():
        dbg = props.loc[missing_team | missing_opp].copy()
        path = DATA / "_debug" / "metrics_missing_identity.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        dbg.to_csv(path, index=False)
        raise RuntimeError(f"Props identity unresolved for {len(dbg)} rows; see {path}")

    out = _join_player_form(props, season, week)
    out = _join_team_context(out, season)
    out = _join_optional(out, week)
    out["season"] = int(season)
    out["week"] = int(week)
    out["team_abbr"] = out["team"]
    out["opponent_abbr"] = out["opponent"]
    out["player_canonical"] = out["player"]
    # Compatibility aliases used by pricing/reporting.
    if "tgt_share" in out.columns and "target_share" not in out.columns:
        out["target_share"] = out["tgt_share"]
    if "yprr" in out.columns and "yprr_proxy" not in out.columns:
        out["yprr_proxy"] = out["yprr"]
    out = out.loc[:, ~out.columns.duplicated()].copy()
    if out.empty:
        raise RuntimeError("metrics_v2 produced 0 rows")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    slate = (args.date if args.date is not None else resolve_slate_date()) or ""
    week = int(args.week if args.week is not None else resolve_week(season=season, slate_date=slate))
    out = build(season, week)
    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA / "metrics_ready.csv", index=False)
    out.to_csv(DATA / "make_metrics_output.csv", index=False)
    print(f"[metrics_v2] wrote rows={len(out)} season={season} week={week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
