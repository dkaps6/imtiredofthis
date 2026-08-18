#!/usr/bin/env python3
"""Coverage v2: authoritative NFL matchup identity plus optional WR/CB intelligence.

NFL facts (season/week/team/opponent/game/kickoff) come from the authoritative
schedule and Ourlads. Sharp/FantasyPoints/Rotowire/Rotoballer only enrich those
facts. Provider failure never removes the underlying matchup.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.build._schedule_utils import get_nfl_schedule
from scripts.runtime_context import resolve_week
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
ROLES = DATA / "roles_ourlads.csv"
SHARP_TEAM_FORM = DATA / "sharp_team_form.csv"
TEAM_OUT = DATA / "cb_coverage_team.csv"
PLAYER_OUT = DATA / "cb_coverage_player.csv"
EXPOSURE_OUT = DATA / "wr_cb_exposure.csv"
LOG = logging.getLogger("coverage_v2")

TEAM_COLS = ["team", "season", "week", "man_rate", "zone_rate", "coverage_available", "coverage_source"]
PLAYER_COLS = [
    "player", "team", "opponent", "season", "week", "game_id", "game_timestamp",
    "slot_pct", "wide_pct", "man_rate", "zone_rate", "primary_cb", "shadow_flag",
    "wr_cb_advantage", "matchup_available", "alignment_available", "matchup_source",
    "alignment_source",
]
EXPOSURE_COLS = [
    "player", "player_pf", "team", "opponent", "week", "season", "game_timestamp",
    "slot_pct", "wide_pct", "man_rate", "zone_rate", "exp_vs_man", "exp_vs_zone",
    "primary_cb", "shadow_flag", "wr_cb_advantage", "matchup_available",
    "alignment_available", "team_coverage_available", "matchup_source",
    "alignment_source", "team_coverage_source",
]


def _safe_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        out = pd.read_csv(path)
        return out if len(out.columns) else pd.DataFrame()
    except Exception as exc:
        LOG.warning("[coverage_v2] failed reading %s: %s", path, exc)
        return pd.DataFrame()


def _canon(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(canon_team)


def _rate(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series.astype("string").str.replace("%", "", regex=False), errors="coerce")
    out.loc[out.gt(1)] = out.loc[out.gt(1)] / 100.0
    return out.clip(0, 1)


def _find(columns, names) -> str | None:
    lower = {str(c).strip().lower(): c for c in columns}
    for name in names:
        if name in lower:
            return lower[name]
    for key, original in lower.items():
        if any(name in key for name in names):
            return original
    return None


def _player_key(name: str) -> str:
    parts = str(name or "").replace("'", "").replace(".", "").split()
    return f"{parts[0][0].upper()}{parts[-1].capitalize()}" if parts else ""


def build_authoritative_team_map(season: int, week: int, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    sched = get_nfl_schedule(season) if schedule is None else schedule.copy()
    if sched is None or sched.empty:
        raise RuntimeError(f"Coverage v2 schedule is empty for season={season}")
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "home", "away", "kickoff_utc"}
    missing = required - set(sched.columns)
    if missing:
        raise RuntimeError(f"Coverage v2 schedule missing required columns: {sorted(missing)}")

    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched = sched.loc[sched["season"].eq(season) & sched["week"].eq(week)].copy()
    if sched.empty:
        raise RuntimeError(f"Coverage v2 found no games for season={season} week={week}")
    sched["home"] = _canon(sched["home"])
    sched["away"] = _canon(sched["away"])
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True, errors="coerce")
    if sched[["home", "away"]].eq("").any().any() or sched["kickoff_utc"].isna().any():
        raise RuntimeError("Coverage v2 schedule contains blank team identity or kickoff")

    if "game_id" not in sched.columns:
        sched["game_id"] = (
            sched["season"].astype(str) + "_" + sched["week"].astype(str).str.zfill(2)
            + "_" + sched["away"] + "_" + sched["home"]
        )

    rows = []
    for _, g in sched.iterrows():
        common = {"season": season, "week": week, "game_id": str(g["game_id"]), "game_timestamp": g["kickoff_utc"]}
        rows.append({**common, "team": g["home"], "opponent": g["away"]})
        rows.append({**common, "team": g["away"], "opponent": g["home"]})
    out = pd.DataFrame(rows)
    if out["team"].duplicated().any():
        raise RuntimeError(f"Coverage v2 duplicate team rows: {out.loc[out['team'].duplicated(False), 'team'].tolist()}")
    # Bye weeks are valid. The invariant is exactly two team rows per scheduled game.
    if len(out) != 2 * len(sched) or len(out) < 2 or len(out) % 2:
        raise RuntimeError(f"Coverage v2 invalid team/game grain: games={len(sched)} team_rows={len(out)}")
    return out.sort_values("team").reset_index(drop=True)


def load_wr_universe(team_map: pd.DataFrame, roles: pd.DataFrame | None = None) -> pd.DataFrame:
    roles = _safe_csv(ROLES) if roles is None else roles.copy()
    if roles.empty:
        raise RuntimeError("Coverage v2 requires non-empty Ourlads roles")
    cols = {str(c).lower(): c for c in roles.columns}
    player_col, team_col = cols.get("player"), cols.get("team")
    pos_col = cols.get("position") or cols.get("pos")
    role_col = cols.get("role")
    if not player_col or not team_col or not pos_col:
        raise RuntimeError("Coverage v2 roles missing player/team/position")
    wr = pd.DataFrame({
        "player": roles[player_col].astype("string").str.strip(),
        "team": _canon(roles[team_col]),
        "position": roles[pos_col].astype("string").str.upper().str.strip(),
        "role": roles[role_col].astype("string").str.strip() if role_col else "",
    })
    wr = wr.loc[wr["position"].str.startswith("WR", na=False) & wr["player"].notna() & wr["player"].ne("") & wr["team"].ne("")]
    wr = wr.drop_duplicates(["player", "team"])
    wr = wr.merge(team_map, on="team", how="inner", validate="many_to_one")
    if wr.empty:
        raise RuntimeError("Coverage v2 found no active-week roster WRs")
    return wr.reset_index(drop=True)


def build_team_coverage(season: int, week: int, team_map: pd.DataFrame, sharp_df: pd.DataFrame | None = None) -> pd.DataFrame:
    source = _safe_csv(SHARP_TEAM_FORM) if sharp_df is None else sharp_df.copy()
    rates = pd.DataFrame(columns=["team", "man_rate", "zone_rate"])
    provider = "unavailable"

    if not source.empty:
        team_col = _find(source.columns, ["team", "team_abbr"])
        man_col = _find(source.columns, ["coverage_man_rate", "man_rate", "man coverage"])
        zone_col = _find(source.columns, ["coverage_zone_rate", "zone_rate", "zone coverage"])
        if team_col and (man_col or zone_col):
            rates = pd.DataFrame({"team": _canon(source[team_col])})
            rates["man_rate"] = _rate(source[man_col]) if man_col else pd.NA
            rates["zone_rate"] = _rate(source[zone_col]) if zone_col else pd.NA
            rates = rates.loc[rates["team"].ne("")].drop_duplicates("team", keep="last")
            provider = "sharp_team_form"

    # Keep the old Sharp scraper as a fallback adapter; do not make it the identity source.
    if rates.empty or not rates[["man_rate", "zone_rate"]].notna().any().any():
        try:
            from scripts.build.build_cb_coverage_team import fetch_sharp_coverage
            legacy = fetch_sharp_coverage()
            if legacy is not None and not legacy.empty:
                legacy = legacy.copy()
                legacy["team"] = _canon(legacy["team"])
                legacy["man_rate"] = _rate(legacy["man_rate"])
                legacy["zone_rate"] = _rate(legacy["zone_rate"])
                rates = legacy[["team", "man_rate", "zone_rate"]].drop_duplicates("team")
                provider = "sharp_legacy_fallback"
        except Exception as exc:
            LOG.warning("[coverage_v2] legacy Sharp coverage unavailable: %s", exc)

    out = team_map[["team"]].drop_duplicates().merge(rates, on="team", how="left", validate="one_to_one")
    out["season"], out["week"] = season, week
    out["coverage_available"] = out[["man_rate", "zone_rate"]].notna().any(axis=1).astype(int)
    out["coverage_source"] = provider
    return out[TEAM_COLS].sort_values("team").reset_index(drop=True)


def _normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "player" not in df.columns:
        return df
    out = df.copy()
    names = []
    for raw in out["player"]:
        name, _ = canonicalize_player_name_safe(raw)
        names.append(name or str(raw or "").strip())
    out["player"] = names
    return out


def _fantasypoints(week: int) -> pd.DataFrame:
    try:
        from scripts.fantasypoints_wr_cb_scraper import fetch_wr_cb_html, extract_wr_cb_json, normalize_wr_cb_data
        payload = extract_wr_cb_json(fetch_wr_cb_html())
        if not payload:
            return pd.DataFrame()
        out = normalize_wr_cb_data(payload, week)
        if out is None or out.empty or "player" not in out.columns:
            return pd.DataFrame()
        out = _normalize_names(out)
        if "team" in out.columns:
            out["team"] = _canon(out["team"])
        return out
    except Exception as exc:
        LOG.warning("[coverage_v2] FantasyPoints unavailable: %s", exc)
        return pd.DataFrame()


def _rotowire() -> pd.DataFrame:
    try:
        from scripts.build.build_cb_coverage_player import fetch_rotowire_alignment
        out = fetch_rotowire_alignment()
        if out is None or out.empty:
            return pd.DataFrame()
        out = _normalize_names(out)
        out["team"] = _canon(out["team"])
        return out
    except Exception as exc:
        LOG.warning("[coverage_v2] Rotowire unavailable: %s", exc)
        return pd.DataFrame()


def _rotoballer() -> pd.DataFrame:
    try:
        from scripts.build.build_cb_coverage_player import fetch_rotoballer_notes
        out = fetch_rotoballer_notes()
        return _normalize_names(out) if out is not None and not out.empty else pd.DataFrame()
    except Exception as exc:
        LOG.warning("[coverage_v2] Rotoballer unavailable: %s", exc)
        return pd.DataFrame()


def build_player_coverage(wr_universe: pd.DataFrame, team_coverage: pd.DataFrame,
                          fantasy: pd.DataFrame | None = None, rotowire: pd.DataFrame | None = None,
                          rotoballer: pd.DataFrame | None = None) -> pd.DataFrame:
    base = wr_universe[["player", "team", "opponent", "season", "week", "game_id", "game_timestamp"]].copy()
    fp = _fantasypoints(int(base["week"].iloc[0])) if fantasy is None else fantasy.copy()
    rw = _rotowire() if rotowire is None else rotowire.copy()
    rb = _rotoballer() if rotoballer is None else rotoballer.copy()

    for frame in (fp, rw, rb):
        if not frame.empty:
            if "player" in frame.columns:
                frame["player"] = frame["player"].astype("string").str.strip()
            if "team" in frame.columns:
                frame["team"] = _canon(frame["team"])

    # FantasyPoints: direct WR-CB matchup data when available.
    base["primary_cb_fp"], base["wr_cb_advantage"], base["slot_pct_fp"] = "", pd.NA, pd.NA
    if not fp.empty and "player" in fp.columns:
        keys = ["player"] + (["team"] if "team" in fp.columns else [])
        tmp = fp[keys].copy()
        primary = _find(fp.columns, ["primary_corner", "corner", "primary_cb", "cb"])
        adv = _find(fp.columns, ["wr_cb_advantage", "advantage"])
        slot = _find(fp.columns, ["slot_rate", "slot_pct", "slot"])
        tmp["primary_cb_fp"] = fp[primary].astype("string").fillna("") if primary else ""
        tmp["wr_cb_advantage_fp"] = fp[adv] if adv else pd.NA
        tmp["slot_pct_fp_new"] = _rate(fp[slot]) if slot else pd.NA
        tmp = tmp.drop_duplicates(keys)
        base = base.drop(columns=["primary_cb_fp", "wr_cb_advantage", "slot_pct_fp"]).merge(tmp, on=keys, how="left")
        base["primary_cb_fp"] = base["primary_cb_fp"].fillna("")
        base["wr_cb_advantage"] = base["wr_cb_advantage_fp"]
        base["slot_pct_fp"] = base["slot_pct_fp_new"]
        base = base.drop(columns=["wr_cb_advantage_fp", "slot_pct_fp_new"])

    # Rotowire: alignment fallback/supplement.
    base["slot_pct_rw"], base["wide_pct_rw"] = pd.NA, pd.NA
    if not rw.empty and {"player", "team"}.issubset(rw.columns):
        tmp = rw[[c for c in ["player", "team", "slot_pct", "wide_pct"] if c in rw.columns]].drop_duplicates(["player", "team"])
        tmp = tmp.rename(columns={"slot_pct": "slot_pct_rw_new", "wide_pct": "wide_pct_rw_new"})
        base = base.drop(columns=["slot_pct_rw", "wide_pct_rw"]).merge(tmp, on=["player", "team"], how="left")
        base["slot_pct_rw"] = pd.to_numeric(base.get("slot_pct_rw_new"), errors="coerce")
        base["wide_pct_rw"] = pd.to_numeric(base.get("wide_pct_rw_new"), errors="coerce")
        base = base.drop(columns=[c for c in ["slot_pct_rw_new", "wide_pct_rw_new"] if c in base.columns])

    # Rotoballer: supplemental shadow note/CB only.
    base["primary_cb_rb"], base["shadow_flag"] = "", ""
    if not rb.empty and "player" in rb.columns:
        tmp = rb[[c for c in ["player", "primary_cb", "shadow_flag"] if c in rb.columns]].drop_duplicates("player")
        tmp = tmp.rename(columns={"primary_cb": "primary_cb_rb_new", "shadow_flag": "shadow_flag_new"})
        base = base.drop(columns=["primary_cb_rb", "shadow_flag"]).merge(tmp, on="player", how="left")
        base["primary_cb_rb"] = base.get("primary_cb_rb_new", "").fillna("")
        base["shadow_flag"] = base.get("shadow_flag_new", "").fillna("")
        base = base.drop(columns=[c for c in ["primary_cb_rb_new", "shadow_flag_new"] if c in base.columns])

    for col in ["primary_cb_fp", "primary_cb_rb", "shadow_flag"]:
        base[col] = base[col].astype("string").fillna("").str.strip()
    base["primary_cb"] = base["primary_cb_fp"].where(base["primary_cb_fp"].ne(""), base["primary_cb_rb"])
    base["slot_pct"] = pd.to_numeric(base["slot_pct_fp"], errors="coerce").combine_first(pd.to_numeric(base["slot_pct_rw"], errors="coerce"))
    base["wide_pct"] = pd.to_numeric(base["wide_pct_rw"], errors="coerce")
    missing_wide = base["wide_pct"].isna() & base["slot_pct"].notna()
    base.loc[missing_wide, "wide_pct"] = (1 - base.loc[missing_wide, "slot_pct"]).clip(0, 1)

    opp_rates = team_coverage[["team", "man_rate", "zone_rate"]].rename(columns={"team": "opponent"})
    base = base.merge(opp_rates, on="opponent", how="left", validate="many_to_one")
    base["matchup_available"] = (base["primary_cb"].ne("") | base["wr_cb_advantage"].notna()).astype(int)
    base["alignment_available"] = base[["slot_pct", "wide_pct"]].notna().any(axis=1).astype(int)
    base["matchup_source"] = ""
    base.loc[base["primary_cb_fp"].ne("") | base["wr_cb_advantage"].notna(), "matchup_source"] = "fantasypoints"
    base.loc[base["matchup_source"].eq("") & base["primary_cb_rb"].ne(""), "matchup_source"] = "rotoballer"
    base["alignment_source"] = ""
    base.loc[pd.to_numeric(base["slot_pct_fp"], errors="coerce").notna(), "alignment_source"] = "fantasypoints"
    base.loc[base["alignment_source"].eq("") & (pd.to_numeric(base["slot_pct_rw"], errors="coerce").notna() | pd.to_numeric(base["wide_pct_rw"], errors="coerce").notna()), "alignment_source"] = "rotowire"

    for col in PLAYER_COLS:
        if col not in base.columns:
            base[col] = "" if col in {"player", "team", "opponent", "game_id", "game_timestamp", "primary_cb", "shadow_flag", "matchup_source", "alignment_source"} else pd.NA
    return base[PLAYER_COLS].drop_duplicates(["player", "team"]).sort_values(["team", "player"]).reset_index(drop=True)


def build_exposure(player_cov: pd.DataFrame, team_cov: pd.DataFrame) -> pd.DataFrame:
    out = player_cov.copy()
    out["player_pf"] = out["player"].map(_player_key)
    rates = team_cov[["team", "man_rate", "zone_rate", "coverage_available", "coverage_source"]].rename(columns={
        "team": "opponent", "man_rate": "exp_vs_man", "zone_rate": "exp_vs_zone",
        "coverage_available": "team_coverage_available", "coverage_source": "team_coverage_source",
    })
    out = out.merge(rates, on="opponent", how="left", validate="many_to_one")
    out["team_coverage_available"] = pd.to_numeric(out["team_coverage_available"], errors="coerce").fillna(0).astype(int)
    out["team_coverage_source"] = out["team_coverage_source"].fillna("unavailable")
    for col in EXPOSURE_COLS:
        if col not in out.columns:
            out[col] = "" if col in {"player", "player_pf", "team", "opponent", "game_timestamp", "primary_cb", "shadow_flag", "matchup_source", "alignment_source", "team_coverage_source"} else pd.NA
    return out[EXPOSURE_COLS].drop_duplicates(["player", "team"]).sort_values(["team", "player"]).reset_index(drop=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    season = int(args.season)
    week = int(args.week) if args.week is not None else int(resolve_week())

    team_map = build_authoritative_team_map(season, week)
    wrs = load_wr_universe(team_map)
    team_cov = build_team_coverage(season, week, team_map)
    player_cov = build_player_coverage(wrs, team_cov)
    exposure = build_exposure(player_cov, team_cov)
    if exposure.empty or exposure["opponent"].fillna("").eq("").any():
        raise RuntimeError("Coverage v2 lost authoritative WR/opponent identity")

    DATA.mkdir(parents=True, exist_ok=True)
    team_cov.to_csv(TEAM_OUT, index=False)
    player_cov.to_csv(PLAYER_OUT, index=False)
    exposure.to_csv(EXPOSURE_OUT, index=False)
    print(f"[coverage_v2] season={season} week={week} scheduled_teams={len(team_map)} WRs={len(exposure)}")
    print(f"[coverage_v2] team_scheme_available={int(team_cov['coverage_available'].sum())}/{len(team_cov)} source={team_cov['coverage_source'].iloc[0] if len(team_cov) else 'unavailable'}")
    print(f"[coverage_v2] direct_WR_CB_matchups={int(player_cov['matchup_available'].sum())}/{len(player_cov)} alignments={int(player_cov['alignment_available'].sum())}/{len(player_cov)}")
    print("[coverage_v2] provider absence is explicit; schedule/Ourlads matchup identity remains authoritative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
