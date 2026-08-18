#!/usr/bin/env python3
"""Coverage v2: authoritative NFL matchup identity + optional WR/CB intelligence.

The core contract is intentionally split into two layers:

1. NFL identity (season/week/team/opponent/game/kickoff) comes only from the
   authoritative schedule/runtime context and Ourlads roster data.
2. Coverage intelligence (man/zone tendencies, WR alignment, primary CB/shadow
   notes) is enrichment. Third-party provider failure must never erase the NFL
   matchup itself.

Outputs:
- data/cb_coverage_team.csv
- data/cb_coverage_player.csv
- data/wr_cb_exposure.csv

Provider hierarchy:
- Team coverage: existing Sharp team-form artifact first, legacy Sharp scraper
  only as fallback.
- Player matchup/alignment: FantasyPoints direct WR-CB data first, Rotowire
  alignment second, Rotoballer notes as supplemental context.

When provider data is unavailable (common in preseason), schema-valid player and
exposure rows are still emitted for active roster WRs with explicit availability
flags and blank enrichment fields. No matchup data is fabricated.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.build._schedule_utils import get_nfl_schedule
from scripts.runtime_context import resolve_week
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
TEAM_OUT = DATA / "cb_coverage_team.csv"
PLAYER_OUT = DATA / "cb_coverage_player.csv"
EXPOSURE_OUT = DATA / "wr_cb_exposure.csv"
ROLES = DATA / "roles_ourlads.csv"
SHARP_TEAM_FORM = DATA / "sharp_team_form.csv"

TEAM_COLS = [
    "team",
    "season",
    "week",
    "man_rate",
    "zone_rate",
    "coverage_available",
    "coverage_source",
]

PLAYER_COLS = [
    "player",
    "team",
    "opponent",
    "season",
    "week",
    "game_id",
    "game_timestamp",
    "slot_pct",
    "wide_pct",
    "man_rate",
    "zone_rate",
    "primary_cb",
    "shadow_flag",
    "wr_cb_advantage",
    "matchup_available",
    "alignment_available",
    "matchup_source",
    "alignment_source",
]

EXPOSURE_COLS = [
    "player",
    "player_pf",
    "team",
    "opponent",
    "week",
    "season",
    "game_timestamp",
    "slot_pct",
    "wide_pct",
    "man_rate",
    "zone_rate",
    "exp_vs_man",
    "exp_vs_zone",
    "primary_cb",
    "shadow_flag",
    "wr_cb_advantage",
    "matchup_available",
    "alignment_available",
    "team_coverage_available",
    "matchup_source",
    "alignment_source",
    "team_coverage_source",
]

LOG = logging.getLogger("coverage_v2")


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOG.warning("[coverage_v2] failed reading %s: %s", path, exc)
        return pd.DataFrame()
    return df if len(df.columns) else pd.DataFrame()


def _numeric_rate(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.replace("%", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    # Accept either decimal rates (0-1) or percentages (0-100).
    mask = out.gt(1.0)
    out.loc[mask] = out.loc[mask] / 100.0
    return out.clip(lower=0.0, upper=1.0)


def _canon_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(canon_team)


def _player_key(name: str) -> str:
    text = "" if name is None else str(name).strip()
    if not text:
        return ""
    parts = text.replace("'", "").replace(".", "").split()
    if not parts:
        return ""
    return f"{parts[0][0].upper()}{parts[-1].capitalize()}"


def build_authoritative_team_map(season: int, week: int, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    sched = get_nfl_schedule(int(season)) if schedule is None else schedule.copy()
    if sched is None or sched.empty:
        raise RuntimeError(f"Coverage v2 schedule is empty for season={season}")

    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "home", "away", "kickoff_utc"}
    missing = required - set(sched.columns)
    if missing:
        raise RuntimeError(f"Coverage v2 schedule missing required columns: {sorted(missing)}")

    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched = sched.loc[sched["season"].eq(int(season)) & sched["week"].eq(int(week))].copy()
    if sched.empty:
        raise RuntimeError(f"Coverage v2 found no authoritative games for season={season} week={week}")

    sched["home"] = _canon_series(sched["home"])
    sched["away"] = _canon_series(sched["away"])
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True, errors="coerce")
    if sched[["home", "away"]].eq("").any().any() or sched["kickoff_utc"].isna().any():
        raise RuntimeError("Coverage v2 authoritative schedule contains blank team identity or kickoff")

    if "game_id" not in sched.columns:
        sched["game_id"] = (
            sched["season"].astype(str)
            + "_"
            + sched["week"].astype(str).str.zfill(2)
            + "_"
            + sched["away"].astype(str)
            + "_"
            + sched["home"].astype(str)
        )

    rows: list[dict] = []
    for _, g in sched.iterrows():
        gid = str(g.get("game_id", "")).strip()
        ts = g["kickoff_utc"]
        rows.append({"team": g["home"], "opponent": g["away"], "season": int(season), "week": int(week), "game_id": gid, "game_timestamp": ts})
        rows.append({"team": g["away"], "opponent": g["home"], "season": int(season), "week": int(week), "game_id": gid, "game_timestamp": ts})

    out = pd.DataFrame(rows)
    if out["team"].duplicated().any():
        dupes = out.loc[out["team"].duplicated(False), "team"].tolist()
        raise RuntimeError(f"Coverage v2 authoritative week has duplicate team rows: {dupes}")
    if len(out) != 32:
        raise RuntimeError(f"Coverage v2 expected 32 team rows for an NFL week; found {len(out)}")
    return out.sort_values("team").reset_index(drop=True)


def load_wr_universe(team_map: pd.DataFrame, roles: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _safe_csv(ROLES) if roles is None else roles.copy()
    if df.empty:
        raise RuntimeError("Coverage v2 requires non-empty Ourlads roles")

    cols = {str(c).lower(): c for c in df.columns}
    player_col = cols.get("player")
    team_col = cols.get("team")
    pos_col = cols.get("position") or cols.get("pos")
    role_col = cols.get("role")
    if not player_col or not team_col or not pos_col:
        raise RuntimeError("Coverage v2 roles artifact missing player/team/position")

    working = pd.DataFrame({
        "player": df[player_col].astype("string").str.strip(),
        "team": _canon_series(df[team_col]),
        "position": df[pos_col].astype("string").str.upper().str.strip(),
        "role": df[role_col].astype("string").str.strip() if role_col else "",
    })
    # Coverage is currently WR-focused; TE/RB matchup models can be separate.
    working = working.loc[working["position"].str.startswith("WR", na=False)].copy()
    working = working.loc[working["player"].notna() & working["player"].ne("") & working["team"].ne("")].copy()
    working = working.drop_duplicates(["player", "team"])
    if working.empty:
        raise RuntimeError("Coverage v2 found no WR rows in Ourlads roles")

    working = working.merge(team_map, on="team", how="inner", validate="many_to_one")
    if working.empty:
        raise RuntimeError("Coverage v2 WR universe could not join to authoritative week")
    return working.reset_index(drop=True)


def _find_first(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in cols}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for raw_lower, original in normalized.items():
        if any(candidate in raw_lower for candidate in candidates):
            return original
    return None


def build_team_coverage(season: int, week: int, team_map: pd.DataFrame, sharp_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per NFL team, preferring the already-fetched Sharp artifact."""
    source = _safe_csv(SHARP_TEAM_FORM) if sharp_df is None else sharp_df.copy()
    rates = pd.DataFrame(columns=["team", "man_rate", "zone_rate"])
    provider = "unavailable"

    if not source.empty:
        team_col = _find_first(source.columns, ["team", "team_abbr"])
        man_col = _find_first(source.columns, ["coverage_man_rate", "man_rate", "man coverage"])
        zone_col = _find_first(source.columns, ["coverage_zone_rate", "zone_rate", "zone coverage"])
        if team_col and (man_col or zone_col):
            rates = pd.DataFrame({"team": _canon_series(source[team_col])})
            rates["man_rate"] = _numeric_rate(source[man_col]) if man_col else pd.NA
            rates["zone_rate"] = _numeric_rate(source[zone_col]) if zone_col else pd.NA
            rates = rates.loc[rates["team"].ne("")].drop_duplicates("team", keep="last")
            provider = "sharp_team_form"

    # Preserve the legacy Sharp scraper as a fallback adapter rather than deleting it.
    if rates.empty or (rates[["man_rate", "zone_rate"]].notna().sum().sum() == 0):
        try:
            from scripts.build.build_cb_coverage_team import fetch_sharp_coverage
            legacy = fetch_sharp_coverage()
            if legacy is not None and not legacy.empty:
                legacy = legacy.copy()
                legacy["team"] = _canon_series(legacy["team"])
                legacy["man_rate"] = _numeric_rate(legacy["man_rate"])
                legacy["zone_rate"] = _numeric_rate(legacy["zone_rate"])
                rates = legacy[["team", "man_rate", "zone_rate"]].drop_duplicates("team")
                provider = "sharp_legacy_fallback"
        except Exception as exc:
            LOG.warning("[coverage_v2] legacy Sharp coverage unavailable: %s", exc)

    out = team_map[["team"]].drop_duplicates().copy()
    out = out.merge(rates, on="team", how="left", validate="one_to_one")
    out["season"] = int(season)
    out["week"] = int(week)
    out["coverage_available"] = out[["man_rate", "zone_rate"]].notna().any(axis=1).astype(int)
    out["coverage_source"] = provider
    return out[TEAM_COLS].sort_values("team").reset_index(drop=True)


def _normalize_provider_player_names(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "player" not in df.columns:
        return df
    x = df.copy()
    normalized: list[str] = []
    for raw in x["player"].tolist():
        name, _ = canonicalize_player_name_safe(raw)
        normalized.append(name or str(raw or "").strip())
    x["player"] = normalized
    return x


def _load_fantasypoints(week: int) -> pd.DataFrame:
    try:
        from scripts.fantasypoints_wr_cb_scraper import fetch_wr_cb_html, extract_wr_cb_json, normalize_wr_cb_data
        html = fetch_wr_cb_html()
        payload = extract_wr_cb_json(html)
        if not payload:
            return pd.DataFrame()
        df = normalize_wr_cb_data(payload, int(week))
        if df is None or df.empty or "player" not in df.columns:
            return pd.DataFrame()
        df = _normalize_provider_player_names(df)
        if "team" in df.columns:
            df["team"] = _canon_series(df["team"])
        return df
    except Exception as exc:
        LOG.warning("[coverage_v2] FantasyPoints matchup unavailable: %s", exc)
        return pd.DataFrame()


def _load_rotowire() -> pd.DataFrame:
    try:
        from scripts.build.build_cb_coverage_player import fetch_rotowire_alignment
        df = fetch_rotowire_alignment()
        if df is None or df.empty:
            return pd.DataFrame()
        df = _normalize_provider_player_names(df)
        df["team"] = _canon_series(df["team"])
        return df
    except Exception as exc:
        LOG.warning("[coverage_v2] Rotowire alignment unavailable: %s", exc)
        return pd.DataFrame()


def _load_rotoballer_notes() -> pd.DataFrame:
    try:
        from scripts.build.build_cb_coverage_player import fetch_rotoballer_notes
        df = fetch_rotoballer_notes()
        if df is None or df.empty:
            return pd.DataFrame()
        return _normalize_provider_player_names(df)
    except Exception as exc:
        LOG.warning("[coverage_v2] Rotoballer notes unavailable: %s", exc)
        return pd.DataFrame()


def build_player_coverage(
    wr_universe: pd.DataFrame,
    team_coverage: pd.DataFrame,
    fantasy: pd.DataFrame | None = None,
    rotowire: pd.DataFrame | None = None,
    rotoballer: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach optional provider intelligence to every authoritative roster WR."""
    base = wr_universe[["player", "team", "opponent", "season", "week", "game_id", "game_timestamp"]].copy()

    fp = _load_fantasypoints(int(base["week"].iloc[0])) if fantasy is None else fantasy.copy()
    rw = _load_rotowire() if rotowire is None else rotowire.copy()
    rb = _load_rotoballer_notes() if rotoballer is None else rotoballer.copy()

    for frame in (fp, rw, rb):
        if not frame.empty:
            if "player" in frame.columns:
                frame["player"] = frame["player"].astype("string").str.strip()
            if "team" in frame.columns:
                frame["team"] = _canon_series(frame["team"])

    # FantasyPoints direct matchup intelligence.
    fp_keep = pd.DataFrame()
    if not fp.empty and "player" in fp.columns:
        fp_keep = pd.DataFrame({"player": fp["player"]})
        if "team" in fp.columns:
            fp_keep["team"] = fp["team"]
        primary_col = _find_first(fp.columns, ["primary_corner", "corner", "primary_cb", "cb"])
        advantage_col = _find_first(fp.columns, ["wr_cb_advantage", "advantage"])
        slot_col = _find_first(fp.columns, ["slot_rate", "slot_pct", "slot"])
        fp_keep["primary_cb_fp"] = fp[primary_col].astype("string").fillna("") if primary_col else ""
        fp_keep["wr_cb_advantage"] = fp[advantage_col] if advantage_col else pd.NA
        fp_keep["slot_pct_fp"] = _numeric_rate(fp[slot_col]) if slot_col else pd.NA
        fp_keep = fp_keep.drop_duplicates([c for c in ["player", "team"] if c in fp_keep.columns])
        merge_keys = ["player", "team"] if "team" in fp_keep.columns else ["player"]
        base = base.merge(fp_keep, on=merge_keys, how="left")
    else:
        base["primary_cb_fp"] = ""
        base["wr_cb_advantage"] = pd.NA
        base["slot_pct_fp"] = pd.NA

    # Rotowire alignment is a fallback/supplement for slot/outside rates.
    if not rw.empty and {"player", "team"}.issubset(rw.columns):
        rw_keep = rw[[c for c in ["player", "team", "slot_pct", "wide_pct"] if c in rw.columns]].drop_duplicates(["player", "team"])
        rw_keep = rw_keep.rename(columns={"slot_pct": "slot_pct_rw", "wide_pct": "wide_pct_rw"})
        base = base.merge(rw_keep, on=["player", "team"], how="left")
    else:
        base["slot_pct_rw"] = pd.NA
        base["wide_pct_rw"] = pd.NA

    # Rotoballer is supplemental shadow/CB context only.
    if not rb.empty and "player" in rb.columns:
        rb_keep = rb[[c for c in ["player", "primary_cb", "shadow_flag"] if c in rb.columns]].drop_duplicates("player")
        rb_keep = rb_keep.rename(columns={"primary_cb": "primary_cb_rb"})
        base = base.merge(rb_keep, on="player", how="left")
    else:
        base["primary_cb_rb"] = ""
        base["shadow_flag"] = ""

    for c in ["primary_cb_fp", "primary_cb_rb", "shadow_flag"]:
        if c not in base.columns:
            base[c] = ""
        base[c] = base[c].astype("string").fillna("").str.strip()

    base["primary_cb"] = base["primary_cb_fp"].where(base["primary_cb_fp"].ne(""), base["primary_cb_rb"])
    base["slot_pct"] = pd.to_numeric(base.get("slot_pct_fp"), errors="coerce").combine_first(
        pd.to_numeric(base.get("slot_pct_rw"), errors="coerce")
    )
    base["wide_pct"] = pd.to_numeric(base.get("wide_pct_rw"), errors="coerce")
    # If only slot rate is known, outside share can be deterministically derived.
    derive_wide = base["wide_pct"].isna() & base["slot_pct"].notna()
    base.loc[derive_wide, "wide_pct"] = (1.0 - base.loc[derive_wide, "slot_pct"]).clip(0.0, 1.0)

    team_rates = team_coverage[["team", "man_rate", "zone_rate"]].rename(columns={"team": "opponent"})
    base = base.merge(team_rates, on="opponent", how="left", validate="many_to_one")

    base["matchup_available"] = (
        base["primary_cb"].ne("") | base["wr_cb_advantage"].notna()
    ).astype(int)
    base["alignment_available"] = base[["slot_pct", "wide_pct"]].notna().any(axis=1).astype(int)
    base["matchup_source"] = ""
    base.loc[base["primary_cb_fp"].ne("") | base["wr_cb_advantage"].notna(), "matchup_source"] = "fantasypoints"
    rb_only = base["matchup_source"].eq("") & base["primary_cb_rb"].ne("")
    base.loc[rb_only, "matchup_source"] = "rotoballer"
    base["alignment_source"] = ""
    base.loc[base["slot_pct_fp"].notna(), "alignment_source"] = "fantasypoints"
    rw_only = base["alignment_source"].eq("") & (base["slot_pct_rw"].notna() | base["wide_pct_rw"].notna())
    base.loc[rw_only, "alignment_source"] = "rotowire"

    for col in PLAYER_COLS:
        if col not in base.columns:
            base[col] = "" if col in {"player", "team", "opponent", "game_id", "game_timestamp", "primary_cb", "shadow_flag", "matchup_source", "alignment_source"} else pd.NA
    return base[PLAYER_COLS].drop_duplicates(["player", "team"]).sort_values(["team", "player"]).reset_index(drop=True)


def build_exposure(player_cov: pd.DataFrame, team_cov: pd.DataFrame) -> pd.DataFrame:
    out = player_cov.copy()
    out["player_pf"] = out["player"].map(_player_key)

    opponent_rates = team_cov[["team", "man_rate", "zone_rate", "coverage_available", "coverage_source"]].rename(
        columns={
            "team": "opponent",
            "man_rate": "exp_vs_man",
            "zone_rate": "exp_vs_zone",
            "coverage_available": "team_coverage_available",
            "coverage_source": "team_coverage_source",
        }
    )
    out = out.drop(columns=[c for c in ["exp_vs_man", "exp_vs_zone", "team_coverage_available", "team_coverage_source"] if c in out.columns])
    out = out.merge(opponent_rates, on="opponent", how="left", validate="many_to_one")
    out["team_coverage_available"] = pd.to_numeric(out["team_coverage_available"], errors="coerce").fillna(0).astype(int)
    out["team_coverage_source"] = out["team_coverage_source"].fillna("unavailable")

    for col in EXPOSURE_COLS:
        if col not in out.columns:
            out[col] = "" if col in {"player", "player_pf", "team", "opponent", "game_timestamp", "primary_cb", "shadow_flag", "matchup_source", "alignment_source", "team_coverage_source"} else pd.NA
    return out[EXPOSURE_COLS].drop_duplicates(["player", "team"]).sort_values(["team", "player"]).reset_index(drop=True)


def write_outputs(team_cov: pd.DataFrame, player_cov: pd.DataFrame, exposure: pd.DataFrame) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    team_cov.to_csv(TEAM_OUT, index=False)
    player_cov.to_csv(PLAYER_OUT, index=False)
    exposure.to_csv(EXPOSURE_OUT, index=False)

    print(
        f"[coverage_v2] team coverage rows={len(team_cov)} available={int(team_cov['coverage_available'].sum())} -> {TEAM_OUT}"
    )
    print(
        f"[coverage_v2] player coverage rows={len(player_cov)} matchup_available={int(player_cov['matchup_available'].sum())} "
        f"alignment_available={int(player_cov['alignment_available'].sum())} -> {PLAYER_OUT}"
    )
    print(
        f"[coverage_v2] exposure rows={len(exposure)} team_coverage_available={int(exposure['team_coverage_available'].sum())} -> {EXPOSURE_OUT}"
    )


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

    if exposure.empty:
        raise RuntimeError("Coverage v2 produced no WR exposure rows from a non-empty authoritative WR universe")
    if exposure["opponent"].fillna("").eq("").any():
        raise RuntimeError("Coverage v2 produced WR rows with unresolved opponents")

    write_outputs(team_cov, player_cov, exposure)
    print(
        f"[coverage_v2] season={season} week={week} authoritative WRs={len(wrs)}; "
        "provider absence is represented explicitly and does not remove NFL matchup identity"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
