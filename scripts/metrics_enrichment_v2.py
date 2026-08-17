"""Optional-but-real feature joins layered onto metrics_v2 output.

Every join is left-preserving: absence of an optional enrichment leaves NaNs,
never removes a sportsbook prop row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")
OUTPUTS = Path("outputs")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[metrics_enrichment] WARN unable to read {path}: {exc}")
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _key(name) -> str:
    try:
        canonical, key = canonicalize_player_name_safe(name)
    except Exception:
        canonical, key = "", ""
    text = (canonical or ("" if name is None else str(name))).strip()
    return (key or "".join(ch.lower() for ch in text if ch.isalnum())).strip()


def _american_prob(value) -> float:
    try:
        odds = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)


def _coalesce_after_merge(df: pd.DataFrame, base_name: str, suffix: str) -> pd.DataFrame:
    alt = f"{base_name}{suffix}"
    if alt not in df.columns:
        return df
    if base_name in df.columns:
        df[base_name] = df[base_name].combine_first(df[alt])
    else:
        df[base_name] = df[alt]
    return df.drop(columns=[alt])


def _merge_team_week(base: pd.DataFrame, path: Path, *, prefix: str = "", opponent: bool = False) -> pd.DataFrame:
    src = _read(path)
    if src.empty or not {"team", "week"}.issubset(src.columns):
        return base
    src = src.copy()
    src["team"] = src["team"].map(canon_team)
    src["week"] = pd.to_numeric(src["week"], errors="coerce").astype("Int64")
    key_name = "opponent" if opponent else "team"
    value_cols = [c for c in src.columns if c not in {"team", "week", "season"}]
    renamed = {c: f"{prefix}{c}" for c in value_cols}
    src = src[["team", "week", *value_cols]].rename(columns={"team": key_name, **renamed})
    src = src.drop_duplicates([key_name, "week"], keep="last")
    return base.merge(src, on=[key_name, "week"], how="left")


def _merge_game_odds(base: pd.DataFrame) -> pd.DataFrame:
    gl = _read(OUTPUTS / "odds_game.csv")
    if gl.empty:
        return base
    for c in ("home_team", "away_team", "home", "away"):
        if c in gl.columns:
            gl[c] = gl[c].map(canon_team)
    home_col = "home_team" if "home_team" in gl.columns else "home" if "home" in gl.columns else None
    away_col = "away_team" if "away_team" in gl.columns else "away" if "away" in gl.columns else None
    if not home_col or not away_col:
        return base

    # If the fetcher already produced win probabilities, use them.
    if {"home_wp", "away_wp"}.issubset(gl.columns):
        games = gl[[c for c in ("event_id", home_col, away_col, "home_wp", "away_wp") if c in gl.columns]].drop_duplicates("event_id" if "event_id" in gl.columns else [home_col, away_col])
    else:
        games = gl[[c for c in ("event_id", home_col, away_col) if c in gl.columns]].drop_duplicates()
        games["home_wp"] = np.nan
        games["away_wp"] = np.nan
        # Common Odds API row schema: market=h2h, name=<team>, price_american.
        if {"market", "name", "price_american"}.issubset(gl.columns):
            h2h = gl.loc[gl["market"].astype(str).str.lower().eq("h2h")].copy()
            if not h2h.empty:
                h2h["name_team"] = h2h["name"].map(canon_team)
                group_keys = [c for c in ("event_id", home_col, away_col) if c in h2h.columns]
                rows = []
                for keys, grp in h2h.groupby(group_keys, dropna=False):
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    rec = dict(zip(group_keys, keys))
                    home = rec.get(home_col); away = rec.get(away_col)
                    hp = [_american_prob(v) for v in grp.loc[grp["name_team"].eq(home), "price_american"]]
                    ap = [_american_prob(v) for v in grp.loc[grp["name_team"].eq(away), "price_american"]]
                    hp = [v for v in hp if pd.notna(v)]; ap = [v for v in ap if pd.notna(v)]
                    if hp and ap:
                        ph, pa = float(np.mean(hp)), float(np.mean(ap))
                        total = ph + pa
                        rec["home_wp"] = ph / total if total > 0 else np.nan
                        rec["away_wp"] = pa / total if total > 0 else np.nan
                    rows.append(rec)
                games = pd.DataFrame(rows)

    games = games.rename(columns={home_col: "home_team", away_col: "away_team"})
    join = ["event_id"] if "event_id" in base.columns and "event_id" in games.columns else []
    if not join:
        # Build team-centric lookup when event ids are unavailable.
        long = pd.concat([
            games.assign(team=games["home_team"], opponent=games["away_team"], team_wp=games["home_wp"]),
            games.assign(team=games["away_team"], opponent=games["home_team"], team_wp=games["away_wp"]),
        ], ignore_index=True)
        keep = [c for c in ("team", "opponent", "team_wp") if c in long.columns]
        return base.merge(long[keep].drop_duplicates(["team", "opponent"], keep="last"), on=["team", "opponent"], how="left")

    merged = base.merge(games[["event_id", "home_team", "away_team", "home_wp", "away_wp"]].drop_duplicates("event_id"), on="event_id", how="left")
    merged["team_wp"] = np.where(
        merged["team"].eq(merged["home_team"]), merged["home_wp"],
        np.where(merged["team"].eq(merged["away_team"]), merged["away_wp"], np.nan),
    )
    return merged.drop(columns=["home_wp", "away_wp"], errors="ignore")


def _merge_coverage(base: pd.DataFrame) -> pd.DataFrame:
    out = base
    team_cov = _read(DATA / "cb_coverage_team.csv")
    if not team_cov.empty and "team" in team_cov.columns:
        team_cov["opponent"] = team_cov["team"].map(canon_team)
        cols = [c for c in ("opponent", "man_rate", "zone_rate") if c in team_cov.columns]
        tc = team_cov[cols].drop_duplicates("opponent", keep="last").rename(columns={
            "man_rate": "coverage_man_rate_opp", "zone_rate": "coverage_zone_rate_opp"
        })
        out = out.merge(tc, on="opponent", how="left")

    exposure = _read(DATA / "wr_cb_exposure.csv")
    if not exposure.empty and {"player", "team"}.issubset(exposure.columns):
        exposure["player_clean_key"] = exposure["player"].map(_key)
        exposure["team"] = exposure["team"].map(canon_team)
        if "opponent" in exposure.columns:
            exposure["opponent"] = exposure["opponent"].map(canon_team)
        join = ["player_clean_key", "team"] + (["opponent"] if "opponent" in exposure.columns else [])
        cols = join + [c for c in ("slot_pct", "wide_pct", "exp_vs_man", "exp_vs_zone", "primary_cb", "shadow_flag") if c in exposure.columns]
        out = out.merge(exposure[cols].drop_duplicates(join, keep="last"), on=join, how="left")

    player_cov = _read(DATA / "cb_coverage_player.csv")
    if not player_cov.empty and {"player", "team"}.issubset(player_cov.columns):
        player_cov["player_clean_key"] = player_cov["player"].map(_key)
        player_cov["team"] = player_cov["team"].map(canon_team)
        join = ["player_clean_key", "team"]
        cols = join + [c for c in ("slot_pct", "wide_pct", "primary_cb", "shadow_flag") if c in player_cov.columns]
        merged = out.merge(player_cov[cols].drop_duplicates(join, keep="last"), on=join, how="left", suffixes=("", "_coverage"))
        for c in ("slot_pct", "wide_pct", "primary_cb", "shadow_flag"):
            merged = _coalesce_after_merge(merged, c, "_coverage")
        out = merged
    return out


def enrich(base: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    out = base.copy()
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    out = _merge_game_odds(out)

    # Own-team environment.
    out = _merge_team_week(out, DATA / "play_volume_splits.csv", prefix="pbp_")
    out = _merge_team_week(out, DATA / "script_escalators.csv", prefix="script_")
    out = _merge_team_week(out, DATA / "volatility_widening.csv", prefix="volatility_")

    # Opponent-defense environment.
    out = _merge_team_week(out, DATA / "run_pass_funnel.csv", prefix="opp_funnel_", opponent=True)
    out = _merge_team_week(out, DATA / "coverage_penalties.csv", prefix="opp_penalty_", opponent=True)
    out = _merge_coverage(out)

    out["season"] = int(season)
    out["week"] = int(week)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out
