#!/usr/bin/env python3
"""Build strict-as-of RB receiving-efficiency features for frozen R27B V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import evaluate_rb_r26_vacancy_gated_r9_v1 as r26
from scripts.backtest import evaluate_rb_r27_receiving_yard_mean_decomposition_v1 as r27
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.pbp import get_pbp

RB_POS = {"RB", "FB", "HB", "TB"}
EPS = 1e-12


def _key(v) -> str:
    try:
        _, k = canonicalize_player_name_safe(v)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _before(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    ss = pd.to_numeric(df["season"], errors="coerce")
    ww = pd.to_numeric(df["week"], errors="coerce")
    return df.loc[(ss < season) | ((ss == season) & (ww < week))].copy()


def _eb(raw: float, n: float, prior: float, k: float) -> float:
    if not np.isfinite(prior):
        return np.nan
    if not np.isfinite(raw) or not np.isfinite(n) or n <= 0:
        return float(prior)
    return float((n * raw + k * prior) / (n + k))


def _rate(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan


def _dedup_logs(hist_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for d in hist_dirs:
        p = d / "player_game_logs_history.csv"
        if p.exists() and p.stat().st_size:
            x = pd.read_csv(p, low_memory=False)
            x.columns = [str(c).lower() for c in x.columns]
            frames.append(x)
    if not frames:
        raise RuntimeError("R27B found no historical player logs")
    x = pd.concat(frames, ignore_index=True, sort=False)
    for c in ("season", "week", "targets", "receptions", "rec_yards", "pass_att"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x.get("opponent", "").map(canon_team)
    x["position"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip()
    x["player_clean_key"] = x.get("player_clean_key", x.get("player", "").map(_key)).fillna("").astype(str)
    keys = [c for c in ("season", "week", "team", "player_identity_key") if c in x.columns]
    if len(keys) < 4:
        keys = ["season", "week", "team", "player_clean_key"]
    x = x.sort_values(["season", "week"]).drop_duplicates(keys, keep="last")
    return x.reset_index(drop=True)


def _pbp_receiver_history(seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    frames = []
    audit = {"seasons_requested": seasons, "seasons_loaded": [], "rows": 0}
    for season in seasons:
        try:
            p = get_pbp(int(season), min_rows=1).copy()
            p.columns = [str(c).lower() for c in p.columns]
        except Exception as exc:
            audit[f"season_{season}_error"] = str(exc)
            continue
        if "season_type" in p.columns:
            reg = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                p = reg
        name_col = next((c for c in ("receiver_player_name", "receiver_name", "receiver") if c in p.columns), None)
        if name_col is None or "week" not in p.columns:
            audit[f"season_{season}_error"] = "receiver/week columns unavailable"
            continue
        p["season"] = int(season)
        p["week"] = pd.to_numeric(p["week"], errors="coerce")
        p["team"] = p.get("posteam", "").map(canon_team)
        p["opponent"] = p.get("defteam", "").map(canon_team)
        p["player_clean_key"] = p[name_col].fillna("").astype(str).map(_key)
        p = p.loc[p["player_clean_key"].ne("") & p["week"].notna()].copy()
        p["complete"] = pd.to_numeric(p.get("complete_pass", 0), errors="coerce").fillna(0).eq(1).astype(int)
        p["yards"] = pd.to_numeric(p.get("yards_gained", 0), errors="coerce").fillna(0.0)
        p["air"] = pd.to_numeric(p.get("air_yards", np.nan), errors="coerce")
        p["yac"] = pd.to_numeric(p.get("yards_after_catch", np.nan), errors="coerce")
        p["screen"] = p["air"].le(0).where(p["air"].notna())
        p["explosive20"] = p["yards"].ge(20).astype(int)
        g = p.groupby(["season", "week", "team", "opponent", "player_clean_key"], dropna=False).agg(
            pbp_targets=("player_clean_key", "size"),
            pbp_receptions=("complete", "sum"),
            pbp_air_sum=("air", "sum"),
            pbp_air_n=("air", "count"),
            pbp_yac_sum=("yac", "sum"),
            pbp_yac_n=("yac", "count"),
            pbp_screen_n=("screen", "sum"),
            pbp_screen_obs=("screen", "count"),
            pbp_explosive20=("explosive20", "sum"),
        ).reset_index()
        frames.append(g)
        audit["seasons_loaded"].append(int(season))
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    audit["rows"] = int(len(out))
    return out, audit


def _r26_flags(r26_root: Path) -> pd.DataFrame:
    frames = []
    for season in range(2020, 2026):
        p = r26_root / f"r27b_r26_parent_{season}" / "r26_predictions.csv"
        if not p.exists() or not p.stat().st_size:
            continue
        x = pd.read_csv(p, low_memory=False)
        keep = [c for c in ("season", "week", "event_id", "team", "player_clean_key", "role", "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl") if c in x.columns]
        frames.append(x[keep].copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["team"] = out["team"].map(canon_team)
    out["player_clean_key"] = out["player_clean_key"].fillna("").astype(str)
    return out.drop_duplicates(["season", "week", "team", "player_clean_key"])


def _production_frames(hist_dirs: dict[int, Path]) -> pd.DataFrame:
    parts = []
    for season, d in sorted(hist_dirs.items()):
        inputs = r26.load_bundle_inputs(d)
        for week in r26.weeks(int(season)):
            try:
                f = r27._efficiency_frame(d, int(season), int(week), inputs).copy()
            except Exception:
                continue
            f["season"] = int(season)
            f["week"] = int(week)
            parts.append(f)
    if not parts:
        raise RuntimeError("R27B produced no production-efficiency context")
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["team"] = out["team"].map(canon_team)
    return out.drop_duplicates(["season", "week", "team", "player_clean_key"])


def _summarize_player(h: pd.DataFrame, prior_ypt: float, prior_catch: float, prior_ypr: float) -> dict:
    t = float(pd.to_numeric(h.get("targets", 0), errors="coerce").fillna(0).sum())
    r = float(pd.to_numeric(h.get("receptions", 0), errors="coerce").fillna(0).sum())
    y = float(pd.to_numeric(h.get("rec_yards", 0), errors="coerce").fillna(0).sum())
    ypt = _rate(y, t); cr = _rate(r, t); ypr = _rate(y, r)
    return {
        "targets": t, "receptions": r,
        "ypt_eb": _eb(ypt, t, prior_ypt, 20.0),
        "catch_eb": _eb(cr, t, prior_catch, 20.0),
        "ypr_eb": _eb(ypr, r, prior_ypr, 12.0),
        "tpg": _rate(t, float(h[["season", "week"]].drop_duplicates().shape[0])),
        "rpg": _rate(r, float(h[["season", "week"]].drop_duplicates().shape[0])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-root", type=Path, required=True)
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    hist_dirs = {s: args.hist_root / f"r27b_hist_{s}" for s in range(2019, 2026)}
    logs = _dedup_logs(list(hist_dirs.values()))
    prod = _production_frames(hist_dirs)
    flags = _r26_flags(args.r26_root)
    pbp, pbp_audit = _pbp_receiver_history(list(range(2018, 2026)))

    rb_logs = logs.loc[logs["position"].isin(RB_POS)].copy()
    base = prod.copy()
    base = base.merge(
        logs[["season", "week", "team", "opponent", "player_clean_key", "player", "position"]].drop_duplicates(["season", "week", "team", "player_clean_key"]),
        on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one",
    )
    if not flags.empty:
        base = base.merge(flags, on=["season", "week", "team", "player_clean_key"], how="left", suffixes=("", "_r26"))
    for c in ("vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl"):
        if c not in base.columns:
            base[c] = 0
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0).astype(int)
    base["role"] = base.get("role", "").fillna("").astype(str)
    base["role_is_rb1"] = base["role"].str.upper().eq("RB1").astype(int)
    base["role_is_rb2plus"] = base["role"].str.upper().eq("RB2+").astype(int)
    base["week1"] = pd.to_numeric(base["week"], errors="coerce").eq(1).astype(int)

    rows = []
    for _, row in base.sort_values(["season", "week", "team", "player_clean_key"]).iterrows():
        season, week = int(row["season"]), int(row["week"])
        pkey, team = str(row["player_clean_key"]), canon_team(row["team"])
        opp = canon_team(row.get("opponent", ""))
        prior_all = _before(rb_logs, season, week)
        league_t = float(prior_all["targets"].fillna(0).sum()); league_r = float(prior_all["receptions"].fillna(0).sum()); league_y = float(prior_all["rec_yards"].fillna(0).sum())
        prior_ypt = _rate(league_y, league_t)
        prior_catch = _rate(league_r, league_t)
        prior_ypr = _rate(league_y, league_r)

        ph = prior_all.loc[prior_all["player_clean_key"].astype(str).eq(pkey)].sort_values(["season", "week"])
        same = ph.loc[pd.to_numeric(ph["season"], errors="coerce").eq(season)]
        career = _summarize_player(ph, prior_ypt, prior_catch, prior_ypr)
        season_s = _summarize_player(same, prior_ypt, prior_catch, prior_ypr)
        last4 = _summarize_player(ph.tail(4), prior_ypt, prior_catch, prior_ypr)
        last8 = _summarize_player(ph.tail(8), prior_ypt, prior_catch, prior_ypr)

        team_h = _before(logs, season, week)
        team_h = team_h.loc[team_h["team"].eq(team)]
        team_rb = team_h.loc[team_h["position"].isin(RB_POS)]
        rb_t = float(team_rb["targets"].fillna(0).sum()); rb_r = float(team_rb["receptions"].fillna(0).sum()); rb_y = float(team_rb["rec_yards"].fillna(0).sum())
        all_t = float(team_h["targets"].fillna(0).sum())
        pass_att = float(team_h["pass_att"].fillna(0).sum())

        def_h = _before(rb_logs, season, week)
        def_h = def_h.loc[def_h["opponent"].eq(opp)]
        dt = float(def_h["targets"].fillna(0).sum()); dr = float(def_h["receptions"].fillna(0).sum()); dy = float(def_h["rec_yards"].fillna(0).sum())

        pp = _before(pbp, season, week) if not pbp.empty else pd.DataFrame()
        player_pp = pp.loc[pp["player_clean_key"].astype(str).eq(pkey)] if not pp.empty else pd.DataFrame()
        team_pp = pp.loc[pp["team"].eq(team)] if not pp.empty else pd.DataFrame()
        def_pp = pp.loc[pp["opponent"].eq(opp)] if not pp.empty else pd.DataFrame()

        pt = float(player_pp.get("pbp_targets", pd.Series(dtype=float)).sum()) if len(player_pp) else 0.0
        pr = float(player_pp.get("pbp_receptions", pd.Series(dtype=float)).sum()) if len(player_pp) else 0.0
        pairn = float(player_pp.get("pbp_air_n", pd.Series(dtype=float)).sum()) if len(player_pp) else 0.0
        pyacn = float(player_pp.get("pbp_yac_n", pd.Series(dtype=float)).sum()) if len(player_pp) else 0.0
        pscn = float(player_pp.get("pbp_screen_obs", pd.Series(dtype=float)).sum()) if len(player_pp) else 0.0
        pexpl_n = float(player_pp.get("pbp_explosive20", pd.Series(dtype=float)).sum()) if len(player_pp) else 0.0
        raw_air = _rate(float(player_pp.get("pbp_air_sum", pd.Series(dtype=float)).sum()), pairn) if len(player_pp) else np.nan
        raw_yac = _rate(float(player_pp.get("pbp_yac_sum", pd.Series(dtype=float)).sum()), pyacn) if len(player_pp) else np.nan
        raw_screen = _rate(float(player_pp.get("pbp_screen_n", pd.Series(dtype=float)).sum()), pscn) if len(player_pp) else np.nan
        raw_expl = _rate(pexpl_n, pt)

        # Strict-prior RB population PBP priors for shrinkage.
        pop_pt = float(pp.get("pbp_targets", pd.Series(dtype=float)).sum()) if len(pp) else 0.0
        pop_pr = float(pp.get("pbp_receptions", pd.Series(dtype=float)).sum()) if len(pp) else 0.0
        pop_air_n = float(pp.get("pbp_air_n", pd.Series(dtype=float)).sum()) if len(pp) else 0.0
        pop_yac_n = float(pp.get("pbp_yac_n", pd.Series(dtype=float)).sum()) if len(pp) else 0.0
        pop_scr_n = float(pp.get("pbp_screen_obs", pd.Series(dtype=float)).sum()) if len(pp) else 0.0
        pop_air = _rate(float(pp.get("pbp_air_sum", pd.Series(dtype=float)).sum()), pop_air_n) if len(pp) else np.nan
        pop_yac = _rate(float(pp.get("pbp_yac_sum", pd.Series(dtype=float)).sum()), pop_yac_n) if len(pp) else np.nan
        pop_scr = _rate(float(pp.get("pbp_screen_n", pd.Series(dtype=float)).sum()), pop_scr_n) if len(pp) else np.nan
        pop_expl = _rate(float(pp.get("pbp_explosive20", pd.Series(dtype=float)).sum()), pop_pt) if len(pp) else np.nan

        tyac_n = float(team_pp.get("pbp_yac_n", pd.Series(dtype=float)).sum()) if len(team_pp) else 0.0
        dyac_n = float(def_pp.get("pbp_yac_n", pd.Series(dtype=float)).sum()) if len(def_pp) else 0.0
        dexp_t = float(def_pp.get("pbp_targets", pd.Series(dtype=float)).sum()) if len(def_pp) else 0.0

        rec = {
            "season": season, "week": week, "event_id": str(row["event_id"]), "team": team, "opponent": opp,
            "player": row.get("player", ""), "player_clean_key": pkey,
            "production_ypt": float(row["production_ypt"]), "production_catch_rate": float(row["production_catch_rate"]),
            "implied_production_ypr": _rate(float(row["production_ypt"]), float(row["production_catch_rate"])),
            "player_career_ypt_eb": career["ypt_eb"], "player_season_ypt_eb": season_s["ypt_eb"],
            "player_last4_ypt_eb": last4["ypt_eb"], "player_last8_ypt_eb": last8["ypt_eb"],
            "player_career_catch_eb": career["catch_eb"], "player_career_ypr_eb": career["ypr_eb"],
            "player_targets_per_game": career["tpg"], "player_receptions_per_game": career["rpg"], "player_prior_targets": career["targets"],
            "player_air_yards_per_target_eb": _eb(raw_air, pairn, pop_air, 20.0),
            "player_yac_per_reception_eb": _eb(raw_yac, pyacn, pop_yac, 12.0),
            "player_screen_rate_eb": _eb(raw_screen, pscn, pop_scr, 20.0),
            "player_explosive20_rate_eb": _eb(raw_expl, pt, pop_expl, 20.0),
            "offense_rb_target_share": _rate(rb_t, all_t), "offense_rb_ypt": _rate(rb_y, rb_t),
            "offense_rb_yac_per_reception": _rate(float(team_pp.get("pbp_yac_sum", pd.Series(dtype=float)).sum()) if len(team_pp) else np.nan, tyac_n),
            "offense_rb_checkdown_proxy": _rate(rb_t, pass_att),
            "opponent_rb_ypt_allowed": _rate(dy, dt),
            "opponent_rb_yac_per_reception_allowed": _rate(float(def_pp.get("pbp_yac_sum", pd.Series(dtype=float)).sum()) if len(def_pp) else np.nan, dyac_n),
            "opponent_rb_explosive20_rate_allowed": _rate(float(def_pp.get("pbp_explosive20", pd.Series(dtype=float)).sum()) if len(def_pp) else np.nan, dexp_t),
            "opponent_rb_catch_rate_allowed": _rate(dr, dt),
            "role_is_rb1": int(row["role_is_rb1"]), "role_is_rb2plus": int(row["role_is_rb2plus"]),
            "vacancy_active": int(row["vacancy_active"]), "vacancy_incumbent": int(row["vacancy_incumbent"]),
            "vacancy_new_veteran": int(row["vacancy_new_veteran"]), "vacancy_no_prior_nfl": int(row["vacancy_no_prior_nfl"]),
            "week1": int(row["week1"]),
        }
        # Missing indicators are frozen for efficiency-history/PBP/context features.
        for c in list(rec):
            if c.startswith("player_") or c.startswith("offense_") or c.startswith("opponent_"):
                if c not in {"player", "player_clean_key", "player_prior_targets", "player_targets_per_game", "player_receptions_per_game"}:
                    rec[f"{c}_missing"] = int(not np.isfinite(rec[c])) if isinstance(rec[c], (int, float, np.integer, np.floating)) else 0
        rows.append(rec)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("R27B produced zero feature rows")
    if features.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("R27B feature rows duplicate season/week/team/player")

    labels = logs.loc[logs["position"].isin(RB_POS), ["season", "week", "team", "player_clean_key", "targets", "receptions", "rec_yards"]].copy()
    labels = labels.rename(columns={"targets": "actual_targets", "receptions": "actual_receptions", "rec_yards": "actual_rec_yards"})
    labels = labels.drop_duplicates(["season", "week", "team", "player_clean_key"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_dir / "r27b_efficiency_features.csv", index=False)
    labels.to_csv(args.out_dir / "r27b_efficiency_labels.csv", index=False)
    audit = {
        "study": "RB_R27B_STRICT_PRIOR_RECEIVING_EFFICIENCY_V1",
        "feature_rows": int(len(features)), "label_rows": int(len(labels)),
        "feature_seasons": sorted(pd.to_numeric(features["season"], errors="coerce").dropna().astype(int).unique().tolist()),
        "sportsbook_inputs_used": 0, "target_game_outcomes_used_in_features": 0,
        "pbp": pbp_audit,
        "feature_columns": features.columns.tolist(),
    }
    (args.out_dir / "r27b_feature_source_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
