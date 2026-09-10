#!/usr/bin/env python3
"""Build the frozen R27B V2 strict-prior novel RB receiving-efficiency dataset.

The feature set is intentionally limited to target-shape/YAC/checkdown/opponent
context that was not already represented by production Bayes YPT or the failed
R23 shrunk-YPR experiment. No sportsbook inputs. No target-game outcomes enter
feature construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import component_predictions as cp
from scripts.backtest import evaluate_rb_r26_vacancy_gated_r9_v1 as r26
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.pbp import get_pbp

RB_POS = {"RB", "FB", "HB", "TB"}
CONTEXT_FEATURES = [
    "player_air_yards_per_target_prior",
    "player_yac_per_reception_prior",
    "player_screen_target_rate_prior",
    "player_explosive20_target_rate_prior",
    "team_rb_targets_per_official_pass_attempt_prior",
    "team_rb_air_yards_per_target_prior",
    "team_rb_yac_per_reception_prior",
    "team_rb_screen_target_rate_prior",
    "opp_rb_air_yards_allowed_per_target_prior",
    "opp_rb_yac_allowed_per_reception_prior",
    "opp_rb_catch_rate_allowed_prior",
    "opp_rb_explosive20_allowed_per_target_prior",
    "opp_rb_screen_target_rate_faced_prior",
]


def _key(v) -> str:
    try:
        _, k = canonicalize_player_name_safe(v)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _rate(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan


def _eb(raw: float, n: float, prior: float, k: float) -> float:
    if not np.isfinite(prior):
        return np.nan
    if not np.isfinite(raw) or not np.isfinite(n) or n <= 0:
        return float(prior)
    return float((n * raw + k * prior) / (n + k))


def _before(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if df.empty:
        return df
    s = pd.to_numeric(df["season"], errors="coerce")
    w = pd.to_numeric(df["week"], errors="coerce")
    return df.loc[(s < season) | ((s == season) & (w < week))].copy()


def _weekly_map(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    x = _normalize_weekly(_to_pandas(raw), int(season)).copy()
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["player_id_norm"] = x.get("player_id", "").astype("string").fillna("").str.strip()
    x["player_clean_key"] = x.get("player_clean_key", x.get("player", "")).astype("string").fillna("").map(_key)
    x["position"] = x.get("position", "").astype("string").fillna("").str.upper().str.strip()
    return x


def _load_rb_pbp(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rb_parts: list[pd.DataFrame] = []
    pa_parts: list[pd.DataFrame] = []
    audit: dict = {"seasons_requested": seasons, "seasons_loaded": [], "rows": 0}
    for season in seasons:
        p = get_pbp(int(season), min_rows=1).copy()
        p.columns = [str(c).strip().lower() for c in p.columns]
        if "season_type" in p.columns:
            reg = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                p = reg
        receiver_name = _col(p, ("receiver_player_name", "receiver_name", "receiver"))
        receiver_id = _col(p, ("receiver_player_id", "receiver_id"))
        req = ["week", "posteam", "defteam", "complete_pass", "yards_gained", "air_yards", "yards_after_catch"]
        missing = [c for c in req if c not in p.columns]
        if (receiver_name is None and receiver_id is None) or missing:
            raise RuntimeError(f"R27B V2 PBP schema incomplete season={season} missing={missing} receiver={receiver_name}/{receiver_id}")

        p["season"] = int(season)
        p["week"] = pd.to_numeric(p["week"], errors="coerce")
        p["team"] = p["posteam"].map(canon_team)
        p["opponent"] = p["defteam"].map(canon_team)
        p["receiver_id_norm"] = p[receiver_id].astype("string").fillna("").str.strip() if receiver_id else ""
        p["receiver_name_norm"] = p[receiver_name].astype("string").fillna("").str.strip() if receiver_name else ""
        p["receiver_name_key"] = p["receiver_name_norm"].map(_key)

        weekly = _weekly_map(int(season))
        id_map = weekly.loc[weekly["player_id_norm"].ne(""), ["week", "team", "player_id_norm", "player_clean_key", "position"]].drop_duplicates(["week", "team", "player_id_norm"])
        name_map = weekly.loc[weekly["player_clean_key"].ne(""), ["week", "team", "player_clean_key", "position"]].drop_duplicates(["week", "team", "player_clean_key"])
        tmask = (p["receiver_id_norm"].ne("") | p["receiver_name_norm"].ne("")) & p["week"].notna() & p["team"].ne("")
        t = p.loc[tmask].copy()
        t = t.merge(
            id_map.rename(columns={"player_id_norm": "receiver_id_norm", "player_clean_key": "key_by_id", "position": "position_by_id"}),
            on=["week", "team", "receiver_id_norm"], how="left", validate="many_to_one",
        )
        t = t.merge(
            name_map.rename(columns={"player_clean_key": "receiver_name_key", "position": "position_by_name"}),
            on=["week", "team", "receiver_name_key"], how="left", validate="many_to_one",
        )
        t["receiver_position"] = t["position_by_id"].replace("", pd.NA).combine_first(t["position_by_name"])
        t["player_clean_key"] = t["key_by_id"].replace("", pd.NA).combine_first(t["receiver_name_key"]).fillna("").astype(str)
        t = t.loc[t["receiver_position"].fillna("").astype(str).str.upper().isin(RB_POS) & t["player_clean_key"].ne("")].copy()
        t["complete"] = pd.to_numeric(t["complete_pass"], errors="coerce").fillna(0).eq(1).astype(int)
        t["yards"] = pd.to_numeric(t["yards_gained"], errors="coerce")
        t["air"] = pd.to_numeric(t["air_yards"], errors="coerce")
        t["yac"] = pd.to_numeric(t["yards_after_catch"], errors="coerce")
        t["screen"] = t["air"].le(0).where(t["air"].notna())
        t["explosive20"] = t["yards"].ge(20).where(t["yards"].notna()).astype("float")
        g = t.groupby(["season", "week", "team", "opponent", "player_clean_key"], dropna=False).agg(
            rb_targets=("player_clean_key", "size"),
            rb_receptions=("complete", "sum"),
            air_sum=("air", "sum"),
            air_n=("air", "count"),
            yac_sum=("yac", "sum"),
            yac_n=("yac", "count"),
            screen_n=("screen", "sum"),
            screen_obs=("screen", "count"),
            explosive20_n=("explosive20", "sum"),
            explosive20_obs=("explosive20", "count"),
        ).reset_index()
        rb_parts.append(g)

        pass_attempt = pd.to_numeric(p.get("pass_attempt", 0), errors="coerce").fillna(0).eq(1)
        sack = pd.to_numeric(p.get("sack", 0), errors="coerce").fillna(0).eq(1)
        pa = p.loc[pass_attempt & ~sack & p["week"].notna() & p["team"].ne("")].groupby(["season", "week", "team"]).size().rename("official_pass_attempts").reset_index()
        pa_parts.append(pa)
        audit["seasons_loaded"].append(int(season))

    rb = pd.concat(rb_parts, ignore_index=True, sort=False) if rb_parts else pd.DataFrame()
    pa = pd.concat(pa_parts, ignore_index=True, sort=False) if pa_parts else pd.DataFrame()
    audit["rows"] = int(len(rb))
    return rb, pa, audit


def _production_rows(hist_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in range(2019, 2026):
        d = hist_root / f"r27b_hist_{season}"
        inputs = r26.load_bundle_inputs(d)
        for week in r26.weeks(int(season)):
            base = r26.build_base(d, inputs, int(season), int(week)).copy()
            pos = base.get("position", pd.Series("", index=base.index)).fillna("").astype(str).str.upper().str.strip()
            base = base.loc[pos.isin(RB_POS)].copy()
            if base.empty:
                continue
            if "player_clean_key" not in base.columns:
                base["player_clean_key"] = base.get("player", "").map(r26.name_key)
            prior_catch = r26._catch_prior(inputs["logs"], int(season), int(week))
            ypt, catch = [], []
            for _, z in base.iterrows():
                cr = r26.finite(z.get("rules_catch_rate"), r26.finite(z.get("bayes_receptions_per_target"), prior_catch))
                yy = max(r26.finite(z.get("rules_ypt"), r26.finite(z.get("bayes_ypt"), 0.0)), 0.0)
                catch.append(float(np.clip(cr, 0.35, 0.95)))
                ypt.append(float(yy))
            base["production_catch_rate"] = catch
            base["production_ypt"] = ypt
            base["season"] = int(season)
            base["week"] = int(week)
            base["team"] = base["team"].map(canon_team)
            base["opponent"] = base.get("opponent", "").map(canon_team)
            base["event_id"] = base["event_id"].astype(str)
            base["player_clean_key"] = base["player_clean_key"].fillna("").astype(str)
            base["_share"] = pd.to_numeric(base.get("entitlement_tgt_share", 0), errors="coerce").fillna(0.0)
            base["_rank"] = base.groupby(["event_id", "team"])["_share"].rank(method="first", ascending=False)
            base["inferred_role"] = np.where(base["_rank"].eq(1), "RB1", "RB2+")
            rows.append(base[["season", "week", "event_id", "team", "opponent", "player", "player_clean_key", "production_ypt", "production_catch_rate", "inferred_role"]])
    if not rows:
        raise RuntimeError("R27B V2 produced no production context rows")
    out = pd.concat(rows, ignore_index=True, sort=False)
    return out.drop_duplicates(["season", "week", "event_id", "team", "player_clean_key"])


def _r26_flags(r26_root: Path) -> pd.DataFrame:
    parts = []
    for season in range(2020, 2026):
        p = r26_root / f"r27b_r26_parent_{season}" / "r26_predictions.csv"
        if not p.exists() or not p.stat().st_size:
            raise RuntimeError(f"R27B V2 missing exact R26 parent {p}")
        x = pd.read_csv(p, low_memory=False)
        keep = [c for c in ["season", "week", "event_id", "team", "player_clean_key", "role", "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl"] if c in x.columns]
        x = x[keep].copy()
        x["team"] = x["team"].map(canon_team)
        x["event_id"] = x["event_id"].astype(str)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
        parts.append(x)
    return pd.concat(parts, ignore_index=True, sort=False).drop_duplicates(["season", "week", "event_id", "team", "player_clean_key"])


def _labels(hist_root: Path) -> pd.DataFrame:
    parts = []
    for season in range(2019, 2026):
        d = hist_root / f"r27b_hist_{season}"
        inputs = r26.load_bundle_inputs(d)
        for week in r26.weeks(int(season)):
            actual = cp.build_actual_rows(inputs["logs"], int(season), int(week))
            if actual.empty:
                continue
            rec = actual.loc[actual["market"].astype(str).eq("receptions"), ["team", "player_clean_key", "actual", "actual_opportunities"]].rename(columns={"actual": "actual_receptions", "actual_opportunities": "actual_targets"})
            yds = actual.loc[actual["market"].astype(str).eq("rec_yards"), ["team", "player_clean_key", "actual"]].rename(columns={"actual": "actual_rec_yards"})
            x = rec.merge(yds, on=["team", "player_clean_key"], how="outer").drop_duplicates(["team", "player_clean_key"])
            x["season"] = int(season)
            x["week"] = int(week)
            x["team"] = x["team"].map(canon_team)
            x["player_clean_key"] = x["player_clean_key"].astype(str)
            parts.append(x)
    if not parts:
        raise RuntimeError("R27B V2 produced no labels")
    return pd.concat(parts, ignore_index=True, sort=False).drop_duplicates(["season", "week", "team", "player_clean_key"])


def _sum(x: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(x.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(x) else 0.0


def _feature_record(row: pd.Series, rb: pd.DataFrame, pa: pd.DataFrame) -> dict:
    season, week = int(row["season"]), int(row["week"])
    team, opp, pkey = canon_team(row["team"]), canon_team(row.get("opponent", "")), str(row["player_clean_key"])
    hist = _before(rb, season, week)
    ph = hist.loc[hist["player_clean_key"].astype(str).eq(pkey)]
    th = hist.loc[hist["team"].eq(team)]
    oh = hist.loc[hist["opponent"].eq(opp)] if opp else hist.iloc[0:0]
    pah = _before(pa, season, week)
    tpa = pah.loc[pah["team"].eq(team)]

    pop_t = _sum(hist, "rb_targets")
    pop_r = _sum(hist, "rb_receptions")
    pop_air_n = _sum(hist, "air_n")
    pop_yac_n = _sum(hist, "yac_n")
    pop_screen_obs = _sum(hist, "screen_obs")
    pop_exp_obs = _sum(hist, "explosive20_obs")
    pop_air = _rate(_sum(hist, "air_sum"), pop_air_n)
    pop_yac = _rate(_sum(hist, "yac_sum"), pop_yac_n)
    pop_screen = _rate(_sum(hist, "screen_n"), pop_screen_obs)
    pop_exp = _rate(_sum(hist, "explosive20_n"), pop_exp_obs)
    pop_catch = _rate(pop_r, pop_t)
    pop_pa = _sum(pah, "official_pass_attempts")
    pop_check = _rate(pop_t, pop_pa)

    def shr(x: pd.DataFrame, num: str, den: str, prior: float, k: float) -> tuple[float, int]:
        n = _sum(x, den)
        raw = _rate(_sum(x, num), n)
        return _eb(raw, n, prior, k), int(n <= 0)

    f: dict[str, float | int | str] = {
        "season": season, "week": week, "event_id": str(row["event_id"]), "team": team, "opponent": opp,
        "player": row.get("player", ""), "player_clean_key": pkey,
        "production_ypt": float(row["production_ypt"]), "production_catch_rate": float(row["production_catch_rate"]),
    }
    f["player_air_yards_per_target_prior"], f["player_air_yards_per_target_prior_missing"] = shr(ph, "air_sum", "air_n", pop_air, 20.0)
    f["player_yac_per_reception_prior"], f["player_yac_per_reception_prior_missing"] = shr(ph, "yac_sum", "yac_n", pop_yac, 12.0)
    f["player_screen_target_rate_prior"], f["player_screen_target_rate_prior_missing"] = shr(ph, "screen_n", "screen_obs", pop_screen, 20.0)
    f["player_explosive20_target_rate_prior"], f["player_explosive20_target_rate_prior_missing"] = shr(ph, "explosive20_n", "explosive20_obs", pop_exp, 20.0)

    team_t = _sum(th, "rb_targets")
    team_pa = _sum(tpa, "official_pass_attempts")
    f["team_rb_targets_per_official_pass_attempt_prior"] = _eb(_rate(team_t, team_pa), team_pa, pop_check, 40.0)
    f["team_rb_targets_per_official_pass_attempt_prior_missing"] = int(team_pa <= 0)
    f["team_rb_air_yards_per_target_prior"], f["team_rb_air_yards_per_target_prior_missing"] = shr(th, "air_sum", "air_n", pop_air, 20.0)
    f["team_rb_yac_per_reception_prior"], f["team_rb_yac_per_reception_prior_missing"] = shr(th, "yac_sum", "yac_n", pop_yac, 12.0)
    f["team_rb_screen_target_rate_prior"], f["team_rb_screen_target_rate_prior_missing"] = shr(th, "screen_n", "screen_obs", pop_screen, 20.0)

    f["opp_rb_air_yards_allowed_per_target_prior"], f["opp_rb_air_yards_allowed_per_target_prior_missing"] = shr(oh, "air_sum", "air_n", pop_air, 20.0)
    f["opp_rb_yac_allowed_per_reception_prior"], f["opp_rb_yac_allowed_per_reception_prior_missing"] = shr(oh, "yac_sum", "yac_n", pop_yac, 12.0)
    f["opp_rb_catch_rate_allowed_prior"], f["opp_rb_catch_rate_allowed_prior_missing"] = shr(oh, "rb_receptions", "rb_targets", pop_catch, 20.0)
    f["opp_rb_explosive20_allowed_per_target_prior"], f["opp_rb_explosive20_allowed_per_target_prior_missing"] = shr(oh, "explosive20_n", "explosive20_obs", pop_exp, 20.0)
    f["opp_rb_screen_target_rate_faced_prior"], f["opp_rb_screen_target_rate_faced_prior_missing"] = shr(oh, "screen_n", "screen_obs", pop_screen, 20.0)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-root", type=Path, required=True)
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    production = _production_rows(a.hist_root)
    flags = _r26_flags(a.r26_root)
    production = production.merge(flags, on=["season", "week", "event_id", "team", "player_clean_key"], how="left", suffixes=("", "_r26"))
    production["role"] = production.get("role", pd.Series(index=production.index, dtype=object)).fillna(production["inferred_role"]).astype(str)
    for c in ("vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl"):
        production[c] = pd.to_numeric(production.get(c, 0), errors="coerce").fillna(0).astype(int)
    production["role_is_rb1"] = production["role"].str.upper().eq("RB1").astype(int)
    production["role_is_rb2plus"] = production["role"].str.upper().eq("RB2+").astype(int)
    production["week1"] = production["week"].eq(1).astype(int)

    rb_pbp, pass_attempts, pbp_audit = _load_rb_pbp(list(range(2019, 2026)))
    rows = []
    for i, row in production.sort_values(["season", "week", "team", "player_clean_key"]).iterrows():
        rec = _feature_record(row, rb_pbp, pass_attempts)
        for c in ("role_is_rb1", "role_is_rb2plus", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl", "week1"):
            rec[c] = int(row[c])
        rec["role"] = row["role"]
        rec["vacancy_active"] = int(row["vacancy_active"])
        rows.append(rec)
        if len(rows) % 1000 == 0:
            print(f"[r27b-v2-features] rows={len(rows)}")

    features = pd.DataFrame(rows)
    labels = _labels(a.hist_root)
    if features.empty or features.duplicated(["season", "week", "event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("R27B V2 feature dataset empty or duplicate")
    missing_primary = [c for c in CONTEXT_FEATURES if c not in features.columns]
    if missing_primary:
        raise RuntimeError(f"R27B V2 missing frozen primary features {missing_primary}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(a.out_dir / "r27b_v2_features.csv", index=False)
    labels.to_csv(a.out_dir / "r27b_v2_labels.csv", index=False)
    miss = {c: float(pd.to_numeric(features[c], errors="coerce").isna().mean()) for c in CONTEXT_FEATURES}
    audit = {
        "study": "RB_R27B_V2_NOVEL_EFFICIENCY_CONTEXT",
        "feature_rows": int(len(features)),
        "label_rows": int(len(labels)),
        "feature_seasons": sorted(features["season"].astype(int).unique().tolist()),
        "primary_features": CONTEXT_FEATURES,
        "primary_feature_nan_rate": miss,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_used_in_features": 0,
        "generic_ypt_ypr_persistence_features_used": 0,
        "pbp_source": pbp_audit,
    }
    (a.out_dir / "r27b_v2_feature_source_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
