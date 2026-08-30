#!/usr/bin/env python3
"""M83: defensive adaptive gameplan source/mechanism audit.

No QB outcomes are read. The audit asks whether a defense's target-game tactical
response can be predicted from how that same defense previously deviated from
its own baseline against offenses with similar *pregame* archetypes.

Mechanism scoring is 2024 only. 2025 is source-coverage audit only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FTN_SEASONS = (2022, 2023, 2024, 2025)
PART_SEASONS = (2023, 2024, 2025)
SCORE_SEASON = 2024
REG_WEEKS = set(range(1, 19))
MIN_JOIN = 0.95
MIN_FIELD_COVERAGE = 0.80
MIN_OFF_HISTORY = 3
MIN_DEF_TARGET_HISTORY = 6
MIN_DEF_LABEL_HISTORY = 3
TRAIL = 8
K_NEIGHBORS = 4
MIN_COMMON_ARCHETYPE = 7
MIN_DENSITY = 0.80
MIN_MEDIAN_SIMILARITY = 0.70
MIN_MAE_GAIN_PCT = 0.05
MIN_CORR_GAIN = 0.05

ARCHETYPE_FIELDS = [
    "motion_rate", "screen_rate", "rpo_rate", "no_huddle_rate",
    "play_action_rate", "shotgun_rate", "backfield_scaled",
    "oop_rate", "qb_fault_sack_rate", "pass_share",
]
FTN_RESPONSE_FIELDS = ["blitzers_mean", "blitz_event_rate", "pass_rushers_mean"]
PART_RESPONSE_FIELDS = ["man_rate", "zone_rate", "avg_box"]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def read_url_parquet(url: str, agent: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": agent})
    with urlopen(req, timeout=180) as r:
        raw = r.read()
        final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final, "bytes": len(raw), "sha256": sha256_bytes(raw)
    }


def present(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.notna()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").notna()
    z = s.astype("string").str.strip().str.lower()
    return z.notna() & z.ne("") & z.ne("nan") & z.ne("none")


def coverage(s: pd.Series) -> float:
    return float(present(s).mean()) if len(s) else 0.0


def bool_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    z = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[z.isin(["1", "true", "t", "yes", "y"])] = 1.0
    out.loc[z.isin(["0", "false", "f", "no", "n"])] = 0.0
    return out


def before(a_season: int, a_week: int, b_season: int, b_week: int) -> bool:
    return (int(a_season), int(a_week)) < (int(b_season), int(b_week))


def load_ftn_pbp(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged, source_rows, cov_rows = [], [], []
    req_ftn = {
        "nflverse_game_id", "nflverse_play_id", "season", "week",
        "is_motion", "is_screen_pass", "is_rpo", "is_no_huddle",
        "is_play_action", "qb_location", "n_offense_backfield",
        "is_qb_out_of_pocket", "is_qb_fault_sack", "n_blitzers",
        "n_pass_rushers",
    }
    req_pbp = {
        "game_id", "play_id", "season", "week", "season_type", "posteam",
        "defteam", "pass_attempt", "rush_attempt", "sack",
    }
    for season in FTN_SEASONS:
        f_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        p_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        try:
            ftn, fm = read_url_parquet(f_url, "m83-defense-adaptation")
            pbp, pm = read_url_parquet(p_url, "m83-defense-adaptation")
            ftn, pbp = lower(ftn), lower(pbp)
            mf, mp = sorted(req_ftn - set(ftn.columns)), sorted(req_pbp - set(pbp.columns))
            if mf or mp:
                raise RuntimeError(f"schema missing ftn={mf} pbp={mp}")
            ftn = ftn.loc[pd.to_numeric(ftn.week, errors="coerce").between(1, 18)].copy()
            pbp = pbp.loc[
                pbp.season_type.astype(str).str.upper().eq("REG")
                & pd.to_numeric(pbp.week, errors="coerce").between(1, 18)
            ].copy()
            fw = set(pd.to_numeric(ftn.week, errors="coerce").dropna().astype(int))
            pw = set(pd.to_numeric(pbp.week, errors="coerce").dropna().astype(int))
            source_rows += [
                {"source": "FTN", "season": season, "rows": len(ftn), "weeks_1_18_complete": REG_WEEKS.issubset(fw), **fm, "status": "OK"},
                {"source": "PBP", "season": season, "rows": len(pbp), "weeks_1_18_complete": REG_WEEKS.issubset(pw), **pm, "status": "OK"},
            ]
            for fld in sorted(req_ftn - {"nflverse_game_id", "nflverse_play_id", "season", "week"}):
                cov_rows.append({"season": season, "field": fld, "coverage": coverage(ftn[fld])})
            ftn["join_game"] = ftn.nflverse_game_id.astype(str)
            ftn["join_play"] = pd.to_numeric(ftn.nflverse_play_id, errors="coerce")
            p = pbp[["game_id", "play_id", "posteam", "defteam", "pass_attempt", "rush_attempt", "sack"]].copy()
            p["join_game"] = p.game_id.astype(str)
            p["join_play"] = pd.to_numeric(p.play_id, errors="coerce")
            m = ftn.merge(
                p.drop(columns=["game_id", "play_id"]),
                on=["join_game", "join_play"], how="left", validate="one_to_one"
            )
            jr = float(m.posteam.notna().mean())
            source_rows.append({
                "source": "FTN_PBP_JOIN", "season": season, "rows": len(m),
                "weeks_1_18_complete": True, "url": "", "bytes": 0,
                "sha256": "", "status": "OK", "join_rate": jr,
            })
            m["game_id"] = m.join_game
            m["season"] = pd.to_numeric(m.season, errors="coerce").astype(int)
            m["week"] = pd.to_numeric(m.week, errors="coerce").astype(int)
            merged.append(m)
        except Exception as exc:
            print(f"[m83_source_error] FTN/PBP season={season} {type(exc).__name__}: {exc}")
            source_rows.append({
                "source": "FTN_PBP_ERROR", "season": season, "rows": 0,
                "weeks_1_18_complete": False, "url": f_url, "bytes": 0,
                "sha256": "", "status": f"ERROR:{type(exc).__name__}:{exc}",
                "join_rate": 0.0,
            })
    src = pd.DataFrame(source_rows)
    cov = pd.DataFrame(cov_rows)
    src.to_csv(out / "m83_ftn_pbp_source_snapshot.csv", index=False)
    cov.to_csv(out / "m83_ftn_field_coverage.csv", index=False)
    z = pd.concat(merged, ignore_index=True, sort=False) if merged else pd.DataFrame()
    return z, src, cov


def build_ftn_game_frames(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if z.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = z.copy()
    for c in ["is_motion", "is_screen_pass", "is_rpo", "is_no_huddle", "is_play_action", "is_qb_out_of_pocket", "is_qb_fault_sack"]:
        x[c] = bool_num(x[c])
    x["n_offense_backfield"] = pd.to_numeric(x.n_offense_backfield, errors="coerce")
    x["n_blitzers"] = pd.to_numeric(x.n_blitzers, errors="coerce")
    x["n_pass_rushers"] = pd.to_numeric(x.n_pass_rushers, errors="coerce")
    qloc = x.qb_location.astype("string").str.upper().str.strip()
    x["shotgun"] = np.where(qloc.str.contains("SHOTGUN", na=False), 1.0, np.where(present(x.qb_location), 0.0, np.nan))
    pa = pd.to_numeric(x.pass_attempt, errors="coerce").fillna(0).eq(1)
    sack = pd.to_numeric(x.sack, errors="coerce").fillna(0).eq(1)
    rush = pd.to_numeric(x.rush_attempt, errors="coerce").fillna(0).eq(1)
    x["pass_play"] = pa | sack
    x["scrimmage"] = x.pass_play | rush
    x["pass_share_num"] = np.where(x.scrimmage, x.pass_play.astype(float), np.nan)
    x["oop_pass"] = x.is_qb_out_of_pocket.where(x.pass_play)
    x["qbfault_pass"] = x.is_qb_fault_sack.where(x.pass_play)
    x["blitz_pass"] = x.n_blitzers.where(x.pass_play)
    x["blitz_event"] = np.where(x.pass_play & x.n_blitzers.notna(), x.n_blitzers.gt(0).astype(float), np.nan)
    x["rushers_pass"] = x.n_pass_rushers.where(x.pass_play)

    off = x.loc[x.posteam.notna() & x.defteam.notna()].groupby(
        ["season", "week", "game_id", "posteam", "defteam"], as_index=False
    ).agg(
        motion_rate=("is_motion", "mean"),
        screen_rate=("is_screen_pass", "mean"),
        rpo_rate=("is_rpo", "mean"),
        no_huddle_rate=("is_no_huddle", "mean"),
        play_action_rate=("is_play_action", "mean"),
        shotgun_rate=("shotgun", "mean"),
        backfield_mean=("n_offense_backfield", "mean"),
        oop_rate=("oop_pass", "mean"),
        qb_fault_sack_rate=("qbfault_pass", "mean"),
        pass_share=("pass_share_num", "mean"),
    ).rename(columns={"posteam": "team", "defteam": "opponent"})
    off["backfield_scaled"] = (pd.to_numeric(off.backfield_mean, errors="coerce") / 3.0).clip(0, 1)

    deff = x.loc[x.defteam.notna() & x.posteam.notna()].groupby(
        ["season", "week", "game_id", "defteam", "posteam"], as_index=False
    ).agg(
        blitzers_mean=("blitz_pass", "mean"),
        blitz_event_rate=("blitz_event", "mean"),
        pass_rushers_mean=("rushers_pass", "mean"),
    ).rename(columns={"defteam": "defense", "posteam": "offense"})
    return off, deff


def build_pregame_offense_profiles(off: pd.DataFrame) -> pd.DataFrame:
    if off.empty:
        return pd.DataFrame()
    rows = []
    off = off.sort_values(["season", "week", "game_id"]).copy()
    for team, g in off.groupby("team", sort=False):
        g = g.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
        for i, r in g.iterrows():
            prior = g.iloc[:i].tail(TRAIL)
            rec = {
                "season": int(r.season), "week": int(r.week), "game_id": str(r.game_id),
                "team": team, "off_history_n": int(len(prior)),
                "off_profile_natural": bool(len(prior) >= MIN_OFF_HISTORY),
            }
            for f in ARCHETYPE_FIELDS:
                vals = pd.to_numeric(prior[f], errors="coerce") if f in prior else pd.Series(dtype=float)
                rec[f] = float(vals.mean()) if len(vals) and vals.notna().any() else np.nan
            rec["off_profile_dims"] = int(sum(np.isfinite(rec[f]) for f in ARCHETYPE_FIELDS))
            if rec["off_profile_dims"] < MIN_COMMON_ARCHETYPE:
                rec["off_profile_natural"] = False
            rows.append(rec)
    return pd.DataFrame(rows)


def add_defense_baselines(resp: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    if resp.empty:
        return pd.DataFrame()
    rows = []
    for defense, g in resp.sort_values(["season", "week", "game_id"]).groupby("defense", sort=False):
        g = g.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
        for i, r in g.iterrows():
            prior = g.iloc[:i].tail(TRAIL)
            rec = r.to_dict()
            rec["def_history_n"] = int(len(prior))
            for f in fields:
                vals = pd.to_numeric(prior[f], errors="coerce") if f in prior else pd.Series(dtype=float)
                base = float(vals.mean()) if len(vals) and vals.notna().any() else np.nan
                rec[f"baseline_{f}"] = base
                actual = pd.to_numeric(pd.Series([r.get(f)]), errors="coerce").iloc[0]
                rec[f"adaptation_{f}"] = float(actual - base) if np.isfinite(actual) and np.isfinite(base) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def profile_distance(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    diffs = []
    for f in ARCHETYPE_FIELDS:
        av = pd.to_numeric(pd.Series([a.get(f)]), errors="coerce").iloc[0]
        bv = pd.to_numeric(pd.Series([b.get(f)]), errors="coerce").iloc[0]
        if np.isfinite(av) and np.isfinite(bv):
            diffs.append(abs(float(av) - float(bv)))
    if len(diffs) < MIN_COMMON_ARCHETYPE:
        return np.nan, len(diffs)
    return float(np.mean(diffs)), len(diffs)


def adaptive_predictions(resp_b: pd.DataFrame, profiles: pd.DataFrame, fields: list[str], source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resp_b.empty or profiles.empty:
        return pd.DataFrame(), pd.DataFrame()
    prof = profiles.copy()
    # Target offense profile and each candidate prior opponent profile are both the
    # versions known before their respective games.
    target_prof = prof.rename(columns={"team": "offense"})
    x = resp_b.merge(target_prof, on=["season", "week", "game_id", "offense"], how="left", validate="one_to_one")
    rows, density = [], []
    for _, t in x.iterrows():
        if int(t.season) != SCORE_SEASON or not (5 <= int(t.week) <= 18):
            continue
        target_natural = bool(t.get("off_profile_natural", False))
        target_dims = int(pd.to_numeric(pd.Series([t.get("off_profile_dims", 0)]), errors="coerce").fillna(0).iloc[0])
        base_ok = int(t.def_history_n) >= MIN_DEF_TARGET_HISTORY
        if not (target_natural and base_ok):
            continue
        prior = x.loc[
            x.defense.eq(t.defense)
            & x.apply(lambda r: before(int(r.season), int(r.week), int(t.season), int(t.week)), axis=1)
            & x.off_profile_natural.fillna(False)
            & x.def_history_n.ge(MIN_DEF_LABEL_HISTORY)
        ].copy()
        cand = []
        for idx, p in prior.iterrows():
            d, dims = profile_distance(t, p)
            if np.isfinite(d):
                cand.append((float(d), int(dims), idx))
        cand.sort(key=lambda q: (q[0], -q[1], q[2]))
        chosen = cand[:K_NEIGHBORS]
        density.append({
            "source": source_name, "season": int(t.season), "week": int(t.week),
            "game_id": str(t.game_id), "defense": t.defense, "offense": t.offense,
            "target_profile_dims": target_dims, "candidate_count": len(cand),
            "has_four": len(chosen) == K_NEIGHBORS,
            "mean_selected_similarity": float(np.mean([max(0.0, min(1.0, 1.0-d)) for d, _, _ in chosen])) if chosen else np.nan,
        })
        if len(chosen) < K_NEIGHBORS:
            continue
        chosen_rows = [(d, x.loc[idx]) for d, _, idx in chosen]
        rec = {
            "source": source_name, "season": int(t.season), "week": int(t.week),
            "game_id": str(t.game_id), "defense": t.defense, "offense": t.offense,
            "selected_mean_distance": float(np.mean([d for d, _ in chosen_rows])),
            "selected_mean_similarity": float(np.mean([1.0-d for d, _ in chosen_rows])),
        }
        for f in fields:
            actual = pd.to_numeric(pd.Series([t.get(f)]), errors="coerce").iloc[0]
            base = pd.to_numeric(pd.Series([t.get(f"baseline_{f}")]), errors="coerce").iloc[0]
            ds, ws = [], []
            for d, p in chosen_rows:
                lab = pd.to_numeric(pd.Series([p.get(f"adaptation_{f}")]), errors="coerce").iloc[0]
                if np.isfinite(lab):
                    ds.append(float(lab)); ws.append(1.0 / (float(d) + 0.05))
            pred_dev = float(np.average(ds, weights=ws)) if ds and ws else np.nan
            rec[f"actual_{f}"] = float(actual) if np.isfinite(actual) else np.nan
            rec[f"baseline_pred_{f}"] = float(base) if np.isfinite(base) else np.nan
            rec[f"adaptive_pred_{f}"] = float(base + pred_dev) if np.isfinite(base) and np.isfinite(pred_dev) else np.nan
            rec[f"predicted_adaptation_{f}"] = pred_dev
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(density)


def corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(x) < 3 or x.a.std(ddof=0) == 0 or x.b.std(ddof=0) == 0:
        return np.nan
    return float(x.a.corr(x.b))


def score_response(pred: pd.DataFrame, fields: list[str], source: str) -> pd.DataFrame:
    rows = []
    for f in fields:
        cols = [f"actual_{f}", f"baseline_pred_{f}", f"adaptive_pred_{f}"]
        if pred.empty or not set(cols).issubset(pred.columns):
            rows.append({"source": source, "metric": f, "n": 0})
            continue
        x = pred[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if x.empty:
            rows.append({"source": source, "metric": f, "n": 0})
            continue
        y, b, a = x[cols[0]], x[cols[1]], x[cols[2]]
        bmae = float(np.mean(np.abs(y-b))); amae = float(np.mean(np.abs(y-a)))
        brmse = float(np.sqrt(np.mean((y-b)**2))); armse = float(np.sqrt(np.mean((y-a)**2)))
        bc, ac = corr(y, b), corr(y, a)
        rows.append({
            "source": source, "metric": f, "n": len(x),
            "baseline_mae": bmae, "adaptive_mae": amae,
            "mae_gain": bmae-amae,
            "mae_gain_pct": (bmae-amae)/bmae if bmae > 0 else np.nan,
            "baseline_rmse": brmse, "adaptive_rmse": armse,
            "rmse_gain": brmse-armse,
            "baseline_corr": bc, "adaptive_corr": ac,
            "corr_gain": (ac-bc) if np.isfinite(ac) and np.isfinite(bc) else np.nan,
        })
    return pd.DataFrame(rows)


def _join_participation(p: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    p, b = lower(p), lower(b)
    candidates = [
        ("nflverse_game_id", "play_id"), ("old_game_id", "play_id"), ("game_id", "play_id")
    ]
    keys = next(([a, c] for a, c in candidates if a in p.columns and c in p.columns and a in b.columns and c in b.columns), None)
    if keys is None:
        # Common case: participation nflverse_game_id corresponds to PBP game_id.
        if {"nflverse_game_id", "play_id"}.issubset(p.columns) and {"game_id", "play_id"}.issubset(b.columns):
            p = p.copy(); p["_jg"] = p.nflverse_game_id.astype(str); p["_jp"] = pd.to_numeric(p.play_id, errors="coerce")
            b = b.copy(); b["_jg"] = b.game_id.astype(str); b["_jp"] = pd.to_numeric(b.play_id, errors="coerce")
            keys = ["_jg", "_jp"]
        else:
            raise RuntimeError("participation/PBP have no supported game+play join keys")
    right_cols = [*keys, "season", "week", "season_type", "defteam", "posteam", "rush_attempt"]
    right_cols = [c for c in right_cols if c in b.columns]
    right = b[right_cols].drop_duplicates(keys)
    return p.merge(right, on=keys, how="inner", suffixes=("", "_pbp"), validate="one_to_one")


def load_participation(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, rows = [], []
    for season in PART_SEASONS:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        try:
            p, pm = read_url_parquet(url, "m83-defense-adaptation")
            b, _ = read_url_parquet(pbp_url, "m83-defense-adaptation")
            p, b = lower(p), lower(b)
            for fld in ["defense_man_zone_type", "defense_coverage_type", "defenders_in_box"]:
                if fld not in p.columns:
                    raise RuntimeError(f"participation missing {fld}")
            j = _join_participation(p, b)
            if "season_type" in j.columns:
                j = j.loc[j.season_type.astype(str).str.upper().eq("REG")].copy()
            j = j.loc[pd.to_numeric(j.week, errors="coerce").between(1, 18)].copy()
            rows.append({
                "season": season, "rows": len(p), "joined_rows": len(j),
                "man_zone_coverage": coverage(p.defense_man_zone_type),
                "shell_coverage": coverage(p.defense_coverage_type),
                "box_coverage": coverage(p.defenders_in_box),
                "shell_categories": int(p.loc[present(p.defense_coverage_type), "defense_coverage_type"].astype(str).str.strip().nunique()),
                **pm, "status": "OK", "in_season_deployable": False,
            })
            frames.append(j)
        except Exception as exc:
            print(f"[m83_source_error] participation season={season} {type(exc).__name__}: {exc}")
            rows.append({
                "season": season, "rows": 0, "joined_rows": 0,
                "man_zone_coverage": 0.0, "shell_coverage": 0.0,
                "box_coverage": 0.0, "shell_categories": 0, "url": url,
                "bytes": 0, "sha256": "", "status": f"ERROR:{type(exc).__name__}:{exc}",
                "in_season_deployable": False,
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "m83_participation_source_audit.csv", index=False)
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), audit


def build_part_response(j: pd.DataFrame) -> pd.DataFrame:
    if j.empty:
        return pd.DataFrame()
    x = lower(j)
    mz = x.defense_man_zone_type.astype("string").str.upper().str.strip()
    x["man_flag"] = np.where(mz.str.contains("MAN", na=False) | mz.eq("M"), 1.0, np.where(mz.str.contains("ZONE", na=False) | mz.eq("Z"), 0.0, np.nan))
    x["zone_flag"] = np.where(np.isfinite(x.man_flag), 1.0-x.man_flag, np.nan)
    x["box_num"] = pd.to_numeric(x.defenders_in_box, errors="coerce")
    rush = pd.to_numeric(x.get("rush_attempt"), errors="coerce").fillna(0).eq(1)
    x["box_rush"] = x.box_num.where(rush)
    gid = "nflverse_game_id" if "nflverse_game_id" in x.columns else ("game_id" if "game_id" in x.columns else "old_game_id")
    x["game_id_norm"] = x[gid].astype(str)
    return x.loc[x.defteam.notna() & x.posteam.notna()].groupby(
        ["season", "week", "game_id_norm", "defteam", "posteam"], as_index=False
    ).agg(
        man_rate=("man_flag", "mean"),
        zone_rate=("zone_flag", "mean"),
        avg_box=("box_rush", "mean"),
    ).rename(columns={"game_id_norm": "game_id", "defteam": "defense", "posteam": "offense"})


def decide(src: pd.DataFrame, cov: pd.DataFrame, ftn_pred: pd.DataFrame, density: pd.DataFrame, ftn_metrics: pd.DataFrame, part_metrics: pd.DataFrame, part_audit: pd.DataFrame) -> dict:
    req_years = {2022, 2023, 2024}
    join_rows = src.loc[src.source.eq("FTN_PBP_JOIN") & src.season.isin(req_years)] if not src.empty else pd.DataFrame()
    joins_ok = len(join_rows) == len(req_years) and pd.to_numeric(join_rows.join_rate, errors="coerce").ge(MIN_JOIN).all()
    ftn_rows = src.loc[src.source.eq("FTN") & src.season.isin(req_years)] if not src.empty else pd.DataFrame()
    weeks_ok = len(ftn_rows) == len(req_years) and ftn_rows.weeks_1_18_complete.fillna(False).all()
    primary_raw = {"n_blitzers", "n_pass_rushers"}
    c = cov.loc[cov.season.isin(req_years) & cov.field.isin(primary_raw)] if not cov.empty else pd.DataFrame()
    cov_ok = len(c) == len(req_years) * len(primary_raw) and pd.to_numeric(c.coverage, errors="coerce").ge(MIN_FIELD_COVERAGE).all()

    den = density.copy()
    if den.empty:
        density_rate = 0.0; median_sim = np.nan
    else:
        density_rate = float(den.has_four.fillna(False).mean())
        selected = pd.to_numeric(den.loc[den.has_four.fillna(False), "mean_selected_similarity"], errors="coerce")
        median_sim = float(selected.median()) if selected.notna().any() else np.nan
    density_ok = density_rate >= MIN_DENSITY and np.isfinite(median_sim) and median_sim >= MIN_MEDIAN_SIMILARITY

    qualifying = []
    if not ftn_metrics.empty:
        for _, r in ftn_metrics.iterrows():
            g = pd.to_numeric(pd.Series([r.get("mae_gain_pct")]), errors="coerce").iloc[0]
            cg = pd.to_numeric(pd.Series([r.get("corr_gain")]), errors="coerce").iloc[0]
            if np.isfinite(g) and np.isfinite(cg) and g >= MIN_MAE_GAIN_PCT and cg >= MIN_CORR_GAIN:
                qualifying.append(str(r.metric))
    other_not_all_bad = True
    if not ftn_metrics.empty and len(ftn_metrics) > 1:
        bad = pd.to_numeric(ftn_metrics.mae_gain_pct, errors="coerce").lt(-MIN_MAE_GAIN_PCT)
        other_not_all_bad = not bool(bad.all())
    source_ok = bool(joins_ok and weeks_ok and cov_ok and density_ok)
    mech_ok = bool(source_ok and qualifying and other_not_all_bad)

    hist_part_signal = False
    if not part_metrics.empty:
        p = part_metrics.copy()
        hist_part_signal = bool(((pd.to_numeric(p.mae_gain_pct, errors="coerce") >= MIN_MAE_GAIN_PCT) & (pd.to_numeric(p.corr_gain, errors="coerce") >= MIN_CORR_GAIN)).any())
    part_source_ok = bool(not part_audit.empty and part_audit.loc[part_audit.season.isin([2023, 2024]), "status"].astype(str).eq("OK").all())

    status = "DEFENSIVE_ADAPTATION_MECHANISM_QUALIFIED" if mech_ok else ("HISTORICAL_SIGNAL_SOURCE_BLOCKED" if hist_part_signal and not mech_ok else "NO_DEFENSIVE_ADAPTATION_MECHANISM")
    return {
        "migration": "M83",
        "status": status,
        "production_actionable": False,
        "qb_outcomes_read": False,
        "sportsbook_features_used": False,
        "mechanism_scoring_season": 2024,
        "ftn_in_season_update_contract": True,
        "participation_in_season_deployable": False,
        "ftn_pbp_join_gate": bool(joins_ok),
        "ftn_weeks_gate": bool(weeks_ok),
        "ftn_primary_field_coverage_gate": bool(cov_ok),
        "comparable_density_rate": density_rate,
        "median_selected_similarity": median_sim,
        "comparable_density_gate": bool(density_ok),
        "source_contract_ok": source_ok,
        "qualifying_ftn_response_metrics": qualifying,
        "mechanism_qualified": mech_ok,
        "participation_source_history_ok": part_source_ok,
        "historical_participation_signal": hist_part_signal,
        "advance_to_m84": mech_ok,
        "m84_boundary": "2024 QB predictive development only; 2025 QB outcomes remain untouched" if mech_ok else "do not fit QB predictive model from same adaptation construction",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/backtests/m83_defensive_adaptation"))
    args = p.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)

    z, src, cov = load_ftn_pbp(out)
    off, ftn_resp = build_ftn_game_frames(z)
    profiles = build_pregame_offense_profiles(off)
    profiles.to_csv(out / "m83_pregame_offense_archetypes.csv", index=False)

    ftn_b = add_defense_baselines(ftn_resp, FTN_RESPONSE_FIELDS)
    ftn_pred, density = adaptive_predictions(ftn_b, profiles, FTN_RESPONSE_FIELDS, "FTN_DEPLOYABLE")
    ftn_metrics = score_response(ftn_pred, FTN_RESPONSE_FIELDS, "FTN_DEPLOYABLE")
    ftn_pred.to_csv(out / "m83_ftn_adaptive_response_predictions_2024.csv", index=False)
    density.to_csv(out / "m83_comparable_opponent_density_2024.csv", index=False)
    ftn_metrics.to_csv(out / "m83_ftn_response_metrics_2024.csv", index=False)

    part_joined, part_audit = load_participation(out)
    part_resp = build_part_response(part_joined)
    part_b = add_defense_baselines(part_resp, PART_RESPONSE_FIELDS)
    part_pred, part_density = adaptive_predictions(part_b, profiles, PART_RESPONSE_FIELDS, "PARTICIPATION_HISTORICAL")
    part_metrics = score_response(part_pred, PART_RESPONSE_FIELDS, "PARTICIPATION_HISTORICAL")
    part_pred.to_csv(out / "m83_participation_adaptive_response_predictions_2024.csv", index=False)
    part_density.to_csv(out / "m83_participation_density_2024.csv", index=False)
    part_metrics.to_csv(out / "m83_participation_response_metrics_2024.csv", index=False)

    decision = decide(src, cov, ftn_pred, density, ftn_metrics, part_metrics, part_audit)
    (out / "m83_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([decision]).to_csv(out / "m83_decision.csv", index=False)

    print("[m83_ftn_response_metrics]")
    print(ftn_metrics.to_string(index=False))
    print("[m83_participation_response_metrics]")
    print(part_metrics.to_string(index=False))
    print("[m83_decision]")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
