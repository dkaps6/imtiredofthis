#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team


REG_METRICS = [
    "drives",
    "plays_per_drive",
    "dropback_rate",
    "attempt_conversion",
    "neutral_dropback_rate",
    "trailing_dropback_rate",
    "leading_dropback_rate",
    "no_huddle_rate",
    "seconds_between_plays",
    "scoring_drive_rate",
]

CANDIDATES = ["raw", "generative_neutral", "generative_gamescript"]


def num(v):
    return pd.to_numeric(v, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def to_pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def first_col(frame: pd.DataFrame, names: list[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = to_pd(nfl.load_pbp(sorted(set(int(s) for s in seasons))))
    if raw.empty:
        raise RuntimeError("nflreadpy returned zero play-by-play rows")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    return raw


def prepare_pbp(raw: pd.DataFrame) -> pd.DataFrame:
    p = raw.copy()
    required = {"season", "week", "game_id", "play_id", "posteam", "defteam"}
    missing = required - set(p.columns)
    if missing:
        raise RuntimeError(f"play-by-play missing required columns: {sorted(missing)}")

    p["season"] = num(p["season"]).astype("Int64")
    p["week"] = num(p["week"]).astype("Int64")
    p = p[p["season"].notna() & p["week"].between(1, 18)].copy()
    p["posteam"] = p["posteam"].map(canon_team)
    p["defteam"] = p["defteam"].map(canon_team)
    p = p[p["posteam"].astype(str).ne("") & p["defteam"].astype(str).ne("")].copy()

    pass_attempt = num(first_col(p, ["pass_attempt"], 0)).fillna(0).gt(0)
    sack = num(first_col(p, ["sack"], 0)).fillna(0).gt(0)
    scramble = num(first_col(p, ["qb_scramble"], 0)).fillna(0).gt(0)
    if "qb_dropback" in p.columns:
        dropback = num(p["qb_dropback"]).fillna(0).gt(0)
    else:
        dropback = pass_attempt | sack | scramble

    rush = num(first_col(p, ["rush_attempt"], 0)).fillna(0).gt(0)
    # Scrambles are dropbacks even when the source also marks them as rush attempts.
    designed_rush = rush & ~scramble
    scrimmage = dropback | designed_rush

    p["_pass_attempt"] = pass_attempt.astype(int)
    p["_dropback"] = dropback.astype(int)
    p["_designed_rush"] = designed_rush.astype(int)
    p["_scrimmage"] = scrimmage.astype(int)
    p["_no_huddle"] = num(first_col(p, ["no_huddle"], 0)).fillna(0).gt(0).astype(int)

    score_diff = num(first_col(p, ["score_differential"], np.nan))
    p["_neutral"] = score_diff.between(-7, 7, inclusive="both")
    p["_trailing"] = score_diff.le(-8)
    p["_leading"] = score_diff.ge(8)

    drive_col = None
    for c in ("drive", "fixed_drive", "drive_id"):
        if c in p.columns:
            drive_col = c
            break
    if drive_col is None:
        raise RuntimeError("play-by-play missing drive identifier")
    p["_drive"] = p[drive_col].astype(str)

    # Drive-scoring indicator. Prefer nflfastR's drive-level flag when present.
    if "drive_ended_with_score" in p.columns:
        p["_score_play"] = num(p["drive_ended_with_score"]).fillna(0).gt(0).astype(int)
    else:
        td = num(first_col(p, ["touchdown"], 0)).fillna(0).gt(0)
        fg = first_col(p, ["field_goal_result"], "").astype(str).str.lower().eq("made")
        p["_score_play"] = (td | fg).astype(int)

    # Seconds between same-offense scrimmage plays within a drive; extreme gaps are excluded.
    p["_game_seconds"] = num(first_col(p, ["game_seconds_remaining"], np.nan))
    s = p[p["_scrimmage"].eq(1)].sort_values(
        ["season", "game_id", "posteam", "_drive", "_game_seconds"],
        ascending=[True, True, True, True, False],
    ).copy()
    s["_seconds_between"] = (
        s.groupby(["season", "game_id", "posteam", "_drive"], dropna=False)["_game_seconds"]
        .diff(-1)
    )
    s.loc[~s["_seconds_between"].between(5, 60), "_seconds_between"] = np.nan
    p = p.merge(
        s[["season", "game_id", "play_id", "_seconds_between"]],
        on=["season", "game_id", "play_id"],
        how="left",
    )

    return p


def summarize_team_games(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["season", "week", "game_id", "posteam", "defteam"]
    for key, g in p.groupby(keys, dropna=False, sort=False):
        season, week, game_id, team, opp = key
        scr = g[g["_scrimmage"].eq(1)].copy()
        if scr.empty:
            continue
        drives = int(scr["_drive"].nunique())
        plays = int(scr["_scrimmage"].sum())
        db = int(scr["_dropback"].sum())
        pa = int(scr["_pass_attempt"].sum())

        def rate(mask: pd.Series) -> float:
            q = scr[mask]
            den = int(q["_scrimmage"].sum())
            return float(q["_dropback"].sum() / den) if den else np.nan

        drive_score = (
            g.groupby("_drive", dropna=False)["_score_play"].max()
            if drives
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "game_id": str(game_id),
                "team": canon_team(team),
                "opponent": canon_team(opp),
                "drives": float(drives),
                "scrimmage_plays": float(plays),
                "plays_per_drive": float(plays / drives) if drives else np.nan,
                "dropbacks": float(db),
                "pass_attempts": float(pa),
                "dropback_rate": float(db / plays) if plays else np.nan,
                "attempt_conversion": float(pa / db) if db else np.nan,
                "neutral_dropback_rate": rate(scr["_neutral"]),
                "trailing_dropback_rate": rate(scr["_trailing"]),
                "leading_dropback_rate": rate(scr["_leading"]),
                "no_huddle_rate": float(scr["_no_huddle"].mean()),
                "seconds_between_plays": float(num(scr["_seconds_between"]).mean()),
                "scoring_drive_rate": float(num(drive_score).mean()) if len(drive_score) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("play-by-play team-game summarization produced zero rows")
    if out.duplicated(["season", "week", "team"]).any():
        dup = out[out.duplicated(["season", "week", "team"], keep=False)].head(10)
        raise RuntimeError(f"duplicate team-game PBP rows: {dup.to_dict(orient='records')}")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def add_defensive_views(off: pd.DataFrame) -> pd.DataFrame:
    d = off[
        ["season", "week", "game_id", "team", "opponent"] + REG_METRICS
    ].copy()
    d = d.rename(columns={"team": "offense", "opponent": "team"})
    d = d.rename(columns={m: f"allowed_{m}" for m in REG_METRICS})
    return d


def before(frame: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    s = num(frame["season"])
    w = num(frame["week"])
    z = frame[
        frame["team"].astype(str).eq(canon_team(team))
        & ((s < int(season)) | ((s == int(season)) & (w < int(week))))
    ].copy()
    return z.sort_values(["season", "week"]).tail(8)


def league_before(frame: pd.DataFrame, season: int, week: int, col: str) -> float:
    s = num(frame["season"])
    w = num(frame["week"])
    z = num(frame.loc[(s < int(season)) | ((s == int(season)) & (w < int(week))), col]).dropna()
    return float(z.mean()) if len(z) else np.nan


def shrunk_recent(
    frame: pd.DataFrame,
    season: int,
    week: int,
    team: str,
    col: str,
    *,
    prior_games: float = 4.0,
) -> float:
    hist = num(before(frame, season, week, team)[col]).dropna()
    lg = league_before(frame, season, week, col)
    if not np.isfinite(lg):
        return float(hist.mean()) if len(hist) else np.nan
    if not len(hist):
        return lg
    return float((hist.sum() + prior_games * lg) / (len(hist) + prior_games))


def player_qb_share(
    logs: pd.DataFrame,
    season: int,
    week: int,
    team: str,
    player_key: str,
) -> float:
    p = logs.copy()
    s = num(p["season"])
    w = num(p["week"])
    z = p[
        p["team"].astype(str).map(canon_team).eq(canon_team(team))
        & p["player_clean_key"].astype(str).eq(str(player_key))
        & ((s < int(season)) | ((s == int(season)) & (w < int(week))))
    ].sort_values(["season", "week"]).tail(8)
    if z.empty:
        return 0.97
    pa = num(z["pass_att"])
    team_pa = num(z["team_dropbacks"]) if "team_dropbacks" in z.columns else pd.Series(np.nan, index=z.index)
    share = (pa / team_pa.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if share.empty:
        return 0.97
    # Fixed starter prior. The stable-QB universe should remain close to 1.0.
    return float(np.clip((share.sum() + 4.0 * 0.97) / (len(share) + 4.0), 0.75, 1.0))


def implied_win_prob_from_spread(spread: float) -> float:
    if not np.isfinite(spread):
        return 0.5
    # Positive spread means the team is an underdog in this project.
    return float(1.0 / (1.0 + np.exp(float(spread) / 6.5)))


def blend(a: float, b: float, fallback: float) -> float:
    vals = [v for v in (a, b) if np.isfinite(v)]
    if vals:
        return float(np.mean(vals))
    return float(fallback)


def actual_team_game(off: pd.DataFrame, season: int, week: int, team: str) -> pd.Series:
    q = off[
        num(off["season"]).eq(int(season))
        & num(off["week"]).eq(int(week))
        & off["team"].astype(str).eq(canon_team(team))
    ]
    return q.iloc[0] if len(q) else pd.Series(dtype=object)


def metric_triplet(actual, pred):
    a = num(actual)
    p = num(pred)
    z = pd.DataFrame({"a": a, "p": p}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = z.p - z.a
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z.a.corr(z.p)) if len(z) >= 2 else np.nan,
    }


def build_rows(games: pd.DataFrame, off: pd.DataFrame, defense: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in games.iterrows():
        season = int(r["season"])
        week = int(r["week"])
        team = canon_team(r["team"])
        opp = canon_team(r["opponent"])
        key = str(r["player_clean_key"])

        # League fallbacks are calculated strictly from games before the target week.
        lg = {m: league_before(off, season, week, m) for m in REG_METRICS}

        team_off = {m: shrunk_recent(off, season, week, team, m) for m in REG_METRICS}
        opp_off = {m: shrunk_recent(off, season, week, opp, m) for m in REG_METRICS}
        team_def = {
            m: shrunk_recent(defense, season, week, team, f"allowed_{m}")
            for m in REG_METRICS
        }
        opp_def = {
            m: shrunk_recent(defense, season, week, opp, f"allowed_{m}")
            for m in REG_METRICS
        }

        # Possession estimate is intentionally symmetric: both offenses and both defenses
        # contribute to the number of possessions available in the game.
        drive_parts = np.asarray([
            team_off["drives"],
            opp_off["drives"],
            team_def["drives"],
            opp_def["drives"],
        ], dtype=float)
        pred_drives = float(np.nanmean(drive_parts)) if np.isfinite(drive_parts).any() else lg["drives"]

        pred_ppd = blend(team_off["plays_per_drive"], opp_def["plays_per_drive"], lg["plays_per_drive"])
        base_dbr = blend(team_off["dropback_rate"], opp_def["dropback_rate"], lg["dropback_rate"])
        neutral_dbr = blend(
            team_off["neutral_dropback_rate"],
            opp_def["neutral_dropback_rate"],
            base_dbr,
        )
        trailing_dbr = blend(
            team_off["trailing_dropback_rate"],
            opp_def["trailing_dropback_rate"],
            base_dbr,
        )
        leading_dbr = blend(
            team_off["leading_dropback_rate"],
            opp_def["leading_dropback_rate"],
            base_dbr,
        )
        pred_conversion = blend(
            team_off["attempt_conversion"],
            opp_def["attempt_conversion"],
            lg["attempt_conversion"],
        )
        qb_share = player_qb_share(logs, season, week, team, key)

        spread = float(num(pd.Series([r.get("market_spread", np.nan)])).iloc[0])
        win_prob = implied_win_prob_from_spread(spread)
        neutral_weight = 0.35
        lead_weight = (1.0 - neutral_weight) * win_prob
        trail_weight = (1.0 - neutral_weight) * (1.0 - win_prob)
        gamescript_dbr = (
            neutral_weight * neutral_dbr
            + lead_weight * leading_dbr
            + trail_weight * trailing_dbr
        )

        # Broad physical bounds protect data errors without recreating M57/M61-style
        # aggressive prediction compression.
        pred_drives = float(np.clip(pred_drives, 7.0, 17.0))
        pred_ppd = float(np.clip(pred_ppd, 3.5, 9.0))
        base_dbr = float(np.clip(base_dbr, 0.30, 0.85))
        gamescript_dbr = float(np.clip(gamescript_dbr, 0.30, 0.85))
        pred_conversion = float(np.clip(pred_conversion, 0.75, 1.00))

        gen_neutral = pred_drives * pred_ppd * base_dbr * pred_conversion * qb_share
        gen_gamescript = pred_drives * pred_ppd * gamescript_dbr * pred_conversion * qb_share

        actual = actual_team_game(off, season, week, team)
        actual_drives = float(num(pd.Series([actual.get("drives", np.nan)])).iloc[0])
        actual_ppd = float(num(pd.Series([actual.get("plays_per_drive", np.nan)])).iloc[0])
        actual_dbr = float(num(pd.Series([actual.get("dropback_rate", np.nan)])).iloc[0])
        actual_conv = float(num(pd.Series([actual.get("attempt_conversion", np.nan)])).iloc[0])
        actual_team_pa = float(num(pd.Series([actual.get("pass_attempts", np.nan)])).iloc[0])
        actual_qb_pa = float(num(pd.Series([r.get("actual_pass_att", np.nan)])).iloc[0])
        actual_qb_share = (
            actual_qb_pa / actual_team_pa
            if np.isfinite(actual_qb_pa) and np.isfinite(actual_team_pa) and actual_team_pa > 0
            else np.nan
        )

        oracle_all = (
            actual_drives * actual_ppd * actual_dbr * actual_conv * actual_qb_share
            if all(np.isfinite(v) for v in [actual_drives, actual_ppd, actual_dbr, actual_conv, actual_qb_share])
            else np.nan
        )
        oracle_drives = (
            actual_drives * pred_ppd * gamescript_dbr * pred_conversion * qb_share
            if np.isfinite(actual_drives)
            else np.nan
        )
        oracle_ppd = (
            pred_drives * actual_ppd * gamescript_dbr * pred_conversion * qb_share
            if np.isfinite(actual_ppd)
            else np.nan
        )
        oracle_dbr = (
            pred_drives * pred_ppd * actual_dbr * pred_conversion * qb_share
            if np.isfinite(actual_dbr)
            else np.nan
        )
        oracle_conversion = (
            pred_drives * pred_ppd * gamescript_dbr * actual_conv * qb_share
            if np.isfinite(actual_conv)
            else np.nan
        )
        oracle_share = (
            pred_drives * pred_ppd * gamescript_dbr * pred_conversion * actual_qb_share
            if np.isfinite(actual_qb_share)
            else np.nan
        )

        ypa = float(num(pd.Series([r.get("ypa_contextual", np.nan)])).iloc[0])
        raw_attempts = float(num(pd.Series([r.get("attempts_raw", np.nan)])).iloc[0])
        raw_pass = float(num(pd.Series([r.get("mc_proj_attempts_raw_only", np.nan)])).iloc[0])

        row = r.to_dict()
        row.update(
            {
                "m64_pred_drives": pred_drives,
                "m64_pred_plays_per_drive": pred_ppd,
                "m64_pred_dropback_rate_neutral": base_dbr,
                "m64_pred_dropback_rate_gamescript": gamescript_dbr,
                "m64_pred_attempt_conversion": pred_conversion,
                "m64_pred_qb_attempt_share": qb_share,
                "m64_market_implied_win_prob": win_prob,
                "m64_attempts_generative_neutral": gen_neutral,
                "m64_attempts_generative_gamescript": gen_gamescript,
                "m64_pass_generative_neutral": gen_neutral * ypa,
                "m64_pass_generative_gamescript": gen_gamescript * ypa,
                "m64_pass_raw_point_product": raw_attempts * ypa,
                "m64_pass_raw_reference": raw_pass,
                "m64_actual_team_drives": actual_drives,
                "m64_actual_plays_per_drive": actual_ppd,
                "m64_actual_dropback_rate": actual_dbr,
                "m64_actual_attempt_conversion": actual_conv,
                "m64_actual_team_pass_attempts": actual_team_pa,
                "m64_actual_qb_attempt_share": actual_qb_share,
                "m64_oracle_all_components_attempts": oracle_all,
                "m64_oracle_drives_attempts": oracle_drives,
                "m64_oracle_plays_per_drive_attempts": oracle_ppd,
                "m64_oracle_dropback_rate_attempts": oracle_dbr,
                "m64_oracle_attempt_conversion_attempts": oracle_conversion,
                "m64_oracle_qb_share_attempts": oracle_share,
                "m64_team_recent_drives": team_off["drives"],
                "m64_opp_recent_drives": opp_off["drives"],
                "m64_team_def_drives_allowed": team_def["drives"],
                "m64_opp_def_drives_allowed": opp_def["drives"],
                "m64_team_recent_no_huddle": team_off["no_huddle_rate"],
                "m64_team_recent_seconds_between_plays": team_off["seconds_between_plays"],
                "m64_opp_recent_scoring_drive_rate": opp_off["scoring_drive_rate"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluation_rows(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts_rows = []
    pass_rows = []
    groups = [("combined", g)] + [(str(s), z) for s, z in g.groupby("season")]
    for season_label, q in groups:
        actual_att = num(q["actual_pass_att"])
        actual_pass = num(q["actual"])
        att_preds = {
            "raw": num(q["attempts_raw"]),
            "generative_neutral": num(q["m64_attempts_generative_neutral"]),
            "generative_gamescript": num(q["m64_attempts_generative_gamescript"]),
        }
        pass_preds = {
            "raw": num(q["m64_pass_raw_reference"]),
            "raw_point_product": num(q["m64_pass_raw_point_product"]),
            "generative_neutral": num(q["m64_pass_generative_neutral"]),
            "generative_gamescript": num(q["m64_pass_generative_gamescript"]),
        }
        for name, pred in att_preds.items():
            m = metric_triplet(actual_att, pred)
            err = (pred - actual_att).abs()
            forty = actual_att.ge(40)
            attempts_rows.append(
                {
                    "season": season_label,
                    "candidate": name,
                    **m,
                    "miss_8plus": int(err.ge(8).sum()),
                    "miss_10plus": int(err.ge(10).sum()),
                    "actual_40plus_n": int(forty.sum()),
                    "actual_40plus_mae": float(err[forty].mean()) if forty.any() else np.nan,
                    "actual_40plus_under_8plus": int(((actual_att - pred).ge(8) & forty).sum()),
                }
            )
        for name, pred in pass_preds.items():
            m = metric_triplet(actual_pass, pred)
            err = (pred - actual_pass).abs()
            pass_rows.append(
                {
                    "season": season_label,
                    "candidate": name,
                    **m,
                    "miss_75plus": int(err.ge(75).sum()),
                    "miss_100plus": int(err.ge(100).sum()),
                    "under_100plus": int((pred - actual_pass).le(-100).sum()),
                    "over_100plus": int((pred - actual_pass).ge(100).sum()),
                }
            )
    return pd.DataFrame(attempts_rows), pd.DataFrame(pass_rows)


def weekly_wins(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, q in g.groupby("season"):
        for week, w in q.groupby("week"):
            raw_att = (num(w["attempts_raw"]) - num(w["actual_pass_att"])).abs().mean()
            gen_att = (num(w["m64_attempts_generative_gamescript"]) - num(w["actual_pass_att"])).abs().mean()
            raw_pass = (num(w["m64_pass_raw_reference"]) - num(w["actual"])).abs().mean()
            gen_pass = (num(w["m64_pass_generative_gamescript"]) - num(w["actual"])).abs().mean()
            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "n": len(w),
                    "raw_attempt_mae": raw_att,
                    "gen_attempt_mae": gen_att,
                    "attempt_gain": raw_att - gen_att,
                    "raw_pass_mae": raw_pass,
                    "gen_pass_mae": gen_pass,
                    "pass_gain": raw_pass - gen_pass,
                    "attempt_win": bool(gen_att < raw_att),
                    "pass_win": bool(gen_pass < raw_pass),
                }
            )
    return pd.DataFrame(rows)


def oracle_rows(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = num(g["actual_pass_att"])
    cols = {
        "generative_gamescript": "m64_attempts_generative_gamescript",
        "oracle_actual_drives": "m64_oracle_drives_attempts",
        "oracle_actual_plays_per_drive": "m64_oracle_plays_per_drive_attempts",
        "oracle_actual_dropback_rate": "m64_oracle_dropback_rate_attempts",
        "oracle_actual_attempt_conversion": "m64_oracle_attempt_conversion_attempts",
        "oracle_actual_qb_share": "m64_oracle_qb_share_attempts",
        "oracle_all_components": "m64_oracle_all_components_attempts",
    }
    for label, col in cols.items():
        rows.append({"component": label, **metric_triplet(actual, num(g[col]))})
    return pd.DataFrame(rows)


def frozen_verdict(att: pd.DataFrame, pas: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    a = att[(att["season"].eq("combined")) & att["candidate"].eq("generative_gamescript")].iloc[0]
    ar = att[(att["season"].eq("combined")) & att["candidate"].eq("raw")].iloc[0]
    p = pas[(pas["season"].eq("combined")) & pas["candidate"].eq("generative_gamescript")].iloc[0]
    pr = pas[(pas["season"].eq("combined")) & pas["candidate"].eq("raw")].iloc[0]

    year_att_nonworse = True
    year_pass_nonworse = True
    for season in ("2024", "2025"):
        ag = att[(att.season.eq(season)) & att.candidate.eq("generative_gamescript")].iloc[0]
        ab = att[(att.season.eq(season)) & att.candidate.eq("raw")].iloc[0]
        pg = pas[(pas.season.eq(season)) & pas.candidate.eq("generative_gamescript")].iloc[0]
        pb = pas[(pas.season.eq(season)) & pas.candidate.eq("raw")].iloc[0]
        year_att_nonworse &= bool(ag.mae <= ab.mae + 1e-12)
        year_pass_nonworse &= bool(pg.mae <= pb.mae + 1e-12)

    gates = {
        "attempt_mae_gain_ge_0_40": float(ar.mae - a.mae) >= 0.40,
        "attempt_mae_nonworse_both_years": year_att_nonworse,
        "attempt_corr_gain_ge_0_03": float(a.corr - ar.corr) >= 0.03,
        "attempt_10plus_misses_reduce_10pct": int(a.miss_10plus) <= int(np.floor(ar.miss_10plus * 0.90)),
        "actual_40plus_attempt_mae_gain_ge_0_75": float(ar.actual_40plus_mae - a.actual_40plus_mae) >= 0.75,
        "pass_mae_gain_ge_1_50": float(pr.mae - p.mae) >= 1.50,
        "pass_mae_nonworse_both_years": year_pass_nonworse,
        "pass_corr_gain_ge_0_03": float(p.corr - pr.corr) >= 0.03,
        "pass_100plus_misses_reduce_10pct": int(p.miss_100plus) <= int(np.floor(pr.miss_100plus * 0.90)),
    }
    n_week_wins = int(weeks["pass_win"].sum())
    total_weeks = int(len(weeks))
    all_pass = bool(all(gates.values()))
    return pd.DataFrame(
        [
            {
                **gates,
                "weekly_pass_wins": n_week_wins,
                "weekly_total": total_weeks,
                "raw_attempt_mae": ar.mae,
                "gen_attempt_mae": a.mae,
                "attempt_mae_gain": ar.mae - a.mae,
                "raw_attempt_corr": ar.corr,
                "gen_attempt_corr": a.corr,
                "attempt_corr_gain": a.corr - ar.corr,
                "raw_pass_mae": pr.mae,
                "gen_pass_mae": p.mae,
                "pass_mae_gain": pr.mae - p.mae,
                "raw_pass_corr": pr.corr,
                "gen_pass_corr": p.corr,
                "pass_corr_gain": p.corr - pr.corr,
                "raw_100plus": int(pr.miss_100plus),
                "gen_100plus": int(p.miss_100plus),
                "m64_architecture_actionable": all_pass,
                "interpretation": "eligible_for_next_stage" if all_pass else "hold_architecture",
            }
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-file", action="append", type=Path, required=True)
    ap.add_argument("--player-logs", action="append", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    games = pd.concat([read(p) for p in args.season_file], ignore_index=True, sort=False)
    games["season"] = num(games["season"]).astype(int)
    games["week"] = num(games["week"]).astype(int)
    games["team"] = games["team"].map(canon_team)
    games["opponent"] = games["opponent"].map(canon_team)
    needed = {
        "season", "week", "team", "opponent", "player_clean_key", "actual_pass_att",
        "attempts_raw", "ypa_contextual", "actual", "mc_proj_attempts_raw_only",
    }
    missing = needed - set(games.columns)
    if missing:
        raise RuntimeError(f"M64 target rows missing columns: {sorted(missing)}")

    logs = pd.concat([read(p) for p in args.player_logs], ignore_index=True, sort=False)
    logs = logs.drop_duplicates(["season", "week", "team", "player_clean_key"]).copy()
    logs["team"] = logs["team"].map(canon_team)

    seasons = sorted(set(games["season"].astype(int)))
    pbp = prepare_pbp(load_pbp([min(seasons) - 1, *seasons]))
    off = summarize_team_games(pbp)
    defense = add_defensive_views(off)

    out = build_rows(games, off, defense, logs)
    att, pas = evaluation_rows(out)
    weeks = weekly_wins(out)
    oracle = oracle_rows(out)
    verdict = frozen_verdict(att, pas, weeks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "m64_game_level.csv", index=False)
    att.to_csv(args.out_dir / "m64_attempt_metrics.csv", index=False)
    pas.to_csv(args.out_dir / "m64_passing_metrics.csv", index=False)
    weeks.to_csv(args.out_dir / "m64_weekly_paired.csv", index=False)
    oracle.to_csv(args.out_dir / "m64_component_oracles.csv", index=False)
    verdict.to_csv(args.out_dir / "m64_precommitted_interpretation.csv", index=False)

    print("=== M64 FROZEN VERDICT ===")
    print(verdict.to_string(index=False))
    print("\n=== M64 ATTEMPT METRICS ===")
    print(att.to_string(index=False))
    print("\n=== M64 PASSING METRICS ===")
    print(pas.to_string(index=False))
    print("\n=== M64 COMPONENT ORACLES ===")
    print(oracle.to_string(index=False))
    print("\n=== M64 WEEKLY WINS ===")
    print(weeks.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
