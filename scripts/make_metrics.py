# scripts/make_metrics.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- IO helpers ----------

def _read_csv(p: str) -> pd.DataFrame:
    fp = Path(p)
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _num(s, default=np.nan):
    try:
        x = float(s)
        return x
    except Exception:
        return default

def _safe(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df

def _norm_col(df: pd.DataFrame, name: str, default=np.nan):
    if name not in df.columns:
        df[name] = default
    return df

# ---------- feature constructors ----------

def weather_multipliers(w: pd.DataFrame) -> pd.DataFrame:
    """Expect optional: event_id, wind_mph, temp_f, precip, roof, surface"""
    if w.empty:
        return w
    w = w.copy()

    for c in ["wind_mph", "temp_f"]:
        w[c] = pd.to_numeric(w.get(c), errors="coerce")

    # Simple weather factors (safe, conservative)
    # wind hurts pass + receptions, mild bump to rush.
    wind = w["wind_mph"].fillna(0.0)
    w["wx_mult_pass"]      = (1.0 - 0.006 * np.clip(wind - 8.0, 0, 40)).clip(0.85, 1.02)
    w["wx_mult_rec"]       = (1.0 - 0.005 * np.clip(wind - 8.0, 0, 40)).clip(0.86, 1.02)
    w["wx_mult_rush"]      = (1.0 + 0.004 * np.clip(wind - 8.0, 0, 40)).clip(0.98, 1.06)
    w["wx_mult_rush_rec"]  = (w["wx_mult_rush"] * w["wx_mult_rec"]).clip(0.86, 1.06)
    return w

def coverage_effects(cvg: pd.DataFrame) -> pd.DataFrame:
    """Expect optional WR shadow info: event_id, player, cb_grade, is_shadow, slot_rate"""
    if cvg.empty:
        return cvg
    cvg = cvg.copy()
    _norm_col(cvg, "cb_grade", np.nan)
    _norm_col(cvg, "is_shadow", 0.0)
    _norm_col(cvg, "slot_rate", 0.0)

    cb = pd.to_numeric(cvg["cb_grade"], errors="coerce")
    is_shadow = pd.to_numeric(cvg["is_shadow"], errors="coerce").fillna(0.0)
    slot_rate = pd.to_numeric(cvg["slot_rate"],  errors="coerce").fillna(0.0)

    # Convert CB grade to small penalty (negative if elite CB)
    # Assume 80+ graded CBs impose ~6-10% yds drop if shadowed.
    penalty = -0.0015 * np.clip(cb - 70.0, 0, 30) * is_shadow
    boost   =  0.12 * slot_rate  # slot usage vs zone → mild boost

    cvg["wr_shadow_penalty"] = penalty.clip(-0.12, 0.0)
    cvg["slot_zone_boost"]   = boost.clip(-0.02, 0.15)
    return cvg

def team_to_pressure(team: pd.DataFrame) -> pd.DataFrame:
    """Yield qb_pressure_mult, sack_elasticity, funnels, proe_z from team defense/offense."""
    if team.empty:
        return team
    team = team.copy()

    # Normalize expected cols if present
    for c in ["pressure_rate_def","def_sack_rate","proe_z","run_funnel","pass_funnel"]:
        _norm_col(team, c, np.nan)

    team["pressure_rate_def"] = pd.to_numeric(team["pressure_rate_def"], errors="coerce")
    team["def_sack_rate"]     = pd.to_numeric(team["def_sack_rate"],     errors="coerce")
    team["proe_z"]            = pd.to_numeric(team["proe_z"],            errors="coerce").fillna(0.0)
    team["run_funnel"]        = pd.to_numeric(team["run_funnel"],        errors="coerce").fillna(0.0)
    team["pass_funnel"]       = pd.to_numeric(team["pass_funnel"],       errors="coerce").fillna(0.0)

    # Multipliers relative to league average
    pr = team["pressure_rate_def"].fillna(team["pressure_rate_def"].median(skipna=True))
    sr = team["def_sack_rate"].fillna(team["def_sack_rate"].median(skipna=True))
    pr_lg = np.nanmean(pr) if np.isfinite(np.nanmean(pr)) else 0.28
    sr_lg = np.nanmean(sr) if np.isfinite(np.nanmean(sr)) else 0.07

    team["qb_pressure_mult"] = (1.0 - 0.80 * (pr - pr_lg)).clip(0.82, 1.10)
    team["sack_elasticity"]  = (1.0 - 0.60 * (sr - sr_lg)).clip(0.85, 1.10)
    return team

def role_priors(pf: pd.DataFrame) -> pd.DataFrame:
    """Create safe role priors: routes, yprr, targets_proj, rush_att_proj, ypc, qb_ypa, qb_att_proj."""
    if pf.empty:
        return pf
    pf = pf.copy()
    for c in ["routes","yprr","targets_proj","rush_att_proj","ypc","qb_ypa","qb_att_proj","target_share"]:
        _norm_col(pf, c, np.nan)
        pf[c] = pd.to_numeric(pf[c], errors="coerce")

    return pf

def injuries_volatility(inj: pd.DataFrame) -> pd.DataFrame:
    """Create rb_attempts_escalator and volatility_flag from injuries depth info if present."""
    if inj.empty:
        return inj
    inj = inj.copy()
    # Try to infer if primary backup RB is out → bump attempts for starter
    _norm_col(inj, "player", "")
    _norm_col(inj, "status", "")
    _norm_col(inj, "position", "")
    inj["is_rb_backup_out"] = ((inj["position"].str.upper() == "RB") &
                               (inj["status"].str.lower().isin(["out","doubtful"]))).astype(float)
    # Aggregate per (event_id, team, starter?) — if you have it
    depth = inj.groupby(["event_id","team"], as_index=False)["is_rb_backup_out"].sum()
    depth["rb_attempts_escalator"] = (depth["is_rb_backup_out"] * 2.0).clip(0, 8)  # add 0–8 attempts ceiling
    depth["volatility_flag"] = (inj["status"].str.lower().isin(["questionable","doubtful"])).groupby(
        [inj.get("event_id", pd.Series(index=inj.index)), inj.get("team", pd.Series(index=inj.index))]
    ).transform("max").fillna(0).astype(float)
    depth = depth.drop(columns=["is_rb_backup_out"], errors="ignore")
    depth = depth.drop_duplicates(subset=["event_id","team"])
    return depth

def sgp_links(pf: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple SGP hooks:
      - qb_wr_pair: qb_player string for a WR/TE
      - sgp_qb_wr_boost: tilt factor ~ f(target_share, proe_z)
    """
    if pf.empty:
        return pf

    pf = pf.copy()
    _norm_col(pf, "position", "")
    _norm_col(pf, "target_share", np.nan)

    # derive qb per team/event if present in team_form
    qb_map = {}
    if not tf.empty:
        if "event_id" in tf.columns and "team" in tf.columns and "qb_player" in tf.columns:
            qb_map = tf.set_index(["event_id","team"])["qb_player"].to_dict()

    pf["qb_wr_pair"] = np.where(
        pf["position"].str.upper().isin(["WR","TE"]),
        pf.apply(lambda r: qb_map.get((r.get("event_id"), r.get("team"))), axis=1),
        np.nan
    )

    ts = pd.to_numeric(pf.get("target_share"), errors="coerce").fillna(0.0)
    proe = pd.to_numeric(pf.get("proe_z"), errors="coerce").fillna(0.0)
    pf["sgp_qb_wr_boost"] = (0.35 * ts + 0.12 * proe).clip(0.0, 0.25)
    return pf

# ---------- build pipeline ----------

def build_metrics(season: int) -> pd.DataFrame:
    PATH = Path("data")

    team_form   = _read_csv(PATH / "team_form.csv")
    player_form = _read_csv(PATH / "player_form.csv")
    injuries    = _read_csv(PATH / "injuries.csv")
    coverage    = _read_csv(PATH / "coverage.csv")
    weather     = _read_csv(PATH / "weather.csv")

    # Normalize keys we try to join on
    for df in (team_form, player_form, coverage, weather):
        if not df.empty:
            for k in ["event_id","team","player","date"]:
                _norm_col(df, k, np.nan)

    tf_feats  = team_to_pressure(team_form) if not team_form.empty else pd.DataFrame()
    pf_roles  = role_priors(player_form)    if not player_form.empty else pd.DataFrame()
    inj_depth = injuries_volatility(injuries) if not injuries.empty else pd.DataFrame()
    wx        = weather_multipliers(weather)  if not weather.empty else pd.DataFrame()
    cvg       = coverage_effects(coverage)    if not coverage.empty else pd.DataFrame()

    # Merge player_form with team pressure/funnel features
    if not pf_roles.empty and not tf_feats.empty:
        m = pf_roles.merge(
            tf_feats[["event_id","team","qb_pressure_mult","sack_elasticity","proe_z","run_funnel","pass_funnel"]],
            on=["event_id","team"], how="left"
        )
    else:
        m = pf_roles.copy() if not pf_roles.empty else pd.DataFrame()

    # join SGP links
    m = sgp_links(m, team_form) if not m.empty else m

    # join coverage (by event_id + player)
    if not m.empty and not cvg.empty:
        m = m.merge(cvg[["event_id","player","wr_shadow_penalty","slot_zone_boost"]],
                    on=["event_id","player"], how="left")

    # join injuries depth → escalator + volatility
    if not m.empty and not inj_depth.empty:
        m = m.merge(inj_depth[["event_id","team","rb_attempts_escalator","volatility_flag"]],
                    on=["event_id","team"], how="left")

    # join weather multipliers
    if not m.empty and not wx.empty:
        m = m.merge(wx[["event_id","wx_mult_pass","wx_mult_rec","wx_mult_rush","wx_mult_rush_rec"]],
                    on="event_id", how="left")

    # Ensure all columns exist (pricing is fail-safe if any are missing)
    needed = [
        "qb_pressure_mult","sack_elasticity","proe_z","run_funnel","pass_funnel",
        "wx_mult_pass","wx_mult_rec","wx_mult_rush","wx_mult_rush_rec",
        "wr_shadow_penalty","slot_zone_boost","rb_attempts_escalator","volatility_flag",
        "routes","yprr","targets_proj","rush_att_proj","ypc","qb_ypa","qb_att_proj",
        "qb_wr_pair","sgp_qb_wr_boost",
    ]
    for c in needed:
        _norm_col(m, c, np.nan)

    # Fill gentle defaults
    m["qb_pressure_mult"] = m["qb_pressure_mult"].fillna(1.0).clip(0.8, 1.2)
    m["sack_elasticity"]  = m["sack_elasticity"].fillna(1.0).clip(0.8, 1.2)
    for c in ["proe_z","run_funnel","pass_funnel","wr_shadow_penalty","slot_zone_boost",
              "rb_attempts_escalator","volatility_flag","sgp_qb_wr_boost"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0.0)

    for c in ["wx_mult_pass","wx_mult_rec","wx_mult_rush","wx_mult_rush_rec"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(1.0).clip(0.80, 1.10)

    # Output
    outp = Path("data") / "metrics_ready.csv"
    m.to_csv(outp, index=False)
    print(f"[metrics] wrote {outp} rows={len(m)}")
    return m

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()
    build_metrics(args.season)

if __name__ == "__main__":
    cli()
