#!/usr/bin/env python3
"""RB-ND1: reverse-engineer 2025 RB rushing-yard misses.

Postgame variables in this file are forensic diagnostics only. They are never
candidate features and may not be consumed by production/pregame projection code.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAX": "JAC", "ARZ": "ARI", "WSH": "WAS"}


def to_pd(v):
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def canon_team(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    return TEAM_MAP.get(s, s) if s and s not in {"NAN", "NONE", "<NA>"} else ""


def short_key(v) -> str:
    s = str(v or "").strip().lower()
    if not s or s in {"nan", "none", "<na>"}:
        return ""
    # nflfastR commonly uses J.Conner / B.Robinson style labels.
    if "." in s:
        parts = [re.sub(r"[^a-z]", "", p) for p in s.split(".") if re.sub(r"[^a-z]", "", p)]
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1]).lower()
    words = re.findall(r"[a-z]+", s)
    if not words:
        return ""
    suffix = {"jr", "sr", "ii", "iii", "iv", "v"}
    if len(words) >= 3 and words[-1] in suffix:
        words = words[:-1]
    if len(words) >= 2:
        return words[0][0] + words[-1]
    return words[0]


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def load_m94c(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = lower(pd.read_csv(find_one(root, "m94c_2025_rb_trace.csv"), low_memory=False))
    team = lower(pd.read_csv(find_one(root, "m94c_2025_team_trace.csv"), low_memory=False))
    rb = rb.loc[num(rb, "season").eq(2025)].copy()
    rb = rb.loc[rb["position"].astype(str).str.upper().isin(["RB", "FB"])].copy()
    rb["team"] = rb["team"].map(canon_team)
    rb["short_key"] = rb["player"].map(short_key)
    for c in ["week", "actual_rush_att", "actual_team", "candidate_team_rush_att", "base_team_share",
              "candidate_rush_att", "actual_team_share", "actual_rush_yards", "candidate_rush_yards", "base_rush_att"]:
        rb[c] = num(rb, c)
    team["team"] = team["team"].map(canon_team)
    team["week"] = num(team, "week")
    keep = [c for c in ["week", "team", "actual_team_rush_att", "candidate_team_rush_att", "pred_mean_margin",
                         "pred_final_margin", "pred_lead_play_share", "pred_neutral_play_share", "pred_trail_play_share",
                         "pred_off_plays", "mean_score_diff", "final_observed_score_diff", "lead_play_share",
                         "neutral_play_share", "trail_play_share", "actual_off_plays", "actual_rush_att_pbp"] if c in team.columns]
    return rb.reset_index(drop=True), team[keep].drop_duplicates(["week", "team"])


def load_market(root: Path) -> pd.DataFrame:
    p = find_one(root, "rb_market_casebook.csv")
    x = lower(pd.read_csv(p, low_memory=False))
    x["team"] = x["team"].map(canon_team)
    x["week"] = num(x, "week")
    if "short_key" not in x.columns:
        x["short_key"] = x["player"].map(short_key)
    keep = ["week", "team", "short_key", "consensus_line", "draftkings", "fanduel", "market_books",
            "model_minus_market", "abs_disagreement", "market_abs_error", "winner"]
    return x[[c for c in keep if c in x.columns]].copy()


def load_m96e(root: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(find_one(root, "m96e_router_trace.csv"), low_memory=False))
    x["team"] = x["team"].map(canon_team)
    x["week"] = num(x, "week")
    x["short_key"] = x["player"].map(short_key)
    keep = ["week", "team", "short_key", "pred_c", "pred_primary", "cal_prob_20", "m95f_p90",
            "prior_top1_unavailable", "workload_risk", "entrenched", "w_guard", "v_guard"]
    return x[[c for c in keep if c in x.columns]].copy()


def load_pbp_2025() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import nflreadpy as nfl
    raw = lower(to_pd(nfl.load_pbp(seasons=[2025])))
    if raw.empty:
        raise RuntimeError("nflverse PBP returned zero rows for 2025")
    raw["week"] = num(raw, "week")
    if "season_type" in raw.columns:
        reg = raw["season_type"].astype(str).str.upper().eq("REG")
        if reg.any():
            raw = raw.loc[reg].copy()
    raw = raw.loc[raw["week"].between(1, 18, inclusive="both")].copy()
    raw["posteam"] = raw.get("posteam", "").map(canon_team)
    for c in ["rush_attempt", "qb_dropback", "qb_kneel", "qb_scramble", "yards_gained", "rushing_yards", "success",
              "score_differential", "down", "yardline_100", "rush_touchdown"]:
        raw[c] = num(raw, c, 0)
    rush = raw.loc[raw["rush_attempt"].fillna(0).eq(1) & ~raw["qb_kneel"].fillna(0).eq(1)].copy()
    rusher_col = next((c for c in ["rusher_player_name", "rusher", "rusher_player_id"] if c in rush.columns), None)
    if rusher_col is None:
        raise RuntimeError("PBP missing rusher identity column")
    rush["short_key"] = rush[rusher_col].map(short_key)
    rush_yards_col = "rushing_yards" if "rushing_yards" in rush.columns and rush["rushing_yards"].notna().any() else "yards_gained"
    rush["rush_yards_diag"] = num(rush, rush_yards_col)
    player = rush.loc[rush.short_key.ne("")].groupby(["week", "posteam", "short_key"], as_index=False).agg(
        pbp_rush_att=("rush_attempt", "sum"),
        pbp_rush_yards=("rush_yards_diag", "sum"),
        max_rush_yards=("rush_yards_diag", "max"),
        rush_success_rate=("success", "mean"),
        explosive10_count=("rush_yards_diag", lambda s: int((s >= 10).sum())),
        explosive20_count=("rush_yards_diag", lambda s: int((s >= 20).sum())),
    ).rename(columns={"posteam": "team"})

    off = raw.loc[(raw["rush_attempt"].fillna(0).eq(1) | raw["qb_dropback"].fillna(0).eq(1)) & raw["posteam"].ne("")].copy()
    off["lead"] = off["score_differential"].fillna(0).gt(0).astype(int)
    off["trail"] = off["score_differential"].fillna(0).lt(0).astype(int)
    off["neutral"] = off["score_differential"].fillna(0).eq(0).astype(int)
    off["neutral_early"] = off["score_differential"].fillna(0).abs().le(8) & off["down"].fillna(0).isin([1, 2])
    trows = []
    for (week, team), g in off.groupby(["week", "posteam"]):
        ne = g.loc[g["neutral_early"]]
        trows.append({
            "week": int(week), "team": canon_team(team),
            "pbp_off_plays": int(len(g)),
            "pbp_team_rush_att": int(g["rush_attempt"].fillna(0).eq(1).sum()),
            "pbp_mean_score_diff": float(g["score_differential"].mean()) if g["score_differential"].notna().any() else np.nan,
            "pbp_lead_play_share": float(g["lead"].mean()),
            "pbp_neutral_play_share": float(g["neutral"].mean()),
            "pbp_trail_play_share": float(g["trail"].mean()),
            "pbp_neutral_early_rush_rate": float(ne["rush_attempt"].mean()) if len(ne) else np.nan,
            "pbp_redzone_rush_att": int(rush.loc[(rush.week.eq(week)) & rush.posteam.eq(team) & rush.yardline_100.le(20)].shape[0]),
        })
    team = pd.DataFrame(trows)
    audit = {"pbp_rows": int(len(raw)), "pbp_rush_rows": int(len(rush)), "pbp_player_groups": int(len(player)), "pbp_team_games": int(len(team))}
    return player, team, audit


def load_injuries_2025() -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl
    try:
        raw = lower(to_pd(nfl.load_injuries(seasons=[2025])))
    except Exception as exc:
        return pd.DataFrame(columns=["week", "team", "short_key", "injury_status", "practice_status", "injury_body_part"]), {"injury_source_error": str(exc), "injury_rows": 0}
    if raw.empty:
        return pd.DataFrame(columns=["week", "team", "short_key", "injury_status", "practice_status", "injury_body_part"]), {"injury_rows": 0}
    name_col = next((c for c in ["full_name", "player_name", "player", "name"] if c in raw.columns), None)
    team_col = next((c for c in ["team", "team_abbr", "team_abbreviation", "club"] if c in raw.columns), None)
    week_col = next((c for c in ["week", "report_week"] if c in raw.columns), None)
    if not all([name_col, team_col, week_col]):
        return pd.DataFrame(columns=["week", "team", "short_key", "injury_status", "practice_status", "injury_body_part"]), {"injury_rows": int(len(raw)), "injury_schema_usable": 0}
    out = pd.DataFrame(index=raw.index)
    out["week"] = pd.to_numeric(raw[week_col], errors="coerce")
    out["team"] = raw[team_col].map(canon_team)
    out["short_key"] = raw[name_col].map(short_key)
    status_col = next((c for c in ["report_status", "game_status", "status"] if c in raw.columns), None)
    prac_col = next((c for c in ["practice_status", "practice_participation"] if c in raw.columns), None)
    body_col = next((c for c in ["report_primary_injury", "primary_injury", "injury", "injury_type"] if c in raw.columns), None)
    out["injury_status"] = raw[status_col].astype("string") if status_col else ""
    out["practice_status"] = raw[prac_col].astype("string") if prac_col else ""
    out["injury_body_part"] = raw[body_col].astype("string") if body_col else ""
    out = out.loc[out.week.between(1, 18, inclusive="both") & out.team.ne("") & out.short_key.ne("")].drop_duplicates(["week", "team", "short_key"], keep="last")
    return out, {"injury_rows": int(len(raw)), "injury_normalized_rows": int(len(out)), "injury_schema_usable": 1}


def classify_ratio(a: float, b: float, label_a: str, label_b: str) -> str:
    aa = abs(a) if np.isfinite(a) else 0.0
    bb = abs(b) if np.isfinite(b) else 0.0
    if aa >= 1.25 * bb and aa > 0:
        return label_a
    if bb >= 1.25 * aa and bb > 0:
        return label_b
    return "MIXED"


def point_mae(df: pd.DataFrame, col: str) -> float:
    q = df[[col, "actual_rush_yards"]].apply(pd.to_numeric, errors="coerce").dropna()
    return float((q[col] - q.actual_rush_yards).abs().mean()) if len(q) else np.nan


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m94c-root", type=Path, required=True)
    p.add_argument("--m96e-root", type=Path, required=True)
    p.add_argument("--market-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_nd1_forensic_atlas"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rb, team94 = load_m94c(args.m94c_root)
    market = load_market(args.market_root)
    m96e = load_m96e(args.m96e_root)
    pbp_player, pbp_team, pbp_audit = load_pbp_2025()
    injuries, injury_audit = load_injuries_2025()

    # Exact two-factor Shapley decomposition: carries = team volume * player share.
    tp = rb.candidate_team_rush_att.astype(float)
    ta = rb.actual_team.astype(float)
    sp = rb.base_team_share.astype(float)
    sa = rb.actual_team_share.astype(float)
    rb["carry_volume_contrib"] = (ta - tp) * (sp + sa) / 2.0
    rb["carry_share_contrib"] = (sa - sp) * (tp + ta) / 2.0
    rb["carry_delta_actual_minus_proj"] = rb.actual_rush_att - rb.candidate_rush_att
    rb["carry_decomp_residual"] = rb.carry_delta_actual_minus_proj - rb.carry_volume_contrib - rb.carry_share_contrib
    rb["carry_primary"] = [classify_ratio(a, b, "TEAM_VOLUME", "PLAYER_SHARE") for a, b in zip(rb.carry_volume_contrib, rb.carry_share_contrib)]

    cp = rb.candidate_rush_att.astype(float)
    ca = rb.actual_rush_att.astype(float)
    ep = np.where(cp.gt(0), rb.candidate_rush_yards / cp, np.nan)
    ea = np.where(ca.gt(0), rb.actual_rush_yards / ca, np.nan)
    rb["projected_ypc"] = ep
    rb["actual_ypc"] = ea
    rb["yard_opportunity_contrib"] = (ca - cp) * (pd.Series(ep).fillna(0).values + pd.Series(ea).fillna(0).values) / 2.0
    rb["yard_efficiency_contrib"] = (pd.Series(ea).fillna(0).values - pd.Series(ep).fillna(0).values) * (cp + ca) / 2.0
    zero_actual = ca.eq(0)
    rb.loc[zero_actual, "yard_opportunity_contrib"] = -rb.loc[zero_actual, "candidate_rush_yards"]
    rb.loc[zero_actual, "yard_efficiency_contrib"] = 0.0
    rb["yard_delta_actual_minus_proj"] = rb.actual_rush_yards - rb.candidate_rush_yards
    rb["yard_decomp_residual"] = rb.yard_delta_actual_minus_proj - rb.yard_opportunity_contrib - rb.yard_efficiency_contrib
    rb["yard_primary"] = [classify_ratio(a, b, "OPPORTUNITY", "EFFICIENCY") for a, b in zip(rb.yard_opportunity_contrib, rb.yard_efficiency_contrib)]

    # Team RB-vs-non-RB competition from official M94C truth.
    rb_team = rb.groupby(["week", "team"], as_index=False).agg(actual_rb_att=("actual_rush_att", "sum"), projected_rb_att=("candidate_rush_att", "sum"))
    rb_team["actual_team_rush_att"] = rb.groupby(["week", "team"])["actual_team"].first().values
    rb_team["projected_team_rush_att"] = rb.groupby(["week", "team"])["candidate_team_rush_att"].first().values
    rb_team["actual_non_rb_att"] = (rb_team.actual_team_rush_att - rb_team.actual_rb_att).clip(lower=0)
    rb_team["actual_non_rb_rush_share"] = np.where(rb_team.actual_team_rush_att.gt(0), rb_team.actual_non_rb_att / rb_team.actual_team_rush_att, np.nan)
    rb = rb.merge(rb_team[["week", "team", "actual_rb_att", "actual_non_rb_att", "actual_non_rb_rush_share"]], on=["week", "team"], how="left", validate="many_to_one")

    rb = rb.merge(team94, on=["week", "team"], how="left", validate="many_to_one", suffixes=("", "_team94"))
    rb = rb.merge(pbp_team, on=["week", "team"], how="left", validate="many_to_one")
    rb = rb.merge(pbp_player, on=["week", "team", "short_key"], how="left", validate="many_to_one")
    rb = rb.merge(injuries, on=["week", "team", "short_key"], how="left", validate="many_to_one")
    rb = rb.merge(market, on=["week", "team", "short_key"], how="left", validate="one_to_one")
    rb = rb.merge(m96e, on=["week", "team", "short_key"], how="left", validate="one_to_one", suffixes=("", "_m96e"))

    rb["model_error"] = rb.candidate_rush_yards - rb.actual_rush_yards
    rb["abs_error"] = rb.model_error.abs()
    rb["miss_direction"] = np.select([rb.model_error.ge(10), rb.model_error.le(-10)], ["MODEL_HIGH", "MODEL_LOW"], default="WITHIN_10")
    rb["role_collapse_flag"] = ((rb.candidate_rush_att.ge(8) & rb.actual_rush_att.le(5)) | (rb.base_team_share.ge(.30) & rb.actual_team_share.le(.15))).astype(int)
    rb["new_role_init_flag"] = ((rb.week.le(3)) & rb.base_rush_att.fillna(0).le(3) & rb.actual_rush_att.ge(8)).astype(int)
    actual_margin = rb["pbp_mean_score_diff"] if "pbp_mean_score_diff" in rb.columns else rb.get("mean_score_diff", np.nan)
    rb["game_script_margin_error"] = actual_margin - rb.get("pred_mean_margin", np.nan)
    rb["game_script_miss_flag"] = rb.game_script_margin_error.abs().ge(10).fillna(False).astype(int)
    rb["explosive_shock_flag"] = (rb.explosive20_count.fillna(0).ge(1) & rb.yard_efficiency_contrib.abs().ge(10)).astype(int)
    rb["non_rb_competition_flag"] = ((rb.actual_non_rb_att.fillna(0).ge(8)) | (rb.actual_non_rb_rush_share.fillna(0).ge(.30))).astype(int)
    rb["market_large_disagreement_flag"] = rb.abs_disagreement.fillna(0).ge(15).astype(int)
    status_txt = rb.injury_status.astype("string").fillna("").str.upper()
    rb["pregame_injury_flag"] = status_txt.str.contains("QUESTION|DOUBT|OUT|INJ", regex=True).astype(int)

    # High-level compound diagnostic label: preserve primary fields separately.
    rb["forensic_label"] = rb.yard_primary + "__" + rb.carry_primary

    # Decomposition invariants are diagnostics, not gates against official-vs-PBP stat differences.
    max_carry_resid = float(rb.carry_decomp_residual.abs().max())
    max_yard_resid = float(rb.yard_decomp_residual.abs().max())
    if max_carry_resid > 1e-6 or max_yard_resid > 1e-6:
        raise RuntimeError(f"Shapley decomposition invariant failed: carry={max_carry_resid} yard={max_yard_resid}")

    # Summary by mechanism; dominant-class error dollars are literal model absolute error sums.
    rows = []
    total_abs = float(rb.abs_error.sum())
    for label, g in rb.groupby("forensic_label"):
        rows.append({
            "forensic_label": label, "n": int(len(g)), "mae": float(g.abs_error.mean()),
            "total_abs_error": float(g.abs_error.sum()), "share_of_total_abs_error": float(g.abs_error.sum() / total_abs) if total_abs else np.nan,
            "mean_carry_error": float((g.candidate_rush_att - g.actual_rush_att).mean()),
            "role_collapse_rate": float(g.role_collapse_flag.mean()), "game_script_miss_rate": float(g.game_script_miss_flag.mean()),
            "explosive_shock_rate": float(g.explosive_shock_flag.mean()), "non_rb_competition_rate": float(g.non_rb_competition_flag.mean()),
        })
    summary = pd.DataFrame(rows).sort_values("total_abs_error", ascending=False)

    contrib = pd.DataFrame([{
        "abs_carry_volume_contrib": float(rb.carry_volume_contrib.abs().sum()),
        "abs_carry_share_contrib": float(rb.carry_share_contrib.abs().sum()),
        "abs_yard_opportunity_contrib": float(rb.yard_opportunity_contrib.abs().sum()),
        "abs_yard_efficiency_contrib": float(rb.yard_efficiency_contrib.abs().sum()),
    }])
    cden = contrib.abs_carry_volume_contrib.iloc[0] + contrib.abs_carry_share_contrib.iloc[0]
    yden = contrib.abs_yard_opportunity_contrib.iloc[0] + contrib.abs_yard_efficiency_contrib.iloc[0]
    contrib["carry_volume_share_of_abs_contrib"] = contrib.abs_carry_volume_contrib / cden
    contrib["carry_share_share_of_abs_contrib"] = contrib.abs_carry_share_contrib / cden
    contrib["yard_opportunity_share_of_abs_contrib"] = contrib.abs_yard_opportunity_contrib / yden
    contrib["yard_efficiency_share_of_abs_contrib"] = contrib.abs_yard_efficiency_contrib / yden

    # Market benchmark by diagnosed mechanism.
    mg = rb.loc[rb.consensus_line.notna()].copy()
    market_rows = []
    for label, g in mg.groupby("forensic_label"):
        market_rows.append({
            "forensic_label": label, "n": int(len(g)),
            "m94c_mae": point_mae(g, "candidate_rush_yards"), "market_mae": point_mae(g.rename(columns={"consensus_line": "tmp"}), "tmp") if False else float((g.consensus_line-g.actual_rush_yards).abs().mean()),
            "market_advantage_mae": float((g.candidate_rush_yards-g.actual_rush_yards).abs().mean() - (g.consensus_line-g.actual_rush_yards).abs().mean()),
            "model_high_rate": float(g.model_error.gt(0).mean()), "role_collapse_rate": float(g.role_collapse_flag.mean()),
        })
    market_summary = pd.DataFrame(market_rows).sort_values("market_advantage_mae", ascending=False)

    # Apples-to-apples M96E audit on its authoritative evaluation window (W6-18 only).
    e = rb.loc[rb.week.ge(6) & rb.pred_primary.notna()].copy()
    e_rows = []
    for label, g in e.groupby("forensic_label"):
        e_rows.append({
            "forensic_label": label, "n": int(len(g)),
            "m94c_mae": float((g.candidate_rush_yards-g.actual_rush_yards).abs().mean()),
            "m96e_mae": float((g.pred_primary-g.actual_rush_yards).abs().mean()),
            "m96e_gain": float((g.candidate_rush_yards-g.actual_rush_yards).abs().mean() - (g.pred_primary-g.actual_rush_yards).abs().mean()),
        })
    e_summary = pd.DataFrame(e_rows).sort_values("m96e_gain", ascending=False)

    # Exact market-covered W6-18 three-way headline.
    em = e.loc[e.consensus_line.notna()].copy()
    headline = pd.DataFrame([
        {"arm": "M94C", "n": len(em), "mae": point_mae(em, "candidate_rush_yards")},
        {"arm": "M96E", "n": len(em), "mae": point_mae(em, "pred_primary")},
        {"arm": "VEGAS_CONSENSUS", "n": len(em), "mae": float((em.consensus_line-em.actual_rush_yards).abs().mean()) if len(em) else np.nan},
    ])

    flags = []
    for name in ["role_collapse_flag", "new_role_init_flag", "game_script_miss_flag", "explosive_shock_flag", "non_rb_competition_flag", "pregame_injury_flag"]:
        g = rb.loc[rb[name].eq(1)]
        flags.append({"flag": name, "n": int(len(g)), "mae": float(g.abs_error.mean()) if len(g) else np.nan,
                      "share_of_total_abs_error": float(g.abs_error.sum()/total_abs) if total_abs and len(g) else 0.0})
    flag_summary = pd.DataFrame(flags).sort_values("share_of_total_abs_error", ascending=False)

    proxy_map = pd.DataFrame([
        {"mechanism": "PLAYER_SHARE / role collapse", "pregame_data_family": "timestamped depth chart; current-week injury/practice/game status; inactives; transactions; recent snaps/routes/carries; competing-RB availability; rookie/new-team role priors"},
        {"mechanism": "TEAM_VOLUME / game script", "pregame_data_family": "team plays/pace; neutral PROE/rush rate; opponent pace; spread-free football score-state model; OL/QB availability; coaching/play-caller tendencies"},
        {"mechanism": "EFFICIENCY", "pregame_data_family": "OL availability/continuity; YBC/YAC/RYOE; run concept/personnel; box/front; opponent run-fit/tackle/edge/interior strength; weather/surface"},
        {"mechanism": "EXPLOSIVE shock", "pregame_data_family": "runner explosive rate and long-run distribution; missed tackles; opponent explosive-run allowance; safety/front structure; concept matchup; distribution tail only"},
        {"mechanism": "NON-RB competition", "pregame_data_family": "QB designed-run/scramble tendency; WR/gadget usage; goal-line QB role; team rush allocation by position and personnel"},
    ])

    case_cols = [c for c in ["season", "week", "team", "opponent", "player", "role", "actual_rush_att", "candidate_rush_att", "actual_rush_yards", "candidate_rush_yards", "abs_error", "model_error", "carry_primary", "yard_primary", "forensic_label", "carry_volume_contrib", "carry_share_contrib", "yard_opportunity_contrib", "yard_efficiency_contrib", "actual_team", "candidate_team_rush_att", "base_team_share", "actual_team_share", "projected_ypc", "actual_ypc", "max_rush_yards", "explosive20_count", "rush_success_rate", "pred_mean_margin", "pbp_mean_score_diff", "game_script_margin_error", "actual_non_rb_att", "actual_non_rb_rush_share", "role_collapse_flag", "new_role_init_flag", "game_script_miss_flag", "explosive_shock_flag", "non_rb_competition_flag", "pregame_injury_flag", "injury_status", "practice_status", "consensus_line", "abs_disagreement", "pred_primary", "cal_prob_20", "m95f_p90", "workload_risk"] if c in rb.columns]
    casebook = rb.sort_values("abs_error", ascending=False)[case_cols].head(300)

    source_audit = {"m94c_rb_rows": int(len(rb)), "market_covered_rows": int(rb.consensus_line.notna().sum()),
                    "m96e_w6_18_rows": int(e.pred_primary.notna().sum()), "m96e_market_w6_18_rows": int(len(em)),
                    "pbp_player_match_rows": int(rb.pbp_rush_att.notna().sum()), "pbp_team_match_rows": int(rb.pbp_off_plays.notna().sum()),
                    "same_week_injury_report_rows_matched": int(rb.injury_status.notna().sum()), **pbp_audit, **injury_audit}
    pd.DataFrame([source_audit]).to_csv(args.out_dir / "rb_nd1_source_audit.csv", index=False)
    rb.to_csv(args.out_dir / "rb_nd1_forensic_trace.csv", index=False)
    casebook.to_csv(args.out_dir / "rb_nd1_large_miss_casebook.csv", index=False)
    summary.to_csv(args.out_dir / "rb_nd1_mechanism_summary.csv", index=False)
    contrib.to_csv(args.out_dir / "rb_nd1_contribution_summary.csv", index=False)
    market_summary.to_csv(args.out_dir / "rb_nd1_market_by_mechanism.csv", index=False)
    e_summary.to_csv(args.out_dir / "rb_nd1_m96e_by_mechanism.csv", index=False)
    headline.to_csv(args.out_dir / "rb_nd1_m96e_market_headline.csv", index=False)
    flag_summary.to_csv(args.out_dir / "rb_nd1_flag_summary.csv", index=False)
    proxy_map.to_csv(args.out_dir / "rb_nd1_pregame_proxy_map.csv", index=False)

    print("=== source audit ===")
    print(pd.DataFrame([source_audit]).to_string(index=False))
    print("=== contribution summary ===")
    print(contrib.to_string(index=False))
    print("=== mechanism summary ===")
    print(summary.head(20).to_string(index=False))
    print("=== special flags ===")
    print(flag_summary.to_string(index=False))
    print("=== M96E vs M94C vs market W6-18 ===")
    print(headline.to_string(index=False))
    print("=== market by mechanism ===")
    print(market_summary.head(20).to_string(index=False))
    print("=== M96E by mechanism ===")
    print(e_summary.head(20).to_string(index=False))
    print("=== largest misses ===")
    print(casebook.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
