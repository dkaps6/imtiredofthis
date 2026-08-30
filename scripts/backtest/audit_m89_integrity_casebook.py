#!/usr/bin/env python3
"""M89 Phase 0/1: source integrity audit and catastrophic QB game casebook.

Postgame PBP is used only for source reconciliation and forensic explanation.
Nothing emitted here is eligible as a Phase-2 pregame feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp

SEASONS = (2023, 2024, 2025)
TAIL_THRESHOLD = 100.0


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def canon(v) -> str:
    return canon_team(v)


def key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def regular(x: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in x.columns:
        r = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not r.empty:
            return r
    if "game_type" in x.columns:
        r = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not r.empty:
            return r
    return x


def load_all_pbp() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        q = regular(lower(get_pbp(season, min_rows=1)))
        q["season"] = pd.to_numeric(q["season"], errors="coerce")
        q["week"] = pd.to_numeric(q["week"], errors="coerce")
        q = q.loc[q["season"].eq(season) & q["week"].between(1, 18)].copy()
        q["season"] = q["season"].astype(int)
        q["week"] = q["week"].astype(int)
        frames.append(q)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_weekly_stats() -> pd.DataFrame:
    import nflreadpy as nfl
    frames = []
    for season in SEASONS:
        raw = nfl.load_player_stats(seasons=[season], summary_level="week")
        q = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        q = lower(q)
        q["season"] = pd.to_numeric(q.get("season", season), errors="coerce").fillna(season).astype(int)
        q["week"] = pd.to_numeric(q.get("week"), errors="coerce")
        q = q.loc[q["season"].eq(season) & q["week"].between(1, 18)].copy()
        frames.append(q)
    return pd.concat(frames, ignore_index=True, sort=False)


def reconcile_actuals(pbp: pd.DataFrame, weekly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = pbp.copy()
    required = {"posteam", "passer_player_id", "pass_attempt", "passing_yards", "season", "week"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"PBP missing reconciliation columns {sorted(missing)}")
    x["team"] = x["posteam"].map(canon)
    x["player_id"] = x["passer_player_id"].astype(str).replace({"nan":"", "None":"", "<NA>":""})
    x["_pa"] = num(x, "pass_attempt", 0).fillna(0).eq(1)
    x["_py"] = num(x, "passing_yards", 0).fillna(0)
    p = (x.loc[x["_pa"] & x["team"].ne("") & x["player_id"].ne("")]
           .groupby(["season","week","team","player_id"], as_index=False)
           .agg(pbp_attempts=("_pa","sum"), pbp_pass_yards=("_py","sum")))

    w = weekly.copy()
    w["team"] = w[next((c for c in ["recent_team","team","posteam"] if c in w.columns), "recent_team")].map(canon)
    idcol = next((c for c in ["player_id","gsis_id","player_gsis_id"] if c in w.columns), None)
    if not idcol:
        raise RuntimeError("weekly stats have no player/GSIS id for truth reconciliation")
    w["player_id"] = w[idcol].astype(str).replace({"nan":"", "None":"", "<NA>":""})
    attcol = next((c for c in ["attempts","passing_attempts","pass_attempts"] if c in w.columns), None)
    ydcol = next((c for c in ["passing_yards","pass_yards"] if c in w.columns), None)
    if not attcol or not ydcol:
        raise RuntimeError("weekly stats missing passing attempts/yards")
    w["weekly_attempts"] = pd.to_numeric(w[attcol], errors="coerce")
    w["weekly_pass_yards"] = pd.to_numeric(w[ydcol], errors="coerce")
    w = w.loc[w["weekly_attempts"].fillna(0).gt(0) & w["team"].ne("") & w["player_id"].ne("")]
    w = w[["season","week","team","player_id","weekly_attempts","weekly_pass_yards"]].drop_duplicates(["season","week","team","player_id"])

    z = w.merge(p, on=["season","week","team","player_id"], how="outer", indicator=True, validate="one_to_one")
    z["attempt_diff"] = z["pbp_attempts"] - z["weekly_attempts"]
    z["yards_diff"] = z["pbp_pass_yards"] - z["weekly_pass_yards"]
    matched = z["_merge"].eq("both")
    z["attempt_exact"] = matched & z["attempt_diff"].abs().le(1e-9)
    z["yards_exact"] = matched & z["yards_diff"].abs().le(1e-9)
    nmatch = int(matched.sum())
    summary = {
        "matched_qb_games": nmatch,
        "weekly_only": int(z["_merge"].eq("left_only").sum()),
        "pbp_only": int(z["_merge"].eq("right_only").sum()),
        "attempt_exact_rate": float(z.loc[matched, "attempt_exact"].mean()) if nmatch else 0.0,
        "yards_exact_rate": float(z.loc[matched, "yards_exact"].mean()) if nmatch else 0.0,
        "attempt_gate_99pct": bool(nmatch and z.loc[matched, "attempt_exact"].mean() >= 0.99),
        "yards_gate_99pct": bool(nmatch and z.loc[matched, "yards_exact"].mean() >= 0.99),
    }
    return z, summary


def source_manifest() -> pd.DataFrame:
    rows = [
        ("play_by_play", "nflverse via nflreadpy", "LIVE_2026", "core football source"),
        ("weekly_player_stats", "nflverse via nflreadpy", "LIVE_2026", "maintained summary_level=week API"),
        ("schedule_rosters", "nflverse via nflreadpy", "LIVE_2026", "schedule and weekly roster backbone"),
        ("injuries", "ESPN public injury JSON via scripts/providers/injuries.py", "LIVE_2026", "M89 replacement; stale cache prohibited"),
        ("historical_injuries_2024_and_prior", "nflverse", "HISTORICAL_ONLY", "usable where source actually existed"),
        ("historical_outdoor_weather", "none", "MISSING", "controlled-venue status only; archived pregame forecast still missing"),
        ("participation_man_zone_box", "nflverse participation / FTN", "HISTORICAL_ONLY", "recent-season releases are not an in-season live contract"),
        ("wr_cb_exposure", "repo exposure sources / external matchup tools", "PROXY", "exact responsibility feed remains source-blocked"),
        ("full_pressure", "public PBP", "MISSING", "PBP sack+QB-hit is only a proxy; hurries not complete"),
        ("hit_sack_pressure_proxy", "nflverse PBP", "PROXY", "explicitly labeled sack-or-QB-hit"),
        ("game_total_spread_history", "nflverse schedules", "HISTORICAL_ONLY", "pregame/closing-style snapshot; timing must be labeled"),
        ("game_total_spread_2026", "TheOddsAPI / configured live market layer", "LIVE_2026", "market layer only, never football-model input unless separately labeled"),
    ]
    return pd.DataFrame(rows, columns=["family","source","deployment_status","note"])


def load_m82(path: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(path, low_memory=False))
    required = {"season","week","team","player_clean_key","actual_pass_yards","actual_attempts","ensemble_proj"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M82 trace missing {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    x["team"] = x["team"].map(canon)
    x["player_clean_key"] = x["player_clean_key"].map(key)
    x["ensemble_error"] = pd.to_numeric(x["ensemble_proj"], errors="coerce") - pd.to_numeric(x["actual_pass_yards"], errors="coerce")
    x["ensemble_abs_error"] = x["ensemble_error"].abs()
    return x


def attach_m86_low_chaos(target: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Attach the exact frozen M86 low-chaos classification.

    M86 emits both a boolean `high_event_chaos` and a string `chaos_class`.
    Prefer the explicit string class so a numeric 0/1 flag can never be mistaken
    for a label. Fall back to the inverse boolean only if the class is absent.
    """
    out = target.copy()
    out["m86_low_event_chaos"] = 0
    if path is None or not path.exists():
        return out
    m = lower(pd.read_csv(path, low_memory=False))
    for c in ["team", "player_clean_key"]:
        if c in m.columns:
            m[c] = m[c].map(canon if c == "team" else key)
    required = {"season", "week", "team", "player_clean_key"}
    if not required.issubset(m.columns):
        return out

    if "chaos_class" in m.columns:
        low = m["chaos_class"].astype(str).str.upper().eq("LOW_EVENT_CHAOS")
    elif "high_event_chaos" in m.columns:
        raw = m["high_event_chaos"]
        if pd.api.types.is_bool_dtype(raw):
            high = raw.fillna(False)
        else:
            txt = raw.astype(str).str.strip().str.upper()
            numeric = pd.to_numeric(raw, errors="coerce")
            high = numeric.fillna(0).astype(bool)
            high = high.where(~txt.isin(["TRUE", "FALSE"]), txt.eq("TRUE"))
        low = ~high
    else:
        return out

    q = m[["season", "week", "team", "player_clean_key"]].copy()
    q["m86_low_event_chaos"] = low.astype(int).to_numpy()
    q = q.drop_duplicates(["season", "week", "team", "player_clean_key"])
    merged = out.drop(columns=["m86_low_event_chaos"]).merge(
        q,
        on=["season", "week", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    merged["m86_low_event_chaos"] = pd.to_numeric(
        merged["m86_low_event_chaos"], errors="coerce"
    ).fillna(0).astype(int)
    return merged


def _match_game_rows(pbp: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    x = pbp.loc[(pbp["season"].eq(season)) & (pbp["week"].eq(week))].copy()
    post = x.get("posteam", pd.Series("", index=x.index)).map(canon)
    deff = x.get("defteam", pd.Series("", index=x.index)).map(canon)
    return x.loc[post.eq(team) | deff.eq(team)].copy()


def case_for_row(row: pd.Series, pbp: pd.DataFrame) -> dict:
    season, week, team = int(row.season), int(row.week), canon(row.team)
    ggame = _match_game_rows(pbp, season, week, team)
    if ggame.empty:
        return {"case_status":"missing_pbp"}
    posteam = ggame.get("posteam", pd.Series("", index=ggame.index)).map(canon)
    off = ggame.loc[posteam.eq(team)].copy()
    if off.empty:
        return {"case_status":"missing_offense_pbp"}

    pa = num(off, "pass_attempt", 0).fillna(0).eq(1)
    passer = off.get("passer_player_id", pd.Series("", index=off.index)).astype(str).replace({"nan":"", "None":"", "<NA>":""})
    candidates = []
    for pid, pg in off.loc[pa & passer.ne("")].assign(_pid=passer.loc[pa & passer.ne("")]).groupby("_pid"):
        candidates.append((pid, int(len(pg)), float(num(pg,"passing_yards",0).fillna(0).sum()), pg))
    if not candidates:
        return {"case_status":"missing_passer_attempts"}
    actual_attempts = float(row.get("actual_attempts", np.nan))
    candidates.sort(key=lambda t: (abs(t[1] - actual_attempts) if np.isfinite(actual_attempts) else -t[1], -t[1]))
    pid, pbp_attempts, pbp_yards, qbp = candidates[0]

    qbp = qbp.copy()
    qbp["_py"] = num(qbp, "passing_yards", 0).fillna(0)
    qbp["_qtr"] = num(qbp, "qtr")
    qbp["_score_diff"] = num(qbp, "score_differential")
    qbp["_yac"] = num(qbp, "yards_after_catch")
    qbp["_complete"] = num(qbp, "complete_pass", 0).fillna(0).eq(1)
    qbp["_int"] = num(qbp, "interception", 0).fillna(0).eq(1)
    longest = float(qbp.loc[qbp["_complete"], "_py"].max()) if qbp["_complete"].any() else 0.0
    max_yac = float(qbp.loc[qbp["_complete"], "_yac"].max()) if qbp["_complete"].any() and qbp["_yac"].notna().any() else np.nan
    yac_total = float(qbp.loc[qbp["_complete"], "_yac"].fillna(0).sum()) if qbp["_complete"].any() else 0.0
    qyards = {q: float(qbp.loc[qbp["_qtr"].eq(q), "_py"].sum()) for q in [1,2,3,4,5]}
    qatts = {q: int(qbp.loc[qbp["_qtr"].eq(q)].shape[0]) for q in [1,2,3,4,5]}
    first_half_yards = qyards[1] + qyards[2]
    second_half_yards = qyards[3] + qyards[4] + qyards[5]
    first_half_att = qatts[1] + qatts[2]
    second_half_att = qatts[3] + qatts[4] + qatts[5]
    trailing = float(qbp["_score_diff"].lt(0).mean()) if qbp["_score_diff"].notna().any() else np.nan
    leading = float(qbp["_score_diff"].gt(0).mean()) if qbp["_score_diff"].notna().any() else np.nan
    neutral = float(qbp["_score_diff"].eq(0).mean()) if qbp["_score_diff"].notna().any() else np.nan
    garbage = float((qbp["_qtr"].ge(4) & qbp["_score_diff"].abs().ge(17)).mean()) if qbp["_score_diff"].notna().any() else np.nan

    receiver_col = "receiver_player_name" if "receiver_player_name" in qbp.columns else "receiver_player_id" if "receiver_player_id" in qbp.columns else None
    top_receiver = ""
    top_receiver_yards = np.nan
    top_receiver_share = np.nan
    if receiver_col:
        rg = qbp.loc[qbp["_complete"] & qbp[receiver_col].notna()].groupby(receiver_col)["_py"].sum().sort_values(ascending=False)
        if len(rg):
            top_receiver = str(rg.index[0])
            top_receiver_yards = float(rg.iloc[0])
            top_receiver_share = float(top_receiver_yards / pbp_yards) if pbp_yards else np.nan

    # Team-level drive/game events.
    drop = num(off, "qb_dropback", 0).fillna(0).eq(1)
    sacks = int(num(off, "sack", 0).fillna(0).eq(1).sum())
    interceptions = int(num(qbp, "interception", 0).fillna(0).eq(1).sum())
    scrambles = int(num(off, "qb_scramble", 0).fillna(0).eq(1).sum()) if "qb_scramble" in off.columns else 0
    fourth_down = int(num(off, "down").eq(4).sum()) if "down" in off.columns else 0
    overtime = int(num(off, "qtr").gt(4).any()) if "qtr" in off.columns else 0
    drives = int(pd.to_numeric(off.get("drive"), errors="coerce").nunique()) if "drive" in off.columns else 0
    max_drive_passes = 0
    if "drive" in qbp.columns:
        max_drive_passes = int(qbp.groupby("drive").size().max()) if len(qbp) else 0

    ensemble = float(row.ensemble_proj)
    actual = float(row.actual_pass_yards)
    err = ensemble - actual
    actual_without_longest = actual - longest
    err_without_longest = ensemble - actual_without_longest
    survives_without_longest = int(abs(err_without_longest) >= TAIL_THRESHOLD)
    pred_attempts = float(row.get("pred_attempts", row.get("mc_expected_pass_attempts", np.nan)))
    pred_ypa = float(row.get("implied_pred_ypa", row.get("pred_ypa", np.nan)))
    actual_ypa = actual / float(row.actual_attempts) if float(row.actual_attempts) else np.nan
    attempt_resid = float(row.actual_attempts) - pred_attempts if np.isfinite(pred_attempts) else np.nan
    ypa_resid = actual_ypa - pred_ypa if np.isfinite(pred_ypa) else np.nan

    # Deterministic descriptive taxonomy. This is not a causal model.
    label = "MIXED"
    if overtime and abs(err) >= 100:
        label = "OVERTIME"
    if longest >= 60 and not survives_without_longest:
        label = "SINGLE_EXPLOSIVE_PLAY"
    elif err < 0 and np.isfinite(max_yac) and max_yac >= 30 and yac_total >= 0.30 * max(actual, 1):
        label = "YAC_DRIVEN_EXPLOSION"
    elif err < 0 and np.isfinite(top_receiver_share) and top_receiver_share >= 0.55:
        label = "RECEIVER_CONCENTRATION"
    elif np.isfinite(attempt_resid) and attempt_resid >= 10:
        label = "FORCED_PASS_VOLUME" if (np.isfinite(trailing) and trailing >= 0.55) else "VOLUNTARY_PASS_VOLUME"
    elif np.isfinite(attempt_resid) and attempt_resid <= -8:
        label = "UNEXPECTED_LOW_VOLUME"
    elif sacks >= 4 and err > 0:
        label = "PROTECTION_COLLAPSE"
    elif interceptions >= 2:
        label = "TURNOVER_POSSESSION_DISTORTION"
    elif np.isfinite(garbage) and garbage >= 0.30 and err < 0:
        label = "GARBAGE_TIME"
    elif np.isfinite(ypa_resid) and ypa_resid >= 2.0:
        label = "SUSTAINED_EFFICIENCY_EXPLOSION"
    elif np.isfinite(ypa_resid) and ypa_resid <= -2.0:
        label = "SUSTAINED_EFFICIENCY_COLLAPSE"

    return {
        "case_status":"ok", "matched_passer_id":pid, "pbp_attempts":pbp_attempts, "pbp_pass_yards":pbp_yards,
        "attempt_match_abs":abs(pbp_attempts-float(row.actual_attempts)), "yards_match_abs":abs(pbp_yards-actual),
        "q1_yards":qyards[1], "q2_yards":qyards[2], "q3_yards":qyards[3], "q4_yards":qyards[4], "ot_yards":qyards[5],
        "q1_attempts":qatts[1], "q2_attempts":qatts[2], "q3_attempts":qatts[3], "q4_attempts":qatts[4], "ot_attempts":qatts[5],
        "first_half_yards":first_half_yards, "second_half_yards":second_half_yards,
        "first_half_attempts":first_half_att, "second_half_attempts":second_half_att,
        "trailing_attempt_share":trailing, "leading_attempt_share":leading, "tied_attempt_share":neutral,
        "garbage_time_attempt_share":garbage, "longest_completion":longest,
        "completions_20plus":int((qbp["_complete"] & qbp["_py"].ge(20)).sum()),
        "completions_40plus":int((qbp["_complete"] & qbp["_py"].ge(40)).sum()),
        "completions_60plus":int((qbp["_complete"] & qbp["_py"].ge(60)).sum()),
        "yac_total":yac_total, "max_yac":max_yac,
        "top_receiver":top_receiver, "top_receiver_yards":top_receiver_yards, "top_receiver_yard_share":top_receiver_share,
        "actual_without_longest_completion":actual_without_longest, "error_without_longest_completion":err_without_longest,
        "tail_survives_without_longest_completion":survives_without_longest,
        "sacks_taken":sacks, "interceptions":interceptions, "scrambles":scrambles,
        "fourth_down_plays":fourth_down, "overtime":overtime, "offensive_drives":drives, "max_pass_attempts_one_drive":max_drive_passes,
        "attempt_residual":attempt_resid, "actual_ypa":actual_ypa, "pred_ypa_case":pred_ypa, "ypa_residual":ypa_resid,
        "forensic_primary_label":label,
    }


def build_casebook(m82: pd.DataFrame, pbp: pd.DataFrame, m86_path: Path | None) -> pd.DataFrame:
    target = m82.loc[m82["ensemble_abs_error"].ge(TAIL_THRESHOLD)].copy()
    if len(target) != 123:
        raise RuntimeError(f"M89 expected 123 M82 catastrophic rows; found {len(target)}")
    target = attach_m86_low_chaos(target, m86_path)
    extra = []
    for _, row in target.iterrows():
        extra.append(case_for_row(row, pbp))
    return pd.concat([target.reset_index(drop=True), pd.DataFrame(extra)], axis=1)


def render_cases(x: pd.DataFrame, title: str, n: int | None = None) -> str:
    q = x.sort_values("ensemble_abs_error", ascending=False).copy()
    if n is not None:
        q = q.head(n)
    lines = [f"# {title}", "", "Postgame material below is forensic only and is not eligible as a pregame model feature.", ""]
    for _, r in q.iterrows():
        lines += [
            f"## {int(r.season)} W{int(r.week)} — {r.team} — {r.player_clean_key}",
            f"- Projection: ensemble {float(r.ensemble_proj):.1f}; actual {float(r.actual_pass_yards):.1f}; error {float(r.ensemble_error):+.1f} yards.",
            f"- Components: attempts residual {float(r.get('attempt_residual', np.nan)):+.1f}; YPA residual {float(r.get('ypa_residual', np.nan)):+.2f}.",
            f"- Game shape: Q1/Q2/Q3/Q4/OT yards {float(r.get('q1_yards',0)):.0f}/{float(r.get('q2_yards',0)):.0f}/{float(r.get('q3_yards',0)):.0f}/{float(r.get('q4_yards',0)):.0f}/{float(r.get('ot_yards',0)):.0f}; trailing-attempt share {float(r.get('trailing_attempt_share',np.nan)):.1%}.",
            f"- Explosives: longest {float(r.get('longest_completion',0)):.0f}; 20+/40+/60+ = {int(r.get('completions_20plus',0))}/{int(r.get('completions_40plus',0))}/{int(r.get('completions_60plus',0))}; max YAC {float(r.get('max_yac',np.nan)):.1f}.",
            f"- Receiver concentration: {r.get('top_receiver','')} {float(r.get('top_receiver_yards',np.nan)):.0f} yards ({float(r.get('top_receiver_yard_share',np.nan)):.1%}).",
            f"- Remove largest completion: actual {float(r.get('actual_without_longest_completion',np.nan)):.1f}; error {float(r.get('error_without_longest_completion',np.nan)):+.1f}; still 100+ miss = {bool(r.get('tail_survives_without_longest_completion',0))}.",
            f"- Other events: sacks {int(r.get('sacks_taken',0))}; INT {int(r.get('interceptions',0))}; scrambles {int(r.get('scrambles',0))}; OT {bool(r.get('overtime',0))}; garbage-attempt share {float(r.get('garbage_time_attempt_share',np.nan)):.1%}.",
            f"- Forensic label: **{r.get('forensic_primary_label','MIXED')}**.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m82-trace", type=Path, required=True)
    p.add_argument("--m86-trace", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pbp = load_all_pbp()
    weekly = load_weekly_stats()
    reconciliation, truth = reconcile_actuals(pbp, weekly)
    manifest = source_manifest()
    m82 = load_m82(args.m82_trace)
    casebook = build_casebook(m82, pbp, args.m86_trace)

    reconciliation.to_csv(args.out_dir / "m89_actual_stat_reconciliation.csv", index=False)
    manifest.to_csv(args.out_dir / "m89_source_availability_manifest.csv", index=False)
    casebook.to_csv(args.out_dir / "m89_catastrophic_casebook.csv", index=False)
    casebook.groupby(["season","forensic_primary_label"], dropna=False).size().reset_index(name="n").to_csv(args.out_dir / "m89_casebook_taxonomy_counts.csv", index=False)

    low = casebook.loc[casebook["m86_low_event_chaos"].eq(1)].copy()
    (args.out_dir / "M89_LARGEST_30_CASEBOOK.md").write_text(render_cases(casebook, "M89 Largest 30 Catastrophic QB Case Files", 30), encoding="utf-8")
    (args.out_dir / "M89_LOW_CHAOS_CASEBOOK.md").write_text(render_cases(low, "M89 M86 Low-Chaos Catastrophic QB Case Files", None), encoding="utf-8")

    summary = {
        "truth_reconciliation": truth,
        "m82_rows": int(len(m82)),
        "catastrophic_rows": int(len(casebook)),
        "low_chaos_rows_recovered": int(casebook["m86_low_event_chaos"].sum()),
        "tails_no_longer_100plus_without_largest_completion": int((casebook["tail_survives_without_longest_completion"].eq(0)).sum()),
        "tails_still_100plus_without_largest_completion": int((casebook["tail_survives_without_longest_completion"].eq(1)).sum()),
        "case_status_ok": int(casebook["case_status"].eq("ok").sum()),
        "postgame_casebook_features_used_for_prediction": False,
        "sportsbook_features_used_for_truth_reconciliation": False,
    }
    (args.out_dir / "m89_integrity_casebook_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== M89 ACTUAL STAT TRUTH ===")
    print(json.dumps(truth, indent=2))
    print("=== M89 CASEBOOK TAXONOMY ===")
    print(casebook["forensic_primary_label"].value_counts(dropna=False).to_string())
    print("=== M89 CASEBOOK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
