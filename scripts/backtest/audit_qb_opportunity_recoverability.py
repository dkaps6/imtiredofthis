#!/usr/bin/env python3
"""Migration 73 — QB passing-opportunity recoverability / mechanism atlas.

M70-M72 failed to find a stable pregame directional or uncertainty correction for
remaining QB efficiency errors. M64/M65 showed that realized dropback behavior
has large oracle value, but specific generative/state-occupancy approaches did
not beat the canonical Raw frontier.

M73 does NOT fit a new production model. It answers two frozen questions:
1) How much passing-yard MAE is theoretically recoverable from perfect attempts
   versus perfect YPA while holding the other canonical component fixed?
2) On large canonical attempt misses, which realized football opportunity
   mechanism moved most: drives, plays/drive, dropback rate, attempt conversion,
   or primary-QB share?

Boundary:
- immutable qb_frontier_canonical_v1 is the projection source
- 2024/2025 are scored; 2023 is history only
- no sportsbook/player-prop/game-market feature is used
- all expected mechanism baselines are strictly prior to the target week
- target-game PBP is used only for realized mechanism attribution
- diagnostic only; production_actionable is always False
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_efficiency_uncertainty as m71

HISTORY_WINDOW = 8
LEAGUE_PRIOR_GAMES = 4.0
MIN_QB_PRIOR = 3
STABLE_PRIMARY_SHARE = 0.80

LARGE_ATTEMPT_MISS = 8.0
EXTREME_ATTEMPT_MISS = 10.0
LARGE_ATTEMPT_HEADROOM_YARDS = 8.0
MODERATE_ATTEMPT_HEADROOM_YARDS = 4.0
DOMINANT_MECHANISM_SHARE = 0.35
DOMINANT_MECHANISM_MEDIAN_ATTEMPTS = 3.0

COMPONENTS = ["drives", "plays_per_drive", "dropback_rate", "attempt_conversion", "qb_share"]
SURVIVAL = ["first_down_rate", "third_down_conversion", "early_down_success", "sack_rate", "turnover_per_drive"]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def safe_div(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def mae(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[ok] - b[ok]))) if ok.any() else np.nan


def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean(np.square(a[ok] - b[ok])))) if ok.any() else np.nan


def corr(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b))


def prior_mask(df, season, week):
    return (num(df.season) < int(season)) | ((num(df.season) == int(season)) & (num(df.week) < int(week)))


def build_team_game_components(pbp):
    x = pbp.copy()
    x = m71.ensure(x, [
        "season", "week", "game_id", "posteam", "defteam", "drive",
        "qb_dropback", "rush_attempt", "pass_attempt", "sack", "scramble",
        "two_point_attempt", "down", "first_down", "third_down_converted",
        "epa", "interception", "fumble_lost", "no_play",
    ])
    x["team"] = x.posteam.map(m71.canon)
    x["defense"] = x.defteam.map(m71.canon)

    no_play = num(x.no_play).fillna(0).eq(1)
    two_pt = num(x.two_point_attempt).fillna(0).eq(1)
    raw_drop = num(x.qb_dropback)
    fallback_drop = num(x.pass_attempt).fillna(0).eq(1) | num(x.sack).fillna(0).eq(1) | num(x.scramble).fillna(0).eq(1)
    x["_dropback"] = np.where(raw_drop.notna(), raw_drop.fillna(0).eq(1), fallback_drop)
    x["_rush"] = num(x.rush_attempt).fillna(0).eq(1)
    x["_opportunity_play"] = (x._dropback | x._rush) & ~two_pt & ~no_play
    x["_official_attempt"] = num(x.official_pass_attempt).fillna(0).eq(1)

    rows = []
    keys = ["season", "week", "game_id", "team", "defense"]
    for key, g0 in x.groupby(keys, sort=True, dropna=False):
        season, week, game_id, team, defense = key
        g = g0[g0._opportunity_play].copy()
        if g.empty or not team or not defense:
            continue
        plays = float(len(g))
        dropbacks = float(g._dropback.sum())
        attempts = float(g._official_attempt.sum())
        drives = num(g.drive).dropna().nunique()
        drives = float(drives) if drives > 0 else np.nan
        down = num(g.down)
        early = g[down.isin([1, 2])]
        thirds = g[down.eq(3)]
        first_down_rate = float(num(g.first_down).fillna(0).mean()) if len(g) else np.nan
        third_conv = float(num(thirds.third_down_converted).fillna(0).mean()) if len(thirds) else np.nan
        early_success = float((num(early.epa) > 0).mean()) if len(early) and num(early.epa).notna().any() else np.nan
        sacks = float(num(g.sack).fillna(0).eq(1).sum())
        turnovers = float(num(g.interception).fillna(0).eq(1).sum() + num(g.fumble_lost).fillna(0).eq(1).sum())
        rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "team": m71.canon(team), "defense": m71.canon(defense),
            "drives": drives,
            "plays": plays,
            "plays_per_drive": safe_div(plays, drives),
            "dropbacks": dropbacks,
            "dropback_rate": safe_div(dropbacks, plays),
            "team_attempts": attempts,
            "attempt_conversion": safe_div(attempts, dropbacks),
            "first_down_rate": first_down_rate,
            "third_down_conversion": third_conv,
            "early_down_success": early_success,
            "sack_rate": safe_div(sacks, dropbacks),
            "turnover_per_drive": safe_div(turnovers, drives),
        })
    out = pd.DataFrame(rows)
    out.sort_values(["season", "week", "game_id", "team"], inplace=True)
    return out.reset_index(drop=True)


def attach_primary_share(passer_games, team_games):
    p = passer_games.copy()
    t = team_games[["season", "week", "game_id", "team", "team_attempts"]].copy()
    p = p.merge(t, on=["season", "week", "game_id", "team"], how="left")
    p["qb_share"] = num(p.attempts) / num(p.team_attempts).replace(0, np.nan)
    p["is_primary"] = False
    if len(p):
        idx = p.groupby(["season", "week", "game_id", "team"])["attempts"].idxmax()
        p.loc[idx, "is_primary"] = True
    return p


def shrink_mean(values, league_mean):
    a = num(values).dropna().to_numpy(dtype=float)
    if not np.isfinite(league_mean):
        league_mean = float(np.mean(a)) if len(a) else np.nan
    if not len(a):
        return league_mean
    return float((a.sum() + LEAGUE_PRIOR_GAMES * league_mean) / (len(a) + LEAGUE_PRIOR_GAMES))


def league_prior(team_games, season, week, col):
    h = team_games[prior_mask(team_games, season, week)]
    a = num(h[col]).dropna()
    return float(a.mean()) if len(a) else np.nan


def component_expectation(team_games, season, week, team, defense, col):
    h = team_games[prior_mask(team_games, season, week)]
    league = float(num(h[col]).dropna().mean()) if len(h) and num(h[col]).notna().any() else np.nan
    off = h[h.team.eq(team)].tail(HISTORY_WINDOW)
    allowed = h[h.defense.eq(defense)].tail(HISTORY_WINDOW)
    off_mean = shrink_mean(off[col], league)
    def_mean = shrink_mean(allowed[col], league)
    vals = [v for v in [off_mean, def_mean] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else league


def qb_share_expectation(passer_share, season, week, pid):
    h_all = passer_share[prior_mask(passer_share, season, week)].copy()
    primary = h_all[h_all.is_primary & num(h_all.qb_share).ge(STABLE_PRIMARY_SHARE)]
    league = float(num(primary.qb_share).dropna().mean()) if len(primary) else 0.97
    q = h_all[
        h_all.passer_id.astype(str).eq(str(pid))
        & h_all.is_primary
        & num(h_all.qb_share).ge(STABLE_PRIMARY_SHARE)
    ].tail(HISTORY_WINDOW)
    if len(q) < MIN_QB_PRIOR:
        return league, int(len(q))
    return shrink_mean(q.qb_share, league), int(len(q))


def get_target_team_game(team_games, r):
    q = team_games[
        team_games.season.eq(int(r.season))
        & team_games.week.eq(int(r.week))
        & team_games.team.eq(m71.canon(r.team))
    ].copy()
    if "game_id" in r.index and pd.notna(r.game_id):
        exact = q[q.game_id.astype(str).eq(str(r.game_id))]
        if len(exact):
            q = exact
    return q.iloc[0] if len(q) else None


def build_atlas(base, pbp):
    passer_games, _, _ = m71.build_game_tables(pbp)
    ident = m71.match_canonical_passers(base, passer_games)
    if float(ident.identity_match_status.isin(["exact", "within2"]).mean()) < 0.98:
        raise RuntimeError("M73 canonical passer identity coverage below 98%")

    team_games = build_team_game_components(pbp)
    passer_share = attach_primary_share(passer_games, team_games)
    rows = []

    for i, r in base.reset_index(drop=True).iterrows():
        season, week = int(r.season), int(r.week)
        team, defense = m71.canon(r.team), m71.canon(r.opponent)
        pid = str(ident.iloc[i].passer_id)
        actual = get_target_team_game(team_games, r)
        if actual is None:
            continue

        rec = r.to_dict()
        rec["passer_id"] = pid
        rec["identity_match_status"] = ident.iloc[i].identity_match_status
        rec["actual_team_attempts_pbp"] = float(actual.team_attempts)

        # Realized target-QB share reconstructs the canonical actual attempts.
        pg = passer_share[
            passer_share.season.eq(season)
            & passer_share.week.eq(week)
            & passer_share.game_id.astype(str).eq(str(actual.game_id))
            & passer_share.team.eq(team)
            & passer_share.passer_id.astype(str).eq(pid)
        ]
        actual_share = float(pg.iloc[0].qb_share) if len(pg) and np.isfinite(pg.iloc[0].qb_share) else safe_div(float(r.actual_attempts), float(actual.team_attempts))

        for c in ["drives", "plays_per_drive", "dropback_rate", "attempt_conversion"] + SURVIVAL:
            rec[f"actual_{c}"] = float(actual[c]) if c in actual.index and np.isfinite(actual[c]) else np.nan
            rec[f"pred_{c}"] = component_expectation(team_games, season, week, team, defense, c)
            rec[f"delta_{c}"] = rec[f"actual_{c}"] - rec[f"pred_{c}"] if np.isfinite(rec[f"actual_{c}"]) and np.isfinite(rec[f"pred_{c}"]) else np.nan

        pred_share, prior_share_games = qb_share_expectation(passer_share, season, week, pid)
        rec["actual_qb_share"] = actual_share
        rec["pred_qb_share"] = pred_share
        rec["delta_qb_share"] = actual_share - pred_share if np.isfinite(actual_share) and np.isfinite(pred_share) else np.nan
        rec["qb_share_prior_games"] = prior_share_games

        vals = [rec.get(f"pred_{c}", np.nan) for c in COMPONENTS]
        actual_vals = [rec.get(f"actual_{c}", np.nan) for c in COMPONENTS]
        if all(np.isfinite(v) for v in vals):
            base_gen = float(np.prod(vals))
        else:
            base_gen = np.nan
        rec["generative_pred_attempts"] = base_gen
        rec["canonical_attempt_residual"] = float(r.actual_attempts - r.pred_attempts)
        rec["generative_attempt_residual"] = float(r.actual_attempts - base_gen) if np.isfinite(base_gen) else np.nan

        contributions = {}
        if np.isfinite(base_gen) and all(np.isfinite(v) for v in actual_vals):
            for j, c in enumerate(COMPONENTS):
                cf = vals.copy(); cf[j] = actual_vals[j]
                contributions[c] = float(np.prod(cf) - base_gen)
                rec[f"contrib_{c}_attempts"] = contributions[c]
            rec["contrib_interaction_remainder_attempts"] = float(r.actual_attempts - base_gen - sum(contributions.values()))
            candidates = {**contributions, "interaction_remainder": rec["contrib_interaction_remainder_attempts"]}
            dom = max(candidates, key=lambda k: abs(candidates[k]))
            rec["dominant_opportunity_mechanism"] = dom
            rec["dominant_opportunity_contribution_attempts"] = candidates[dom]
        else:
            for c in COMPONENTS:
                rec[f"contrib_{c}_attempts"] = np.nan
            rec["contrib_interaction_remainder_attempts"] = np.nan
            rec["dominant_opportunity_mechanism"] = "unavailable"
            rec["dominant_opportunity_contribution_attempts"] = np.nan

        # Exact point-projection oracles. Use implied canonical YPA so the base
        # product is exactly the frozen passing-yard point projection.
        implied_pred_ypa = safe_div(float(r.pred_pass_yards), float(r.pred_attempts))
        rec["implied_pred_ypa"] = implied_pred_ypa
        rec["oracle_actual_attempts_pred_ypa"] = float(r.actual_attempts) * implied_pred_ypa if np.isfinite(implied_pred_ypa) else np.nan
        rec["oracle_pred_attempts_actual_ypa"] = float(r.pred_attempts) * float(r.actual_ypa)
        rec["oracle_actual_attempts_actual_ypa"] = float(r.actual_attempts) * float(r.actual_ypa)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) < 0.98 * len(base):
        raise RuntimeError(f"M73 realized component coverage too low: {len(out)}/{len(base)}")
    recon_diff = np.abs(num(out.actual_attempts) - num(out.actual_team_attempts_pbp) * num(out.actual_qb_share))
    if float((recon_diff <= 0.01).mean()) < 0.98:
        raise RuntimeError("M73 target-QB attempt reconstruction below 98% exact coverage")
    return out


def score_projection(df, col, label, season_label):
    actual = num(df.actual_pass_yards)
    pred = num(df[col])
    err = actual - pred
    return {
        "season": season_label,
        "projection": label,
        "rows": int((actual.notna() & pred.notna()).sum()),
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "bias": float(err.dropna().mean()) if err.notna().any() else np.nan,
        "corr": corr(actual, pred),
        "cat100": int((err.abs() >= 100).sum()),
    }


def oracle_summary(atlas):
    rows = []
    specs = [
        ("pred_pass_yards", "canonical_raw"),
        ("oracle_actual_attempts_pred_ypa", "oracle_perfect_attempts"),
        ("oracle_pred_attempts_actual_ypa", "oracle_perfect_ypa"),
        ("oracle_actual_attempts_actual_ypa", "oracle_perfect_both"),
    ]
    for season in [2024, 2025, "combined"]:
        q = atlas if season == "combined" else atlas[num(atlas.season).eq(int(season))]
        base_mae = mae(num(q.actual_pass_yards), num(q.pred_pass_yards))
        for col, label in specs:
            r = score_projection(q, col, label, season)
            r["mae_gain_vs_raw"] = base_mae - r["mae"]
            rows.append(r)
    return pd.DataFrame(rows)


def attempt_band_summary(atlas):
    rows = []
    bands = {
        "actual_le25": num(atlas.actual_attempts) <= 25,
        "actual_40plus": num(atlas.actual_attempts) >= 40,
        "actual_45plus": num(atlas.actual_attempts) >= 45,
        "actual_50plus": num(atlas.actual_attempts) >= 50,
        "canonical_miss8plus": (num(atlas.actual_attempts) - num(atlas.pred_attempts)).abs() >= LARGE_ATTEMPT_MISS,
        "canonical_miss10plus": (num(atlas.actual_attempts) - num(atlas.pred_attempts)).abs() >= EXTREME_ATTEMPT_MISS,
    }
    for name, mask in bands.items():
        q = atlas[mask].copy()
        rows.append({
            "band": name,
            "games": int(len(q)),
            "attempt_mae": mae(q.actual_attempts, q.pred_attempts),
            "attempt_bias": float((num(q.actual_attempts) - num(q.pred_attempts)).mean()) if len(q) else np.nan,
            "pass_mae": mae(q.actual_pass_yards, q.pred_pass_yards),
            "perfect_attempts_pass_mae": mae(q.actual_pass_yards, q.oracle_actual_attempts_pred_ypa),
            "perfect_attempts_mae_gain": mae(q.actual_pass_yards, q.pred_pass_yards) - mae(q.actual_pass_yards, q.oracle_actual_attempts_pred_ypa) if len(q) else np.nan,
        })
    return pd.DataFrame(rows)


def mechanism_summary(atlas):
    q = atlas[(num(atlas.actual_attempts) - num(atlas.pred_attempts)).abs() >= EXTREME_ATTEMPT_MISS].copy()
    rows = []
    if len(q):
        for mechanism, g in q.groupby("dominant_opportunity_mechanism"):
            rows.append({
                "mechanism": mechanism,
                "games": int(len(g)),
                "share_of_10plus_misses": float(len(g) / len(q)),
                "median_abs_contribution_attempts": float(num(g.dominant_opportunity_contribution_attempts).abs().median()),
                "mean_canonical_attempt_residual": float(num(g.canonical_attempt_residual).mean()),
                "mean_pass_error": float((num(g.actual_pass_yards) - num(g.pred_pass_yards)).mean()),
            })
    return pd.DataFrame(rows).sort_values("games", ascending=False) if rows else pd.DataFrame(columns=["mechanism","games","share_of_10plus_misses","median_abs_contribution_attempts","mean_canonical_attempt_residual","mean_pass_error"])


def survival_summary(atlas):
    err = num(atlas.actual_attempts) - num(atlas.pred_attempts)
    groups = {
        "all": pd.Series(True, index=atlas.index),
        "10plus_underpredicted_attempts": err >= EXTREME_ATTEMPT_MISS,
        "10plus_overpredicted_attempts": err <= -EXTREME_ATTEMPT_MISS,
        "actual_40plus_attempts": num(atlas.actual_attempts) >= 40,
        "actual_le25_attempts": num(atlas.actual_attempts) <= 25,
    }
    rows = []
    for name, mask in groups.items():
        q = atlas[mask]
        for c in SURVIVAL:
            rows.append({
                "group": name,
                "metric": c,
                "games": int(len(q)),
                "mean_actual": float(num(q[f"actual_{c}"]).mean()) if len(q) else np.nan,
                "mean_expected": float(num(q[f"pred_{c}"]).mean()) if len(q) else np.nan,
                "mean_surprise": float(num(q[f"delta_{c}"]).mean()) if len(q) else np.nan,
                "surprise_corr_with_attempt_residual": corr(num(q[f"delta_{c}"]), num(q.actual_attempts) - num(q.pred_attempts)) if len(q) else np.nan,
            })
    return pd.DataFrame(rows)


def interpretation(oracle, mechanisms, atlas):
    comb = oracle[oracle.season.astype(str).eq("combined")]
    raw = float(comb[comb.projection.eq("canonical_raw")].iloc[0].mae)
    att = float(comb[comb.projection.eq("oracle_perfect_attempts")].iloc[0].mae)
    ypa = float(comb[comb.projection.eq("oracle_perfect_ypa")].iloc[0].mae)
    att_gain = raw - att
    ypa_gain = raw - ypa

    if len(mechanisms):
        top = mechanisms.iloc[0]
        top_mech = str(top.mechanism)
        top_share = float(top.share_of_10plus_misses)
        top_med = float(top.median_abs_contribution_attempts)
    else:
        top_mech, top_share, top_med = "none", 0.0, np.nan

    if att_gain >= LARGE_ATTEMPT_HEADROOM_YARDS:
        headroom = "large"
    elif att_gain >= MODERATE_ATTEMPT_HEADROOM_YARDS:
        headroom = "moderate"
    else:
        headroom = "limited"

    concentrated = bool(top_share >= DOMINANT_MECHANISM_SHARE and np.isfinite(top_med) and top_med >= DOMINANT_MECHANISM_MEDIAN_ATTEMPTS)
    if headroom in {"large", "moderate"} and concentrated:
        verdict = f"m73_attempt_headroom_{headroom}_dominant_{top_mech}_target_for_m74"
    elif headroom in {"large", "moderate"}:
        verdict = f"m73_attempt_headroom_{headroom}_mechanisms_diffuse_seek_new_opportunity_information"
    else:
        verdict = "m73_attempt_headroom_limited_deprioritize_attempt_point_correction"

    return pd.DataFrame([{
        "rows": int(len(atlas)),
        "canonical_raw_mae": raw,
        "perfect_attempts_mae": att,
        "perfect_attempts_mae_gain": att_gain,
        "perfect_ypa_mae": ypa,
        "perfect_ypa_mae_gain": ypa_gain,
        "attempt_headroom": headroom,
        "top_10plus_miss_mechanism": top_mech,
        "top_mechanism_share": top_share,
        "top_mechanism_median_abs_attempt_contribution": top_med,
        "dominant_mechanism_concentrated": concentrated,
        "m73_interpretation": verdict,
        "production_actionable": False,
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--history-seasons", default="2023,2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    base = m71.lower(pd.read_csv(a.canonical, low_memory=False))
    if len(base) != 643:
        raise RuntimeError(f"M73 canonical invariant expected 643 rows, got {len(base)}")
    base["team"] = base.team.map(m71.canon)
    base["opponent"] = base.opponent.map(m71.canon)

    seasons = [int(v) for v in a.history_seasons.split(",") if v.strip()]
    pbp, manifest = m71.load_pbp(seasons)
    atlas = build_atlas(base, pbp)
    oracle = oracle_summary(atlas)
    bands = attempt_band_summary(atlas)
    mechanisms = mechanism_summary(atlas)
    survival = survival_summary(atlas)
    interp = interpretation(oracle, mechanisms, atlas)

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(out / "m73_game_component_atlas.csv", index=False)
    atlas[(num(atlas.actual_attempts) - num(atlas.pred_attempts)).abs() >= LARGE_ATTEMPT_MISS].to_csv(out / "m73_large_attempt_miss_atlas.csv", index=False)
    oracle.to_csv(out / "m73_oracle_summary.csv", index=False)
    bands.to_csv(out / "m73_attempt_band_summary.csv", index=False)
    mechanisms.to_csv(out / "m73_mechanism_summary.csv", index=False)
    survival.to_csv(out / "m73_drive_survival_summary.csv", index=False)
    manifest.to_csv(out / "m73_source_manifest.csv", index=False)
    interp.to_csv(out / "m73_precommitted_interpretation.csv", index=False)

    print("=== M73 INTERPRETATION ===")
    print(interp.to_string(index=False))
    print("=== M73 ORACLE SUMMARY ===")
    print(oracle.to_string(index=False))
    print("=== M73 MECHANISM SUMMARY ===")
    print(mechanisms.to_string(index=False))
    print("=== M73 ATTEMPT BANDS ===")
    print(bands.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
