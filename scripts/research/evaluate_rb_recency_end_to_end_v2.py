#!/usr/bin/env python3
"""RB_CARRY_SHARE_RECENCY_END_TO_END_PREREGISTRATION_V2.md, as frozen.

C0  = production before recency (production's own weekly normalizer and season totals).
C1u = RB/FB only, 0.15 x season-to-date + 0.85 x last-3 snap games, ungated.
C1g = C1u with the anomaly gate: the production candidate.
All three run through production's Bayes helpers, then top 5, clip 0.95 and scale to 0.95,
times actual team carries. The window, labels and blend come from
scripts/modeling/rb_carry_recency_v2.py, the same module production uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

import scripts.modeling.rb_carry_recency_v2 as rec
import scripts.player_form_v2 as pf
from scripts._opponent_map import canon_team
from scripts.modeling.bayesian_v2 import _posterior_mean, _position_group, _weighted_group_mean
from scripts.simulation_v2 import _top_n_shares

POS = {"QB", "RB", "FB", "WR", "TE"}


def _pd(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def logs_for(season: int) -> pd.DataFrame:
    raw = _pd(nfl.load_player_stats(seasons=[season], summary_level="week"))
    return pf._normalize_weekly(raw, season)


def posterior(U: pd.DataFrame, cur_col: str) -> np.ndarray:
    df = U.rename(columns={"rush_share_prior": "rush_share_prior"}).copy()
    df["bayes_position_group"] = df["position"].map(_position_group)
    gm = {p: _weighted_group_mean(df, "rush_share", p) for p in df["bayes_position_group"].unique()}
    out = []
    for _, r in df.iterrows():
        pv, cv = r["rush_share_prior"], r[cur_col]
        m, _ = _posterior_mean(gm[r["bayes_position_group"]],
                               float(pv) if pd.notna(pv) else np.nan, float(r["prior_games"]),
                               float(cv) if pd.notna(cv) else np.nan, float(r["current_games"]), "rush_share")
        out.append(float(np.clip(m, 0, 1)))
    return np.asarray(out)


def chain(U: pd.DataFrame, post: np.ndarray) -> np.ndarray:
    fin = np.zeros(len(U))
    for _, idx in U.groupby("team").groups.items():
        loc = U.index.get_indexer(idx)
        s = np.clip(_top_n_shares(post[loc], 5), 0, 0.95)
        if s.sum() > 0.95:
            s = s * 0.95 / s.sum()
        fin[loc] = s
    return fin


def run_season(season: int, weeks, players: pd.DataFrame) -> pd.DataFrame:
    prior_logs, cur_logs = logs_for(season - 1), logs_for(season)
    prior_tot = pf._season_totals(prior_logs).set_index("player_identity_key")
    gsis_to_key = (pd.concat([prior_logs, cur_logs])[["player_id", "player_identity_key"]]
                   .query("player_id != ''").drop_duplicates("player_id").set_index("player_id")["player_identity_key"])
    pfr_to_key = players.assign(key=players["gsis_id"].map(gsis_to_key)).dropna(subset=["key"]).set_index("pfr_id")["key"].to_dict()

    snaps = _pd(nfl.load_snap_counts(seasons=[season]))
    snaps = snaps[snaps["game_type"].eq("REG")]
    inj = _pd(nfl.load_injuries(seasons=[season]))
    inj["key"] = inj["gsis_id"].map(gsis_to_key)
    inj_k = inj.dropna(subset=["key"])
    out_inj = inj[inj["report_status"].astype(str).str.lower().isin(["out", "doubtful"])][["gsis_id", "week"]]
    ro = _pd(nfl.load_rosters_weekly(seasons=[season]))
    ro = ro[(ro["status"] == "ACT") & ro["position"].isin(POS)][["gsis_id", "week", "team", "position"]].dropna(subset=["gsis_id"])
    ro["position"] = ro["position"].replace({"FB": "RB"})
    ro["team"] = ro["team"].astype(str).map(canon_team)

    logs = cur_logs.rename(columns={"player_identity_key": "key"})
    stats = _pd(nfl.load_player_stats(seasons=[season], summary_level="week"))
    stats = stats[stats["season_type"].eq("REG")].copy()
    stats["team"] = stats["team"].astype(str).map(canon_team)
    act = stats.groupby(["player_id", "week"])["carries"].sum()
    team_c = stats.groupby(["team", "week"])["carries"].sum()
    snap_played = set(zip(snaps.loc[snaps["offense_snaps"].fillna(0) > 0, "pfr_player_id"].map(
        players.set_index("pfr_id")["gsis_id"].to_dict()), snaps.loc[snaps["offense_snaps"].fillna(0) > 0, "week"]))

    rows = []
    for w in weeks:
        U = ro[ro["week"] == w]
        U = U[~U.set_index(["gsis_id", "week"]).index.isin(out_inj.set_index(["gsis_id", "week"]).index)]
        U = U.drop_duplicates(["gsis_id"]).copy()
        if U.empty:
            continue
        U["key"] = U["gsis_id"].map(gsis_to_key).fillna("gsis:" + U["gsis_id"])
        U = U.drop_duplicates("key").reset_index(drop=True)
        cur_tot = pf._season_totals(cur_logs[cur_logs["week"] < w].copy()).set_index("player_identity_key")
        U["rush_share_prior"] = U["key"].map(prior_tot["rush_share"])
        U["prior_games"] = U["key"].map(prior_tot["games"]).fillna(0)
        U["rush_share_current"] = U["key"].map(cur_tot["rush_share"])
        U["current_games"] = U["key"].map(cur_tot["games"]).fillna(0)

        hist = logs[logs["week"] < w]
        games = rec.snap_games(snaps, pfr_to_key, hist, w)
        team_weeks = hist[["team", "week"]].drop_duplicates()
        labeled = rec.label_games(games, team_weeks, inj_k[inj_k["week"] < w + 1])
        win = rec.last3(labeled, U[["key", "team"]])
        g = rec.apply(U.rename(columns={"key": "player_identity_key"}), win)
        ungated = win.assign(window_clean=True)
        u = rec.apply(U.rename(columns={"key": "player_identity_key"}), ungated)
        U["cur_c0"] = U["rush_share_current"]
        U["cur_c1u"] = u["rush_share_current"].to_numpy()
        U["cur_c1g"] = g["rush_share_current"].to_numpy()
        U["status_c1g"] = g["rb_carry_recency_status"].to_numpy()
        U["window_states"] = g["rb_carry_recency_window_states"].to_numpy()
        for arm in ("c0", "c1u", "c1g"):
            U[f"share_{arm}"] = chain(U, posterior(U, f"cur_{arm}"))
        U["team_carries"] = [team_c.get((t, w), np.nan) for t in U["team"]]
        S = U[U["position"].eq("RB") & [(gid, w) in snap_played for gid in U["gsis_id"]]].copy()
        S["carries"] = [float(act.get((gid, w), 0.0)) for gid in S["gsis_id"]]
        S["week"], S["season"] = w, season
        rows.append(S)
    return pd.concat(rows, ignore_index=True)


def score(D: pd.DataFrame, arm: str, rng) -> dict:
    out = {}
    for season, x in D.groupby("season"):
        e0, e1 = (x.carries - x.proj_c0).abs(), (x.carries - x[f"proj_{arm}"]).abs()
        gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
        bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum()
              for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
        lead = x.loc[x.groupby("gid").proj_c0.idxmax()]
        l0, l1 = (lead.carries - lead.proj_c0).abs(), (lead.carries - lead[f"proj_{arm}"]).abs()
        lg = pd.DataFrame({"gid": lead.gid, "d": l0 - l1}).groupby("gid").d.sum()
        lbs = [lg.to_numpy()[rng.integers(0, len(lg), len(lg))].mean() for _ in range(2000)]
        out[season] = dict(arm=arm, season=season, n=len(x), mae_c0=e0.mean(), mae_arm=e1.mean(),
                           gain=e0.mean() - e1.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                           lead_n=len(lead), lead_gain=l0.mean() - l1.mean(),
                           lead_ci_lo=np.percentile(lbs, 2.5), lead_ci_hi=np.percentile(lbs, 97.5),
                           rows_changed=int((x[f"proj_{arm}"] - x.proj_c0).abs().gt(1e-9).sum()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/rb_recency_end_to_end_v2.csv")
    a = ap.parse_args()
    players = _pd(nfl.load_players())[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    D = pd.concat([run_season(2025, range(4, 19), players), run_season(2026, range(4, 11), players)], ignore_index=True)
    D = D.dropna(subset=["team_carries"])
    for arm in ("c0", "c1u", "c1g"):
        D[f"proj_{arm}"] = D[f"share_{arm}"] * D.team_carries
    D["gid"] = D.season.astype(str) + "_" + D.week.astype(str) + "_" + D.team
    rng = np.random.default_rng(20261124)
    R = pd.DataFrame([r for arm in ("c1u", "c1g") for r in score(D, arm, rng).values()])
    gate = D.groupby(["season", "status_c1g"]).size().unstack(fill_value=0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    R.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(gate.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
