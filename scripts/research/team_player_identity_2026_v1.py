#!/usr/bin/env python3
"""Team identities, player roles inside them, how Vegas sets player lines, and whether
Vegas's misses repeat. 2026 weeks 1-10. Research only, descriptive + persistence tests.

Outputs data/research/identity/:
  team_profiles_2026_v1.csv          one row per team
  team_metric_stability_v1.csv       odd-week vs even-week correlation of each team metric
  player_roles_2026_v1.csv           usage/production per player with team context
  vegas_line_model_v1.csv            what FanDuel lines are built from (regressions)
  line_residual_persistence_v1.csv   do player/team/opponent misses vs the line repeat?
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

OUT = Path("data/research/identity")


def team_games(p: pd.DataFrame) -> pd.DataFrame:
    x = p[((p["pass"] == 1) | (p["rush"] == 1)) & (p.qb_kneel != 1) & (p.qb_spike != 1) & p.posteam.notna()].copy()
    x["neutral"] = x.wp.between(.2, .8) & (x.down <= 3) & (x.half_seconds_remaining > 120)
    x["lead8"] = x.score_differential >= 8; x["trail8"] = x.score_differential <= -8
    g = x.groupby(["week", "game_id", "posteam", "defteam"])
    T = g.agg(plays=("pass", "size"), dropbacks=("pass", "sum"), rushes=("rush", "sum"), epa=("epa", "mean"),
              success=("success", "mean")).reset_index()
    for nm, m in [("neutral", x.neutral), ("lead8", x.lead8), ("trail8", x.trail8)]:
        q = x[m].groupby(["game_id", "posteam"]).agg(**{f"pr_{nm}": ("pass", "mean"), f"n_{nm}": ("pass", "size")})
        T = T.merge(q.reset_index(), on=["game_id", "posteam"], how="left")
    pr = x[x.neutral].groupby(["game_id", "posteam"]).pass_oe.mean().rename("proe")
    T = T.merge(pr.reset_index(), on=["game_id", "posteam"], how="left")
    d = x.groupby(["game_id", "defteam"]).epa.mean().rename("def_epa_allowed").reset_index().rename(columns={"defteam": "posteam"})
    T = T.merge(d, on=["game_id", "posteam"], how="left")
    s = nfl.load_schedules(seasons=[2026]).to_pandas()[["game_id", "home_team", "home_score", "away_score", "spread_line", "total_line"]]
    T = T.merge(s, on="game_id", how="left")
    h = T.posteam == T.home_team
    T["pts"] = np.where(h, T.home_score, T.away_score); T["opp_pts"] = np.where(h, T.away_score, T.home_score)
    T["margin"] = T.pts - T.opp_pts; T["fav_by"] = np.where(h, T.spread_line, -T.spread_line)
    T["implied"] = (T.total_line + T.fav_by) / 2
    T["ats"] = T.margin - T.fav_by; T["pts_vs_implied"] = T.pts - T.implied
    T["lead_share"] = T.n_lead8 / T.plays; T["trail_share"] = T.n_trail8 / T.plays
    return T


def concentration(st: pd.DataFrame) -> pd.DataFrame:
    tt = st.groupby(["team", "week"])[["targets", "carries"]].sum().add_prefix("tm_").reset_index()
    s = st.merge(tt, on=["team", "week"])
    tot = s.groupby(["team", "player_id", "player_display_name", "position"])[["targets", "carries", "tm_targets", "tm_carries"]].sum().reset_index()
    tot["tgt_share"] = tot.targets / tot.tm_targets; tot["car_share"] = tot.carries / tot.tm_carries
    rows = []
    for t, g in tot.groupby("team"):
        ts = g.sort_values("tgt_share", ascending=False); rb = g[g.position == "RB"].sort_values("car_share", ascending=False)
        rows.append(dict(team=t, top_target=ts.player_display_name.iloc[0], top_tgt_share=ts.tgt_share.iloc[0],
                         top2_tgt_share=ts.tgt_share.iloc[:2].sum(), n_core_receivers=int((ts.tgt_share >= 0.15).sum()),
                         lead_rb=rb.player_display_name.iloc[0] if len(rb) else "", lead_rb_car_share=rb.car_share.iloc[0] if len(rb) else np.nan))
    return pd.DataFrame(rows)


def stability(T: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        o = T[T.week % 2 == 1].groupby("posteam")[c].mean(); e = T[T.week % 2 == 0].groupby("posteam")[c].mean()
        rows.append(dict(metric=c, odd_vs_even_r=round(o.corr(e), 2)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--graded", required=True, help="FanDuel graded board, both sides (board_all_fd)")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    pbp = nfl.load_pbp(seasons=[2026]).to_pandas(); pbp = pbp[(pbp.season_type == "REG") & (pbp.week <= 10)]
    T = team_games(pbp)
    metrics = ["epa", "def_epa_allowed", "proe", "plays", "rushes", "dropbacks", "pr_lead8", "pr_trail8", "margin", "ats", "pts_vs_implied", "lead_share"]
    S = stability(T, metrics); S.to_csv(OUT / "team_metric_stability_v1.csv", index=False)
    st = nfl.load_player_stats(seasons=[2026]).to_pandas(); st = st[(st.season_type == "REG") & (st.week <= 10)]
    P = T.groupby("posteam").agg(games=("game_id", "nunique"), wins=("margin", lambda m: int((m > 0).sum())), pt_diff_pg=("margin", "mean"),
                                 off_epa=("epa", "mean"), def_epa_allowed=("def_epa_allowed", "mean"), proe=("proe", "mean"),
                                 plays_pg=("plays", "mean"), rushes_pg=("rushes", "mean"), pr_lead8=("pr_lead8", "mean"), pr_trail8=("pr_trail8", "mean"),
                                 lead_share=("lead_share", "mean"), trail_share=("trail_share", "mean"), fav_by=("fav_by", "mean"),
                                 implied_pts=("implied", "mean"), pts_pg=("pts", "mean"), ats_pg=("ats", "mean"), pts_vs_implied=("pts_vs_implied", "mean")).reset_index().rename(columns={"posteam": "team"})
    P = P.merge(concentration(st), on="team", how="left")
    P["identity"] = np.select([P.proe >= 3, P.proe <= -3], ["pass-first", "run-first"], "balanced")
    P["quality"] = np.select([(P.off_epa > 0.05) & (P.def_epa_allowed < 0), (P.off_epa < -0.05) & (P.def_epa_allowed > 0)], ["good both sides", "bad both sides"],
                             np.where(P.off_epa > P.off_epa.median(), "offense-led", "defense-led"))
    P.round(3).to_csv(OUT / "team_profiles_2026_v1.csv", index=False)
    # players: usage + production + team context
    tt = st.groupby(["team", "week"])[["targets", "carries"]].sum().add_prefix("tm_").reset_index()
    s = st.merge(tt, on=["team", "week"])
    pl = s.groupby(["player_display_name", "team", "position"]).agg(games=("week", "size"), tgt_share=("targets", "sum"), tm_t=("tm_targets", "sum"),
                                                                     car_share=("carries", "sum"), tm_c=("tm_carries", "sum"), rec_ypg=("receiving_yards", "mean"),
                                                                     rush_ypg=("rushing_yards", "mean"), fp=("fantasy_points_ppr", "mean")).reset_index()
    pl["tgt_share"] = pl.tgt_share / pl.tm_t; pl["car_share"] = pl.car_share / pl.tm_c
    pl = pl.merge(P[["team", "pt_diff_pg", "off_epa", "quality", "identity"]], on="team", how="left")
    pl = pl[pl.games >= 4].drop(columns=["tm_t", "tm_c"]); pl.round(3).to_csv(OUT / "player_roles_2026_v1.csv", index=False)
    # Vegas: how lines are built
    b = pd.read_csv(a.graded); b = b[(b.side == "OVER") & (b.week >= 6)].copy()
    stt = st.copy(); stt["t"] = stt.week
    feats = []
    for mk, col in [("rec_yards", "receiving_yards"), ("rush_yards", "rushing_yards"), ("receptions", "receptions"), ("pass_yards", "passing_yards")]:
        d = b[b.market == mk]
        for w in sorted(d.week.unique()):
            pr = stt[stt.week < w]; avg = pr.groupby("player_display_name")[col].mean(); l3 = pr[pr.week >= w - 3].groupby("player_display_name")[col].mean()
            q = d[d.week == w].copy(); q["season_avg"] = q.player.map(avg); q["last3_avg"] = q.player.map(l3); feats.append(q)
    F = pd.concat(feats).dropna(subset=["season_avg", "last3_avg", "fav_by", "total_line"])
    F["implied"] = (F.total_line + F.fav_by) / 2
    V = []
    for mk, g in F.groupby("market"):
        X = np.c_[np.ones(len(g)), g.season_avg, g.last3_avg, g.implied - 22]
        bta, *_ = np.linalg.lstsq(X, g.vegas_line, rcond=None); pred = X @ bta
        r2 = 1 - ((g.vegas_line - pred) ** 2).sum() / ((g.vegas_line - g.vegas_line.mean()) ** 2).sum()
        r2_avg = np.corrcoef(g.vegas_line, g.season_avg)[0, 1] ** 2
        V.append(dict(market=mk, n=len(g), r2_full=r2, r2_season_avg_only=r2_avg, w_season=bta[1], w_last3=bta[2], per_implied_pt=bta[3],
                      line_vs_season_avg_ratio=(g.vegas_line / g.season_avg.replace(0, np.nan)).median()))
    pd.DataFrame(V).round(3).to_csv(OUT / "vegas_line_model_v1.csv", index=False)
    # persistence of misses vs the line
    a2 = pd.read_csv(a.graded); a2 = a2[a2.side == "OVER"].copy(); a2["res"] = (a2.actual - a2.vegas_line) / a2.vegas_line.clip(lower=1)
    a2["beat"] = (a2.actual > a2.vegas_line).astype(float)
    a2 = a2.merge(nfl.load_schedules(seasons=[2026]).to_pandas()[["week", "home_team", "away_team"]].melt(id_vars="week", value_name="tm")[["week", "tm"]], left_on=["week", "team"], right_on=["week", "tm"], how="left")
    R = []
    for unit in ["player", "team", "opponent"]:
        for mk in ["all", "rec_yards", "rush_yards", "receptions"]:
            d = a2 if mk == "all" else a2[a2.market == mk]
            for metric in ["res", "beat"]:
                o = d[d.week % 2 == 1].groupby(unit)[metric].agg(["mean", "size"]); e = d[d.week % 2 == 0].groupby(unit)[metric].agg(["mean", "size"])
                j = o.join(e, lsuffix="_o", rsuffix="_e", how="inner"); j = j[(j.size_o >= 2) & (j.size_e >= 2)]
                if len(j) >= 10:
                    R.append(dict(unit=unit, market=mk, metric=metric, n_units=len(j), odd_vs_even_r=round(j.mean_o.corr(j.mean_e), 3)))
    pd.DataFrame(R).to_csv(OUT / "line_residual_persistence_v1.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(S.to_string(index=False)); print(pd.DataFrame(V).round(3).to_string(index=False)); print(pd.DataFrame(R).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
