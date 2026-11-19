#!/usr/bin/env python3
"""Which parts of a player's production repeat, and how much weight recent games deserve.

Research only (2026-11-19, REVERSE_ENGINEER_2026_W1_10_V1.md part 2).

  reliability  odd-week vs even-week correlation, per player, of role (snap,
               target, carry and air-yards share; volume per game) and efficiency
               (yards per target, catch rate, yards per carry), current season.
  recency      next-game error when the projection is (1-w)*season-to-date mean
               + w*last-3 mean, for w in 0..1, over three seasons; the w that
               minimizes MAE, overall and per season. History resets on a team
               change. Players who took an offensive snap with no stat row count
               as zeros (snap counts are the denominator).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

STATS = ["targets", "receiving_yards", "carries", "rushing_yards", "receptions"]


def games(season: int) -> pd.DataFrame:
    st = nfl.load_player_stats(seasons=[season]).to_pandas(); st = st[st.season_type == "REG"]
    sn = nfl.load_snap_counts(seasons=[season]).to_pandas()
    sn = sn[sn.offense_snaps.fillna(0) > 0][["player", "week", "team", "offense_pct"]]
    d = sn.rename(columns={"player": "player_display_name"}).merge(st, on=["player_display_name", "week", "team"], how="left")
    d[STATS] = d[STATS].fillna(0)
    d["position"] = d.groupby("player_display_name").position.transform(lambda s: s.ffill().bfill())
    tt = d.groupby(["team", "week"])[["targets", "carries", "receiving_air_yards"]].sum().add_prefix("tm_").reset_index()
    d = d.merge(tt, on=["team", "week"])
    d["tgt_share"] = d.targets / d.tm_targets.replace(0, np.nan)
    d["car_share"] = d.carries / d.tm_carries.replace(0, np.nan)
    d["ay_share"] = d.receiving_air_yards / d.tm_receiving_air_yards.replace(0, np.nan)
    d["season"] = season
    return d.sort_values("week")


def reliability(d: pd.DataFrame) -> pd.DataFrame:
    def r(x, col=None, num=None, den=None):
        halves = []
        for _, g in x.groupby(x.week % 2):
            gg = g.groupby("player_display_name")
            v = gg[col].mean() if col else gg[num].sum() / gg[den].sum().replace(0, np.nan)
            halves.append(v[gg.size() >= 2])
        j = pd.concat(halves, axis=1, keys=["odd", "even"]).dropna()
        return round(j.odd.corr(j.even), 2), len(j)
    spec = {"snap share": dict(col="offense_pct"), "target share": dict(col="tgt_share"),
            "air-yards share": dict(col="ay_share"), "targets per game": dict(col="targets"),
            "yards per target": dict(num="receiving_yards", den="targets"),
            "catch rate": dict(num="receptions", den="targets"), "receiving yards per game": dict(col="receiving_yards"),
            "carry share": dict(col="car_share"), "carries per game": dict(col="carries"),
            "yards per carry": dict(num="rushing_yards", den="carries"), "rushing yards per game": dict(col="rushing_yards")}
    rows = []
    for pos in ["WR", "TE", "RB"]:
        x = d[(d.position == pos) & (d.offense_pct >= .3)]
        for k, kw in spec.items():
            if pos != "RB" and k in ("carry share", "carries per game", "yards per carry", "rushing yards per game"):
                continue
            c, n = r(x, **kw)
            rows.append(dict(position=pos, measure=k, kind="role/volume" if "share" in k or "per game" in k and "yards" not in k else
                             ("efficiency" if k in ("yards per target", "catch rate", "yards per carry") else "outcome"),
                             odd_vs_even_r=c, players=n))
    return pd.DataFrame(rows)


def recency(seasons) -> pd.DataFrame:
    D = []
    for yr in seasons:
        d = games(yr)
        g = d.groupby(["player_display_name", "team"])
        for c in ["targets", "receiving_yards", "carries", "rushing_yards", "tgt_share", "car_share"]:
            d[c + "_szn"] = g[c].transform(lambda s: s.shift().expanding().mean())
            d[c + "_l3"] = g[c].transform(lambda s: s.shift().rolling(3, min_periods=3).mean())
        d["n_prior"] = g.cumcount()
        D.append(d)
    D = pd.concat(D); D = D[D.n_prior >= 4]
    rows, W = [], np.linspace(0, 1, 21)
    for pos, stat in [("WR", "targets"), ("WR", "tgt_share"), ("WR", "receiving_yards"), ("TE", "targets"),
                      ("TE", "receiving_yards"), ("RB", "carries"), ("RB", "car_share"), ("RB", "rushing_yards"), ("RB", "targets")]:
        x = D[D.position == pos].dropna(subset=[stat + "_szn", stat + "_l3"])
        x = x[x.targets_szn >= 3] if pos in ("WR", "TE") else x[x.carries_szn >= 6]
        mae = lambda xx, w: np.abs(xx[stat] - ((1 - w) * xx[stat + "_szn"] + w * xx[stat + "_l3"])).mean()
        best = min(W, key=lambda w: mae(x, w))
        per = " ".join(f"{yr}:{min(W, key=lambda w: mae(xx, w)):.2f}" for yr, xx in x.groupby("season"))
        rows.append(dict(position=pos, stat=stat, n=len(x), mae_season_avg=round(mae(x, 0), 2), mae_last3=round(mae(x, 1), 2),
                         mae_best=round(mae(x, best), 2), best_weight_on_last3=round(best, 2), best_weight_by_season=per))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out-dir", default="data/research/reverse_engineer")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    R = reliability(games(a.season)); R.to_csv(out / "player_stat_reliability_v1.csv", index=False)
    W = recency([a.season - 2, a.season - 1, a.season]); W.to_csv(out / "player_recency_weights_v1.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False)); print(); print(W.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
