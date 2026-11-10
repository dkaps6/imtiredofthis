"""PROSPECTIVE CAPTURE v2 — frozen before kickoff, four arms.

Generalises `capture_w9_prospective_board_v1.py` and adds the D1 arm, which that
capture had to leave pending: `build_injuries_weekly` mislabelled practice-only
pulls as `official_report`, so `status` was null and the injury overlay was inert.
That defect was fixed 2026-11-10 (`classify_report_state`), so D1 is testable for
the first time.

ARMS
  base    production-equivalent. Injury designations have NO effect, which is the
          defect: production's implied retention is OUT 1.00 / DOUBTFUL 1.00.
  d1      A2 from `PRISM_D1_INJURY_STATUS_RESULTS_V1`: opportunity retention by
          designation, **calibrated on strictly prior weeks**, mass DISCARDED.
          A2 cut error on designated players by 67%; A3 (the redistributing
          variant) lost half that gain and its CI crossed zero, so no
          redistribution is applied here.
  d3      phantom-yards fix: a player with zero allocated opportunity scores zero
          yards. `PRISM_D3_PHANTOM_YARDS_RESULTS_V1`.
  d1d3    both. `PRISM_JOINT_D1_D2_D3_RESULTS_V1` found the triple over-corrects
          on rushing and the admissible recommendation is D1 for receiving and D3
          for rushing, so this arm exists to test that split prospectively.

WHY D1 APPLIES AFTER SHARPENING: `_sharpen_wr_target_shares` preserves team WR
target mass, so multiplying before it would hand the freed mass back to other
receivers — that is A3's redistribution, which failed. Applying after sharpening
and before `_allocate_counts` lets the freed opportunity fall into the
allocator's residual bucket and vanish, which is A2.

No sportsbook variable is read anywhere here (AGENTS.md §3).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.simulation_v2 import (
    _allocate_counts,
    _sharpen_rb_rush_shares,
    _sharpen_wr_target_shares,
    _top_n_shares,
)

ITER = 4000
GS = {"tgt_share": 3.0, "rush_share": 3.0, "ypt": 5.0, "ypc": 5.0}
CAP = {"tgt_share": 6.0, "rush_share": 6.0, "ypt": 8.0, "ypc": 8.0}
DEF = {"tgt_share": {"QB": 0.0, "RB": 0.08, "WR": 0.16, "TE": 0.12, "OTHER": 0.06},
       "rush_share": {"QB": 0.14, "RB": 0.32, "WR": 0.02, "TE": 0.00, "OTHER": 0.02},
       "ypt": {"QB": np.nan, "RB": 6.0, "WR": 8.0, "TE": 7.4, "OTHER": 7.0},
       "ypc": {"QB": 5.0, "RB": 4.2, "WR": 6.0, "TE": 4.0, "OTHER": 4.2}}
DESIGNATIONS = ("OUT", "DOUBTFUL", "QUESTIONABLE")


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def grp(p: str) -> str:
    p = str(p).upper().strip()
    if p in ("RB", "FB", "HB"):
        return "RB"
    return p if p in ("QB", "WR", "TE") else "OTHER"


def calibrate_retention(ps: pd.DataFrame, inj: pd.DataFrame, prior_week: int) -> dict:
    """A2 retention by designation from weeks 2..prior_week ONLY.

    The calibration window must exclude the target week or the arm is fitted on
    the thing it predicts.
    """
    tt = ps.groupby(["team", "wk"]).targets.sum().rename("tt").reset_index()
    tc = ps.groupby(["team", "wk"]).carries.sum().rename("tc").reset_index()
    x = ps.merge(tt, on=["team", "wk"]).merge(tc, on=["team", "wk"])
    x["ts"] = np.where(x.tt > 0, x.targets / x.tt, np.nan)
    x["rs"] = np.where(x.tc > 0, x.carries / x.tc, np.nan)
    rows = []
    for wk in sorted(x.wk.dropna().unique()):
        if wk < 2 or wk > prior_week:
            continue
        base = x[x.wk < wk].groupby("k")[["ts", "rs"]].mean().rename(columns={"ts": "b_ts", "rs": "b_rs"})
        cur = x[x.wk == wk].groupby("k")[["ts", "rs"]].mean().rename(columns={"ts": "c_ts", "rs": "c_rs"})
        d = inj[inj.wk == wk].drop_duplicates("k").set_index("k")
        j = d.join(base, how="inner").join(cur, how="left")
        j["c_ts"] = j.c_ts.fillna(0.0)
        j["c_rs"] = j.c_rs.fillna(0.0)
        rows.append(j.reset_index())
    if not rows:
        raise RuntimeError("no prior weeks available to calibrate D1 retention")
    r = pd.concat(rows, ignore_index=True)
    out = {}
    for st in DESIGNATIONS:
        s = r[r.st == st]
        a = s[s.b_ts > 0]
        b = s[s.b_rs > 0]
        out[st] = {
            "tgt": float((a.c_ts / a.b_ts).mean()) if len(a) else 1.0,
            "rush": float((b.c_rs / b.b_rs).mean()) if len(b) else 1.0,
            "n": int(len(s)), "n_tgt": int(len(a)), "n_rush": int(len(b)),
        }
    return out


def build(target_week: int, prior_week: int, include_played: bool = False):
    ps = nfl.load_player_stats(seasons=[2025, 2026]).to_pandas()
    ps = ps[ps.season_type == "REG"].copy()
    ps["k"] = ps.player_display_name.map(norm)
    ps["wk"] = pd.to_numeric(ps.week, errors="coerce")
    ps["season"] = pd.to_numeric(ps.season, errors="coerce")
    for c in ("targets", "receptions", "receiving_yards", "carries", "rushing_yards", "attempts"):
        ps[c] = pd.to_numeric(ps.get(c), errors="coerce").fillna(0.0)

    cur = ps[(ps.season == 2026) & (ps.wk < target_week)]
    # HARD GUARD: nothing at or beyond the target week may ever be read.
    assert cur.wk.max() <= prior_week, f"LOOKAHEAD: read week {cur.wk.max()}"
    pri = ps[ps.season == 2025]

    inj_raw = nfl.load_injuries(seasons=[2026]).to_pandas().copy()
    inj_raw["k"] = inj_raw.full_name.map(norm)
    inj_raw["wk"] = pd.to_numeric(inj_raw.week, errors="coerce")
    inj_raw["st"] = inj_raw.report_status.astype(str).str.upper().str.strip()
    inj = inj_raw[inj_raw.st.isin(DESIGNATIONS)][["k", "wk", "team", "st"]].drop_duplicates(["k", "wk"])

    cur26 = ps[(ps.season == 2026) & (ps.wk <= prior_week)]
    retention = calibrate_retention(cur26, inj[inj.wk <= prior_week], prior_week)
    target_desig = inj[inj.wk == target_week].drop_duplicates("k").set_index("k")

    def rates(df):
        tt = df.groupby(["team", "season", "wk"]).targets.sum().rename("tt").reset_index()
        tc = df.groupby(["team", "season", "wk"]).carries.sum().rename("tc").reset_index()
        x = df.merge(tt, on=["team", "season", "wk"]).merge(tc, on=["team", "season", "wk"])
        x["ts"] = np.where(x.tt > 0, x.targets / x.tt, np.nan)
        x["rs"] = np.where(x.tc > 0, x.carries / x.tc, np.nan)
        a = x.groupby("k").agg(games=("wk", "nunique"), tgt=("targets", "sum"), ry=("receiving_yards", "sum"),
                               car=("carries", "sum"), rsh=("rushing_yards", "sum"),
                               ts=("ts", "mean"), rs=("rs", "mean"))
        o = pd.DataFrame(index=a.index)
        o["games"] = a.games
        o["tgt_share"] = a.ts
        o["rush_share"] = a.rs
        o["ypt"] = np.where(a.tgt > 0, a.ry / a.tgt, np.nan)
        o["ypc"] = np.where(a.car > 0, a.rsh / a.car, np.nan)
        return o

    P, C = rates(pri), rates(cur)

    sc = nfl.load_schedules(seasons=[2026]).to_pandas()
    sc["week"] = pd.to_numeric(sc.week, errors="coerce")
    w = sc[sc.week == target_week].copy()
    # Exclude games already played: a capture is only prospective for games ahead.
    # A capture is only prospective for games still ahead. include_played is for
    # retrospective verification of the arms, never for producing a holdout.
    unplayed = w if include_played else w[w.home_score.isna()]
    excluded = sorted(set(w.away_team) | set(w.home_team)) if unplayed.empty else \
        sorted((set(w.away_team) | set(w.home_team)) - (set(unplayed.away_team) | set(unplayed.home_team)))
    opp = {}
    for _, r in unplayed.iterrows():
        opp[r.home_team] = r.away_team
        opp[r.away_team] = r.home_team
    teams = sorted(opp)

    last = cur.sort_values("wk").groupby("k").tail(1)[["k", "team", "position"]]
    use = cur.groupby("k").agg(t=("targets", "sum"), c=("carries", "sum"), a=("attempts", "sum")).reset_index()
    u = last.merge(use, on="k")
    u = u[(u.t + u.c + u.a) > 0]
    u = u[u.team.isin(teams)].copy()
    u["g"] = u.position.map(grp)
    u["opponent"] = u.team.map(opp)
    u["prior_games"] = P.games.reindex(u.k).fillna(0.0).to_numpy()
    u["current_games"] = C.games.reindex(u.k).fillna(0.0).to_numpy()

    def gm(metric, pos):
        m = (u.g == pos)
        pv = P[metric].reindex(u.k).to_numpy()
        pg = u.prior_games.to_numpy()
        ok = m.to_numpy() & np.isfinite(pv) & (pg > 0)
        if ok.any() and pg[ok].sum() > 0:
            return float(np.average(pv[ok], weights=pg[ok]))
        v = pv[np.isfinite(pv)]
        return float(np.median(v)) if len(v) else float(DEF[metric][pos])

    for metric in GS:
        anchors = {p: gm(metric, p) for p in ("QB", "RB", "WR", "TE", "OTHER")}
        pv = P[metric].reindex(u.k).to_numpy()
        cv = C[metric].reindex(u.k).to_numpy()
        pg = u.prior_games.to_numpy()
        cg = u.current_games.to_numpy()
        est = []
        for i in range(len(u)):
            wts = [GS[metric]]
            vals = [anchors[u.g.iloc[i]]]
            if np.isfinite(pv[i]) and pg[i] > 0:
                wts.append(min(pg[i], CAP[metric])); vals.append(pv[i])
            if np.isfinite(cv[i]) and cg[i] > 0:
                wts.append(cg[i]); vals.append(cv[i])
            est.append(float(np.average(vals, weights=wts)))
        u[metric] = est

    _d = target_desig.st.astype("string") if len(target_desig) else pd.Series(dtype="string")
    u["designation"] = u.k.map(_d).astype("object").where(lambda x: x.notna(), "")
    u["ret_tgt"] = u.designation.map(lambda s: retention[s]["tgt"] if s in retention else 1.0).fillna(1.0)
    u["ret_rush"] = u.designation.map(lambda s: retention[s]["rush"] if s in retention else 1.0).fillna(1.0)

    tp = cur.groupby(["team", "wk"]).agg(a=("attempts", "sum"), c=("carries", "sum")).reset_index()
    tp["plays"] = tp.a + tp.c
    agg = tp.groupby("team").agg(plays=("plays", "mean"), att=("a", "mean")).reset_index()
    agg["pass_rate"] = (agg.att / agg.plays).clip(0.35, 0.75)
    agg["plays"] = agg.plays.clip(50, 80)
    u = u.merge(agg[["team", "plays", "pass_rate"]], on="team", how="left")
    u["plays"] = u.plays.fillna(64.0)
    u["pass_rate"] = u.pass_rate.fillna(0.58)
    return u, retention, excluded, int(len(target_desig))


def project(u: pd.DataFrame, target_week: int, d1: bool, d3: bool) -> pd.DataFrame:
    out = []
    for tm, g in u.groupby("team"):
        g = g.reset_index(drop=True)
        rng = np.random.default_rng(abs(hash((tm, target_week))) % (2 ** 32))
        plays = int(round(float(g.plays.iloc[0])))
        pr = float(g.pass_rate.iloc[0])
        pa = rng.binomial(plays, pr, ITER)
        ra = plays - pa
        tsh = _sharpen_wr_target_shares(g, g.tgt_share.to_numpy())
        rsh = _sharpen_rb_rush_shares(g, _top_n_shares(g.rush_share.to_numpy(), 5))
        if d1:
            # A2: applied AFTER sharpening so freed mass falls into the
            # allocator's residual bucket and is discarded, not redistributed.
            tsh = tsh * g.ret_tgt.to_numpy()
            rsh = rsh * g.ret_rush.to_numpy()
        T = _allocate_counts(rng, pa, tsh)
        C = _allocate_counts(rng, ra, rsh)
        pes = np.clip(rng.normal(1.0, 0.09, ITER), 0.65, 1.35)
        res = np.clip(rng.normal(1.0, 0.10, ITER), 0.60, 1.40)
        for j in range(len(g)):
            ypt = g.ypt.iloc[j]
            ypt = 7.5 if not np.isfinite(ypt) or ypt <= 0 else ypt
            ypc = g.ypc.iloc[j]
            ypc = 4.2 if not np.isfinite(ypc) or ypc <= 0 else ypc
            t = T[:, j]
            c = C[:, j]
            rsd = np.maximum(6.0, np.sqrt(np.maximum(t, 1)) * ypt * 0.55)
            csd = np.maximum(3.0, np.sqrt(np.maximum(c, 1)) * ypc * 0.65)
            ry = np.clip(rng.normal(t * ypt * pes, rsd), 0.0, None)
            cy = np.clip(rng.normal(c * ypc * res, csd), 0.0, None)
            if d3:
                ry = np.where(t > 0, ry, 0.0)
                cy = np.where(c > 0, cy, 0.0)
            out.append({"k": g.k.iloc[j], "rec_yards": float(ry.mean()), "rush_yards": float(cy.mean()),
                        "targets": float(t.mean()), "carries": float(c.mean())})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out-dir", default="data/market_track_record")
    ap.add_argument("--dry-run", action="store_true", help="print only, write nothing")
    ap.add_argument("--include-played", action="store_true",
                    help="retrospective verification only; NOT a valid holdout")
    a = ap.parse_args()
    week = int(a.week)
    prior = week - 1

    u, retention, excluded, n_desig = build(week, prior, include_played=a.include_played)
    print(f"W{week} universe: {len(u)} players across {u.team.nunique()} teams")
    if excluded:
        print(f"teams excluded (game already played): {excluded}")
    print(f"week {week} designations available: {n_desig} players")
    print(f"designated in universe: {int(u.designation.ne('').sum())} "
          f"({dict(u.designation[u.designation.ne('')].value_counts())})")
    print("A2 retention (calibrated weeks 2-%d): " % prior
          + json.dumps({k: {"tgt": round(v["tgt"], 4), "rush": round(v["rush"], 4), "n": v["n"]}
                        for k, v in retention.items()}))

    arms = {"base": (False, False), "d1": (True, False), "d3": (False, True), "d1d3": (True, True)}
    frames = {}
    for name, (f1, f3) in arms.items():
        frames[name] = project(u, week, f1, f3)

    m = u[["k", "team", "opponent", "position", "designation", "ret_tgt", "ret_rush"]].copy()
    m.insert(0, "player", u.k)
    for name, fr in frames.items():
        m = m.merge(fr.rename(columns={"rec_yards": f"rec_yards_{name}", "rush_yards": f"rush_yards_{name}",
                                       "targets": f"targets_{name}", "carries": f"carries_{name}"}),
                    on="k", how="left")
    m["week"] = week
    m["season"] = 2026
    m["captured_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    meta = {
        "season": 2026, "week": week, "captured_utc": m.captured_utc.iloc[0],
        "players": int(len(m)), "teams": int(m.team.nunique()),
        "max_player_stats_week_read": prior,
        "designations_available": n_desig,
        "designated_players_in_universe": int(u.designation.ne("").sum()),
        "teams_excluded_already_played": excluded,
        "include_played": bool(a.include_played),
        "arms_frozen": sorted(arms),
        "d1_retention_calibration_window": f"2026 weeks 2-{prior}",
        "d1_retention": retention,
        "d1_semantics": "A2 — mass discarded, applied after sharpening, no redistribution",
        "totals": {f"{mk}_{arm}": float(m[f"{mk}_{arm}"].sum())
                   for arm in arms for mk in ("rec_yards", "rush_yards")},
    }
    print("\ntotals:\n" + json.dumps(meta["totals"], indent=2))
    if a.dry_run:
        print("\n[dry-run] nothing written")
        return 0
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"w{week}_prospective_frozen_v2.csv"
    js = out / f"w{week}_prospective_frozen_v2.json"
    m.to_csv(csv, index=False)
    js.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nfrozen -> {csv} ({len(m)} rows)")
    print(f"meta   -> {js}")
    print(f"captured_utc = {meta['captured_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
