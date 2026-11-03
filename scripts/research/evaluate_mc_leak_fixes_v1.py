"""MC opportunity leak: evaluate C1/F0/E1/E2/E3 allocation rules.
Preregistration: docs/research/PRISM_MC_LEAK_FIX_PREREGISTRATION_V1.md (frozen first).
Scores the ALLOCATION RULE in expectation. No MC run, no production code modified.
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd, nflreadpy as nfl

U7 = "data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv"
U8 = "data/market_track_record/explanatory/2026_wk08_trace__data__football_simulation_universe.csv"
CANDS = ["rules_tgt_share", "bayes_tgt_share", "target_share", "tgt_share"]
CAP, E2_THRESH, E3_N, MIN_WEEKS, MIN_PLAYERS = 0.95, 0.5, 7, 2, 100


def norm(s): return re.sub(r"[^a-z]", "", str(s).lower())


def eff_share(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    filled = pd.Series(False, index=df.index)
    for c in CANDS:
        if c not in df.columns: continue
        v = pd.to_numeric(df[c], errors="coerce")
        take = (~filled) & v.notna()
        out[take] = v[take]; filled |= take
    return out.clip(0.0, CAP)


def c1_normalise(s: pd.Series, team: pd.Series) -> pd.Series:
    """clip then scale the team sum down to CAP if it exceeds it."""
    s = s.clip(0.0, CAP)
    tot = s.groupby(team).transform("sum")
    factor = np.where(tot > CAP, CAP / tot.replace(0, np.nan), 1.0)
    return (s * pd.Series(factor, index=s.index)).fillna(0.0)


def build() -> pd.DataFrame:
    d = nfl.load_player_stats(seasons=[2026], summary_level="week")
    ps = d.to_pandas() if hasattr(d, "to_pandas") else d
    if "season_type" in ps.columns: ps = ps[ps.season_type == "REG"]
    ps["k"] = ps.player_display_name.map(norm)
    ps["week"] = pd.to_numeric(ps.week, errors="coerce")
    ps["targets"] = pd.to_numeric(ps.targets, errors="coerce").fillna(0.0)

    frames = []
    for path, W in ((U7, 7), (U8, 8)):
        u = pd.read_csv(path, low_memory=False)
        u["k"] = u.player.map(norm); u["wk"] = W
        u["share"] = eff_share(u)
        # availability from weeks 1..W-1 only
        hist = ps[ps.week < W]
        wks = max(1, hist.week.nunique())
        hit = hist[hist.targets >= 1].groupby(["k", "team"]).week.nunique().rename("hits").reset_index()
        u = u.merge(hit, on=["k", "team"], how="left")
        u["A"] = (u.hits.fillna(0.0) / wks).clip(0, 1)
        # actuals for week W
        wk = ps[ps.week == W]
        tt = wk.groupby("team").targets.sum().rename("team_tgt")
        act = wk.groupby(["k", "team"]).targets.sum().rename("act_tgt").reset_index()
        u = u.merge(act, on=["k", "team"], how="left").merge(tt, on="team", how="left")
        u["act_tgt"] = u.act_tgt.fillna(0.0)
        u["actual"] = np.where(u.team_tgt > 0, u.act_tgt / u.team_tgt, np.nan)
        frames.append(u[["k", "player", "team", "wk", "share", "A", "act_tgt", "actual"]])
    a = pd.concat(frames, ignore_index=True).dropna(subset=["actual"])

    key = a.team + "_" + a.wk.astype(str)
    a["C1"] = c1_normalise(a.share, key)
    a["F0"] = c1_normalise(a.share.where(a.A > 0, 0.0), key)
    a["E1"] = c1_normalise(a.share * a.A, key)
    a["E2"] = c1_normalise(a.share.where(a.A >= E2_THRESH, 0.0), key)
    rank = a.groupby(key).share.rank(method="first", ascending=False)
    a["E3"] = c1_normalise(a.share.where(rank <= E3_N, 0.0), key)
    return a


def boot(df, fn, iters=10000, seed=47):
    rng = np.random.default_rng(seed)
    return np.array([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(iters)])
