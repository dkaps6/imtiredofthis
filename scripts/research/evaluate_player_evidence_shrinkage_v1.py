"""Player-evidence shrinkage: score the five frozen candidates.

Preregistration: docs/research/PRISM_PLAYER_EVIDENCE_SHRINKAGE_PREREGISTRATION_V1.md
Frozen at 5069b9bc. No candidate may be added.

C0 calls the REAL scripts.modeling.bayesian_v2._posterior_mean so the baseline is
production's actual arithmetic, not a reimplementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.modeling import bayesian_v2 as bv

METRIC = "rush_share"
SEASONS = list(range(2013, 2026))
RB = {"RB", "HB", "TB", "FB"}
MIN_BACKS = 2
MIN_TEAM_CARRIES = 10
BOOT, SEED = 4000, 42
C4_GROUP_STRENGTH = 1.0


def _is_rb(p: object) -> bool:
    s = str(p).strip().upper()
    return s in RB or s.startswith("RB")


def build_panel() -> pd.DataFrame:
    import nflreadpy as nfl
    w = nfl.load_player_stats(seasons=SEASONS).to_pandas()
    w = w[["season", "week", "team", "player_id", "position", "carries"]].copy()
    w = w[w["position"].map(_is_rb)]
    w["carries"] = pd.to_numeric(w["carries"], errors="coerce").fillna(0.0)
    w["week"] = pd.to_numeric(w["week"], errors="coerce")
    w = w[w["week"].notna()]
    w["week"] = w["week"].astype(int)

    tw = (w.groupby(["season", "team", "week"], as_index=False)["carries"].sum()
            .rename(columns={"carries": "team_carries"}))
    w = w.merge(tw, on=["season", "team", "week"], how="left")
    w = w[w["team_carries"] >= MIN_TEAM_CARRIES].copy()
    w["game_share"] = w["carries"] / w["team_carries"]

    # prior-SEASON aggregates per (player, team, season)
    pri = (w.groupby(["season", "team", "player_id"])
             .agg(p_games=("game_share", "size"), p_mean=("game_share", "mean"),
                  p_var=("game_share", "var"))
             .reset_index())
    pri["season"] = pri["season"] + 1          # becomes the PRIOR season for the next year
    pri = pri.rename(columns={"p_games": "prior_games", "p_mean": "prior_value",
                              "p_var": "prior_var"})

    # current-season-to-date, strictly prior weeks only
    w = w.sort_values(["season", "team", "player_id", "week"])
    g = w.groupby(["season", "team", "player_id"])["game_share"]
    w["cur_n"] = g.cumcount()
    w["cur_sum"] = g.cumsum() - w["game_share"]
    w["cur_sumsq"] = (w["game_share"] ** 2).groupby(
        [w["season"], w["team"], w["player_id"]]).cumsum() - w["game_share"] ** 2
    w["current_games"] = w["cur_n"]
    w["current_value"] = np.where(w["cur_n"] > 0, w["cur_sum"] / w["cur_n"].replace(0, np.nan), np.nan)
    with np.errstate(invalid="ignore"):
        ex2 = w["cur_sumsq"] / w["cur_n"].replace(0, np.nan)
        w["current_var"] = (ex2 - w["current_value"] ** 2).clip(lower=0.0)

    w = w.merge(pri, on=["season", "team", "player_id"], how="left")
    w["actual_share"] = w["game_share"]
    # must have SOME strictly-prior evidence
    w = w[(w["prior_games"].fillna(0) > 0) | (w["current_games"].fillna(0) > 0)].copy()
    n_backs = w.groupby(["season", "week", "team"])["player_id"].transform("size")
    return w[n_backs >= MIN_BACKS].copy()


def allocative_table(panel: pd.DataFrame, upto_season: int) -> dict:
    """Mean ACTUAL share of the rank-k back among K, ranked by a PREGAME proxy."""
    tr = panel[panel["season"] < upto_season].copy()
    if tr.empty:
        return {}
    tr["proxy"] = tr["current_value"].fillna(tr["prior_value"]).fillna(0.0)
    out: dict = {}
    for (_, _, _), g in tr.groupby(["season", "week", "team"], sort=False):
        K = len(g)
        order = g["proxy"].rank(ascending=False, method="first").astype(int).to_numpy()
        act = g["actual_share"].to_numpy()
        tot = act.sum()
        if tot <= 0:
            continue
        for k, a in zip(order, act / tot):
            out.setdefault((K, int(k)), []).append(float(a))
    return {kk: float(np.mean(vv)) for kk, vv in out.items() if len(vv) >= 20}


def precision_weight(n: float, var_p: float, var_g: float) -> float:
    if not np.isfinite(n) or n <= 0:
        return 0.0
    if not np.isfinite(var_p) or n < 3:
        var_p = var_g          # regularize thin samples to the group (numerical guard)
    if var_g <= 0:
        return float(n)
    return float(n / (1.0 + n * (max(var_p, 0.0) / var_g)))


def posterior(cand: str, row, group_mean: float, alloc_mean: float, var_g: float) -> float:
    pv, pg = row.prior_value, (row.prior_games if np.isfinite(row.prior_games) else 0.0)
    cv, cg = row.current_value, (row.current_games if np.isfinite(row.current_games) else 0.0)
    if cand == "C0_incumbent":            # the REAL production function
        return bv._posterior_mean(group_mean, pv, pg, cv, cg, METRIC)[0]
    if cand == "C4_uniform_less_shrink":  # control: same math, weaker group pull
        saved = bv.GROUP_STRENGTH[METRIC]
        try:
            bv.GROUP_STRENGTH[METRIC] = C4_GROUP_STRENGTH
            return bv._posterior_mean(group_mean, pv, pg, cv, cg, METRIC)[0]
        finally:
            bv.GROUP_STRENGTH[METRIC] = saved
    target = alloc_mean if cand in ("C1_allocative", "C3_both") else group_mean
    ws, vs = [bv.GROUP_STRENGTH[METRIC]], [target]
    if cand in ("C2_precision", "C3_both"):
        if np.isfinite(pv) and pg > 0:
            ws.append(precision_weight(pg, row.prior_var, var_g)); vs.append(pv)
        if np.isfinite(cv) and cg > 0:
            ws.append(precision_weight(cg, row.current_var, var_g)); vs.append(cv)
    else:
        if np.isfinite(pv) and pg > 0:
            ws.append(min(float(pg), bv.PRIOR_PLAYER_CAP[METRIC])); vs.append(pv)
        if np.isfinite(cv) and cg > 0:
            ws.append(float(cg)); vs.append(cv)
    return float(np.average(np.asarray(vs, float), weights=np.asarray(ws, float)))


CANDS = ("C0_incumbent", "C1_allocative", "C2_precision", "C3_both", "C4_uniform_less_shrink")


def main() -> int:
    print("building historical panel 2013-2025 ...")
    panel = build_panel()
    print(f"  rows {len(panel)}  team-weeks {panel.groupby(['season','week','team']).ngroups}")

    recs = []
    for season in sorted(panel["season"].unique()):
        if season <= SEASONS[0]:
            continue
        tr = panel[panel["season"] < season]
        if tr.groupby(["season", "week", "team"]).ngroups < 200:
            continue
        alloc = allocative_table(panel, season)
        group_mean = float(tr["actual_share"].mean())
        var_g = float(tr["actual_share"].var())
        te = panel[panel["season"] == season]
        for key, g in te.groupby(["season", "week", "team"], sort=False):
            K = len(g)
            tot = g["actual_share"].sum()
            if tot <= 0:
                continue
            act = (g["actual_share"] / tot).to_numpy()
            proxy = g["current_value"].fillna(g["prior_value"]).fillna(0.0)
            ranks = proxy.rank(ascending=False, method="first").astype(int).to_numpy()
            ev = [float(precision_weight(
                    r.prior_games if np.isfinite(r.prior_games) else 0.0, r.prior_var, var_g)
                  + precision_weight(
                    r.current_games if np.isfinite(r.current_games) else 0.0, r.current_var, var_g))
                  for r in g.itertuples()]
            per = {}
            for c in CANDS:
                raw = np.array([posterior(c, r, group_mean,
                                          alloc.get((K, int(k)), group_mean), var_g)
                                for r, k in zip(g.itertuples(), ranks)])
                s = raw.sum()
                per[c] = raw / s if s > 0 else np.full(K, 1.0 / K)
            for i, r in enumerate(g.itertuples()):
                row = {"season": int(season), "week": int(r.week), "team": r.team,
                       "K": K, "rank": int(ranks[i]), "actual": float(act[i]),
                       "evidence": ev[i], "raw_sum": float(
                           np.array([posterior(c, r, group_mean,
                                               alloc.get((K, int(ranks[i])), group_mean), var_g)
                                     for c in ("C0_incumbent",)]).sum())}
                for c in CANDS:
                    row[f"err_{c}"] = abs(float(per[c][i]) - float(act[i]))
                    row[f"shr_{c}"] = float(per[c][i])
                recs.append(row)
    d = pd.DataFrame(recs)
    if d.empty:
        print("FAIL CLOSED (K6): no scored rows", file=sys.stderr)
        return 2
    print(f"  scored player-weeks {len(d)}  seasons {d['season'].min()}-{d['season'].max()}")

    print("\n=== PRIMARY 1 — within-team relative share error (lower is better) ===")
    base = d["err_C0_incumbent"]
    res = {}
    print(f"  {'candidate':26s} {'mean err':>9} {'median':>8} {'vs C0':>9} {'95% CI':>22}")
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(d))
    boots = {c: [] for c in CANDS if c != "C0_incumbent"}
    for _ in range(BOOT):
        s = rng.choice(idx, len(idx), replace=True)
        b0 = base.to_numpy()[s].mean()
        for c in boots:
            boots[c].append(b0 - d[f"err_{c}"].to_numpy()[s].mean())
    for c in CANDS:
        e = d[f"err_{c}"]
        if c == "C0_incumbent":
            print(f"  {c:26s} {e.mean():>9.5f} {e.median():>8.5f} {'—':>9} {'—':>22}")
            res[c] = {"mean_err": float(e.mean()), "median_err": float(e.median())}
            continue
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        imp = float(base.mean() - e.mean())
        res[c] = {"mean_err": float(e.mean()), "median_err": float(e.median()),
                  "improvement_vs_C0": round(imp, 6),
                  "ci95": [round(float(lo), 6), round(float(hi), 6)],
                  "ci_excludes_zero": bool(lo > 0)}
        print(f"  {c:26s} {e.mean():>9.5f} {e.median():>8.5f} {imp:>+9.5f}"
              f" [{lo:+.5f}, {hi:+.5f}]")

    print("\n=== PRIMARY 2 — is the gain CONCENTRATED in high-evidence players? ===")
    d["ev_q"] = pd.qcut(d["evidence"], 4, labels=["Q1 thin", "Q2", "Q3", "Q4 strong"],
                        duplicates="drop")
    het = {}
    print(f"  {'stratum':12s} {'n':>6} " + " ".join(f"{c.split('_')[0]:>8}" for c in CANDS))
    for q, g in d.groupby("ev_q", observed=True):
        cells = " ".join(f"{base.loc[g.index].mean()-g[f'err_{c}'].mean():>+8.5f}"
                         if c != "C0_incumbent" else f"{g[f'err_{c}'].mean():>8.5f}"
                         for c in CANDS)
        print(f"  {str(q):12s} {len(g):>6} {cells}")
        het[str(q)] = {c: float(g[f"err_{c}"].mean()) for c in CANDS}
        het[str(q)]["n"] = int(len(g))

    print("\n=== GUARDRAILS ===")
    top = d[d["rank"] == 1]
    print(f"  top-1 within-RB share:  actual {top['actual'].mean():.4f}")
    for c in CANDS:
        print(f"    {c:26s} {top[f'shr_{c}'].mean():.4f}"
              f"  gap {top[f'shr_{c}'].mean()-top['actual'].mean():+.4f}")
    comm = d[(d["K"] >= 3) & (d["rank"] >= 2)]
    print(f"\n  committee backs (K>=3, rank>=2), n={len(comm)}:")
    for c in CANDS:
        print(f"    {c:26s} err {comm[f'err_{c}'].mean():.5f}"
              f"  vs C0 {comm['err_C0_incumbent'].mean()-comm[f'err_{c}'].mean():+.5f}")

    c3, c4 = res["C3_both"], res["C4_uniform_less_shrink"]
    k2 = abs(c3["improvement_vs_C0"] - c4["improvement_vs_C0"]) < 0.1 * max(
        abs(c3["improvement_vs_C0"]), 1e-9)
    gains = [base.loc[g.index].mean() - g["err_C3_both"].mean()
             for _, g in d.groupby("ev_q", observed=True)]
    k3 = len(gains) >= 2 and (max(gains) - min(gains)) < 0.1 * abs(np.mean(gains) or 1e-9)
    print("\n=== GATES / KILLS ===")
    print(f"  G1 C3 beats C0, CI excludes 0 : {c3.get('ci_excludes_zero') and c3['improvement_vs_C0']>0}")
    print(f"  G3 C4 does NOT match C3       : {not k2}   (C3 {c3['improvement_vs_C0']:+.5f}"
          f" vs C4 {c4['improvement_vs_C0']:+.5f})")
    print(f"  K2 TRIGGERED (C4 ~= C3)       : {k2}")
    print(f"  K3 TRIGGERED (gain flat)      : {k3}")

    out = Path("data/research/player_evidence_shrinkage")
    out.mkdir(parents=True, exist_ok=True)
    (out / "player_evidence_shrinkage_v1.json").write_text(json.dumps({
        "version": "PLAYER_EVIDENCE_SHRINKAGE_EVAL_V1",
        "preregistration": "docs/research/PRISM_PLAYER_EVIDENCE_SHRINKAGE_PREREGISTRATION_V1.md",
        "metric": METRIC, "scored_player_weeks": int(len(d)),
        "seasons": [int(d["season"].min()), int(d["season"].max())],
        "primary1": res, "primary2_heterogeneity": het,
        "top1_actual": float(top["actual"].mean()),
        "top1_by_candidate": {c: float(top[f"shr_{c}"].mean()) for c in CANDS},
        "committee_err": {c: float(comm[f"err_{c}"].mean()) for c in CANDS},
        "K2_c4_matches_c3": bool(k2), "K3_gain_flat": bool(k3),
        "sportsbook_inputs_used": False, "promoted": False,
    }, indent=2) + "\n")
    print(f"\nwrote {out}/player_evidence_shrinkage_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
