"""Per-tail quantile replication for RB rushing yards and WR/TE receptions.

Executes PRISM_DIRECT_QUANTILE_RUSH_REC_PREREGISTRATION_V1, frozen at 50b4c4f0 before
any scoring. This is a REPLICATION: `_design`, `fit_tail`, `evaluate` and `decide` are
imported from the validated receiving-yards study rather than reimplemented, so the
model form, estimator, evaluation and decision rule provably cannot drift per market.
Only the market definition (production sigma, position filter, volume/outcome columns,
eligibility) varies, and each market is judged separately.

Production carries an undocumented constant per market in pricing_v2.py:36-43 --
rush_yards 24.0, receptions 1.8 -- each with a normal CDF, so both inherit the defect
the receiving-yards study established: both tails forced to move 1:1 with the mean and
role made irrelevant.

Receptions is a small integer count. Per the preregistration this is disclosed in
advance: coverage of an integer outcome is intrinsically lumpy and exactly 0.60 may be
unattainable, which can suppress rule 1 for reasons unrelated to model quality. The
rules are NOT relaxed for it.

No sportsbook variable is read at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.measure_direct_quantile_fit_v1 import (
    ARMS, MIN_HOLDOUT_GAMES, TAILS, Z20, Z80, _design, _load, decide, evaluate,
    fit_tail, last_week,
)
from scripts.research.measure_player_dispersion_v1 import state_label

OUT_JSON = Path("data/research/direct_quantile/direct_quantile_rush_rec_v1.json")
OUT_ROWS = Path("data/research/direct_quantile/direct_quantile_rush_rec_v1_{market}_players.csv")

# ---- frozen windows (preregistration section 2) ----
FIT_SEASONS = (2020, 2021, 2022)
HOLDOUT_SEASONS = (2014, 2015, 2016)          # never inspected for ANY market
REPORTED_SEASONS = (2017, 2018, 2019, 2023, 2024, 2025)
FIRST_SCORABLE_WEEK = 4

# ---- frozen market definitions (section 3); production sigmas from pricing_v2.py ----
MARKETS = {
    "rush_yards": {
        "production_sigma": 24.0,
        "positions": ("RB",),
        "outcome": "rushing_yards",
        "volume": "carries",
        "min_prior_volume": 60,
        "min_prior_games": 3,
        "min_game_volume": 5,          # >=5 carries in week W, and for a qualifying prior game
        "discrete": False,
    },
    "receptions": {
        "production_sigma": 1.8,
        "positions": ("WR", "TE"),
        "outcome": "receptions",
        "volume": "targets",
        "min_prior_volume": 20,
        "min_prior_games": 3,
        "min_game_volume": 1,
        "discrete": True,
    },
}


def build_panel(season: int, market: str) -> tuple[pd.DataFrame, dict]:
    spec = MARKETS[market]
    w = _load("weekly", season)
    w = w[w["week"].between(1, last_week(season))].copy()
    pcol = "position" if "position" in w.columns else "position_group"
    w["pos"] = w[pcol].astype(str).str.upper()
    w = w[w["pos"].isin(spec["positions"])].copy()
    name = "player_display_name" if "player_display_name" in w.columns else "player_name"
    w["player"] = w[name].astype(str)
    w["pid"] = w["player_id"].astype(str)
    for c in (spec["outcome"], spec["volume"], "targets", "carries"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)

    # role centrality: his share of the team's volume for THIS market
    tv = w.groupby(["team", "week"])[spec["volume"]].sum().rename("team_volume").reset_index()
    w = w.merge(tv, on=["team", "week"], how="left")
    w["share"] = w[spec["volume"]] / w["team_volume"].replace(0, np.nan)

    try:
        s = _load("snaps", season)[["player", "week", "offense_snaps"]].copy()
        s["week"] = pd.to_numeric(s["week"], errors="coerce")
        s = s.groupby(["player", "week"], as_index=False)["offense_snaps"].max()
        w = w.merge(s, on=["player", "week"], how="left")
    except Exception:
        w["offense_snaps"] = np.nan
    if "offense_snaps" not in w.columns:
        w["offense_snaps"] = np.nan

    norm = w.groupby("pid")["offense_snaps"].median().to_dict()
    w["state"] = [state_label(t, c, sn, norm.get(p))
                  for t, c, sn, p in zip(w["targets"], w["carries"], w["offense_snaps"], w["pid"])]

    usable = w[~w["state"].isin(("DNP_INACTIVE", "UNRESOLVED_CONTEXT"))]
    vol, out_col, mgv = spec["volume"], spec["outcome"], spec["min_game_volume"]
    rows = []
    for pid, grp in usable.groupby("pid"):
        grp = grp.sort_values("week")
        for _, cur in grp.iterrows():
            week = int(cur["week"])
            if week < FIRST_SCORABLE_WEEK or cur["state"] == "PARTIAL_GAME" or cur[vol] < mgv:
                continue
            prior = grp[grp["week"] < week]
            pv = float(prior[vol].sum())
            if pv < spec["min_prior_volume"] or int((prior[vol] >= mgv).sum()) < spec["min_prior_games"]:
                continue
            n_prior = int(len(prior))
            rows.append({
                "season": season, "pid": pid, "player": cur["player"], "pos": cur["pos"], "week": week,
                # efficiency per unit volume x volume per game -- identical construction to the
                # validated study, with this market's own volume and outcome
                "mu_hat": (float(prior[out_col].sum()) / pv) * (pv / n_prior),
                "centrality": float(pd.to_numeric(prior["share"], errors="coerce").mean()),
                "y": float(cur[out_col]),
            })
    panel = pd.DataFrame(rows).dropna(subset=["mu_hat", "centrality"])
    return panel, {"season": season, "market": market, "last_week": last_week(season),
                   "state_counts": w["state"].value_counts().to_dict(),
                   "panel_rows": int(len(panel)),
                   "panel_players": int(panel["pid"].nunique()) if len(panel) else 0}


def predict(df: pd.DataFrame, arm: str, coefs: dict, production_sigma: float) -> dict:
    if arm == "Q0":
        mu = df["mu_hat"].to_numpy(float)
        return {"p20": mu + Z20 * production_sigma, "p80": mu + Z80 * production_sigma}
    X = _design(df, arm)
    return {name: X @ coefs[arm][name] for name, _ in TAILS}


def run_market(market: str) -> dict:
    spec = MARKETS[market]
    panels, diags = {}, []
    for yr in FIT_SEASONS + HOLDOUT_SEASONS + REPORTED_SEASONS:
        p, d = build_panel(yr, market)
        panels[yr] = p
        diags.append(d)

    fit = pd.concat([panels[y] for y in FIT_SEASONS], ignore_index=True)
    coefs = {arm: {name: fit_tail(fit, arm, q) for name, q in TAILS} for arm in ("Q1", "Q2")}

    def score(df: pd.DataFrame, min_games: int = MIN_HOLDOUT_GAMES):
        preds = {arm: predict(df, arm, coefs, spec["production_sigma"]) for arm in ARMS}
        return evaluate(df, preds, min_games)

    hold = pd.concat([panels[y] for y in HOLDOUT_SEASONS], ignore_index=True)
    ev, per = score(hold)
    decision = decide(ev)
    per.reset_index().to_csv(str(OUT_ROWS).format(market=market), index=False)

    return {
        "market": market,
        "production_sigma": spec["production_sigma"],
        "discrete_outcome": spec["discrete"],
        "fit_rows": int(len(fit)),
        "fitted_coefficients": {a: {k: [float(x) for x in v] for k, v in c.items()}
                                for a, c in coefs.items()},
        "diagnostics": diags,
        "holdout_primary": ev,
        "decision": decision,
        "sensitivity_min_holdout_games": {
            str(g): {a: score(hold, g)[0][a]["frac_well_calibrated"] for a in ARMS} for g in (6, 10)
        },
        "reported_not_decisive": {str(y): score(panels[y])[0] for y in REPORTED_SEASONS
                                  if len(panels[y]) and panels[y]["pid"].nunique() >= 10},
    }


def main() -> int:
    prereg = Path("docs/research/PRISM_DIRECT_QUANTILE_RUSH_REC_PREREGISTRATION_V1.md")
    if not prereg.exists():
        raise RuntimeError("preregistration missing; this study may not be scored unfrozen")
    text = prereg.read_text(encoding="utf-8")
    for token in ("2014, 2015, 2016", "24.0", "1.8", "are **not** relaxed"):
        if token not in text:
            raise RuntimeError(f"preregistration no longer carries frozen value {token!r}")

    report = {"preregistration": prereg.as_posix(),
              "fit_seasons": list(FIT_SEASONS), "holdout_seasons": list(HOLDOUT_SEASONS),
              "markets": {m: run_market(m) for m in MARKETS},
              "sportsbook_inputs_used": False}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON}")
    for m, r in report["markets"].items():
        ev, dec = r["holdout_primary"], r["decision"]
        print(f"\n{'='*72}\n{m.upper()}  (production sigma={r['production_sigma']}"
              f"{', DISCRETE outcome' if r['discrete_outcome'] else ''})")
        print(f"  fit {r['fit_rows']} rows | holdout {HOLDOUT_SEASONS}: {ev['rows']} rows, "
              f"{ev['players_scored']} players")
        for arm in ("Q1", "Q2"):
            for name, _ in TAILS:
                c = r["fitted_coefficients"][arm][name]
                extra = f" + {c[2]:+.3f}*centrality" if len(c) > 2 else ""
                print(f"    {arm} {name}: {c[0]:+8.3f} + {c[1]:.4f}*mu{extra}")
        for arm in ARMS:
            a = ev[arm]
            print(f"  {arm}: well-calibrated {a['frac_well_calibrated']:.1%}  "
                  f"coverage median {a['coverage_median']:.3f}  "
                  f"median width {a['interval_width_median']:.2f}  crossed {a['frac_rows_crossed']:.1%}")
        for arm in ("Q1", "Q2"):
            print(f"  {arm}: improved -- p20 {ev[arm]['frac_players_pin20_improved']:.1%}, "
                  f"p80 {ev[arm]['frac_players_pin80_improved']:.1%}")
        for arm in ARMS:
            b = ev["bias_by_mean_tercile"][arm]
            print(f"  {arm} bias: " + "  ".join(
                f"{k} fb={b[k]['floor_bias']:+6.2f} cb={b[k]['ceiling_bias']:+6.2f}"
                for k in ("low", "mid", "high")))
        for arm in ("Q1", "Q2"):
            d = dec[arm]
            print(f"  RULE {arm}: r1={d['rule_1_more_calibrated_players']} "
                  f"r2={d['rule_2_pinball_majority_BOTH_tails']} "
                  f"r3={d['rule_3_high_mean_floor_better_ceiling_not_worse']} "
                  f"r4={d['rule_4_no_crossed_quantiles']} -> "
                  f"{'SUPPORTED' if d['SUPPORTED'] else 'not supported'}")
        print(f"  >>> {m}: H1 SUPPORTED = {dec['H1_SUPPORTED']}   preferred arm: {dec['preferred_arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
