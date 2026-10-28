"""Walk-forward per-tail quantiles for all three markets.

Executes PRISM_WALK_FORWARD_QUANTILE_PREREGISTRATION_V1, frozen at 42097573 before any
scoring.

Replaces the earlier three-season holdout design, which cost ~13x the available power,
conflated data leakage with researcher degrees of freedom, and drove validation toward
decade-old football. Here the coefficients for target season S are fitted on seasons
[2013, S-1] ONLY, and within a season mu_hat still uses weeks 1..W-1, so every
prediction is out-of-sample in both dimensions and no season needs quarantining.

`_design`, `fit_tail` and `evaluate` are imported from the validated receiving-yards
study rather than reimplemented, so the model form provably cannot drift.

Rule 3 is computed BOTH ways and both are published -- the original strict form and the
joint form that cannot be satisfied by trading one tail for the other. Neither is
suppressed.

No sportsbook variable is read at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.measure_direct_quantile_fit_v1 import (
    ARMS, MIN_HOLDOUT_GAMES, TAILS, Z20, Z80, _design, _load, evaluate, fit_tail, last_week,
)
from scripts.research.measure_player_dispersion_v1 import state_label

OUT_JSON = Path("data/research/walk_forward/walk_forward_quantiles_v1.json")
OUT_PRED = Path("data/research/walk_forward/walk_forward_predictions_{market}.csv")

# ---- frozen walk-forward structure (preregistration section 2) ----
FIRST_FIT_SEASON = 2013              # first year snap counts exist (the anomaly gate needs them)
MIN_PRIOR_SEASONS = 3
TARGET_SEASONS = tuple(range(2016, 2026))    # 2026 excluded: partial and live
FIRST_SCORABLE_WEEK = 4
ERAS = {"2016-2019": (2016, 2019), "2020-2022": (2020, 2022), "2023-2025": (2023, 2025)}

# ---- frozen market definitions (section 3); production sigmas from pricing_v2.py ----
MARKETS = {
    "rec_yards": {"production_sigma": 26.0, "positions": ("WR", "TE"), "outcome": "receiving_yards",
                  "volume": "targets", "min_prior_volume": 20, "min_game_volume": 1, "min_prior_games": 3},
    "rush_yards": {"production_sigma": 24.0, "positions": ("RB",), "outcome": "rushing_yards",
                   "volume": "carries", "min_prior_volume": 60, "min_game_volume": 5, "min_prior_games": 3},
    "receptions": {"production_sigma": 1.8, "positions": ("WR", "TE"), "outcome": "receptions",
                   "volume": "targets", "min_prior_volume": 20, "min_game_volume": 1, "min_prior_games": 3},
}


def build_panel(season: int, market: str) -> tuple[pd.DataFrame, dict]:
    spec = MARKETS[market]
    w = _load("weekly", season)
    w = w[w["week"].between(1, last_week(season))].copy()
    pcol = "position" if "position" in w.columns else "position_group"
    w["pos"] = w[pcol].astype(str).str.upper()
    w = w[w["pos"].isin(spec["positions"])].copy()
    nm = "player_display_name" if "player_display_name" in w.columns else "player_name"
    w["player"] = w[nm].astype(str)
    w["pid"] = w["player_id"].astype(str)
    for c in (spec["outcome"], spec["volume"], "targets", "carries"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)

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
                "mu_hat": (float(prior[out_col].sum()) / pv) * (pv / n_prior),
                "centrality": float(pd.to_numeric(prior["share"], errors="coerce").mean()),
                "y": float(cur[out_col]),
            })
    panel = pd.DataFrame(rows).dropna(subset=["mu_hat", "centrality"])
    return panel, {"season": season, "market": market,
                   "state_counts": w["state"].value_counts().to_dict(),
                   "panel_rows": int(len(panel)),
                   "panel_players": int(panel["pid"].nunique()) if len(panel) else 0}


def walk_forward(panels: dict, market: str) -> tuple[pd.DataFrame, dict]:
    """For each target season, fit on strictly-earlier seasons only."""
    sigma = MARKETS[market]["production_sigma"]
    preds, coefs_by_season = [], {}
    for target in TARGET_SEASONS:
        fit_years = [y for y in range(FIRST_FIT_SEASON, target) if y in panels and len(panels[y])]
        if len(fit_years) < MIN_PRIOR_SEASONS or target not in panels or not len(panels[target]):
            continue
        fit = pd.concat([panels[y] for y in fit_years], ignore_index=True)
        coefs = {arm: {name: fit_tail(fit, arm, q) for name, q in TAILS} for arm in ("Q1", "Q2")}
        coefs_by_season[target] = {a: {k: [float(x) for x in v] for k, v in c.items()}
                                   for a, c in coefs.items()}
        tgt = panels[target].copy()
        mu = tgt["mu_hat"].to_numpy(float)
        tgt["p20_Q0"], tgt["p80_Q0"] = mu + Z20 * sigma, mu + Z80 * sigma
        for arm in ("Q1", "Q2"):
            X = _design(tgt, arm)
            tgt[f"p20_{arm}"] = X @ coefs[arm]["p20"]
            tgt[f"p80_{arm}"] = X @ coefs[arm]["p80"]
        tgt["fit_seasons"] = f"{min(fit_years)}-{max(fit_years)}"
        preds.append(tgt)
    if not preds:
        # fail closed and say why, rather than an opaque concat error
        raise RuntimeError(
            f"no target season qualified for {market}: every candidate in "
            f"{TARGET_SEASONS[0]}-{TARGET_SEASONS[-1]} lacked {MIN_PRIOR_SEASONS} prior "
            f"seasons of panel data from {FIRST_FIT_SEASON} onward, or had an empty panel"
        )
    return pd.concat(preds, ignore_index=True), coefs_by_season


def _score(pred: pd.DataFrame, min_games: int = MIN_HOLDOUT_GAMES):
    """Re-use the validated evaluator by handing it the already-computed tails."""
    packed = {arm: {"p20": pred[f"p20_{arm}"].to_numpy(float),
                    "p80": pred[f"p80_{arm}"].to_numpy(float)} for arm in ARMS}
    return evaluate(pred, packed, min_games)


def decide_dual(ev: dict, production_sigma: float) -> dict:
    """Rule 3 both ways, per the preregistration. Neither verdict is suppressed."""
    tol = 0.25 * production_sigma
    res = {"rule_3_joint_tolerance": tol}
    for arm in ("Q1", "Q2"):
        a, q0 = ev[arm], ev["Q0"]
        r1 = a["frac_well_calibrated"] > q0["frac_well_calibrated"]
        r2 = a["frac_players_pin20_improved"] > 0.5 and a["frac_players_pin80_improved"] > 0.5
        hi_a, hi_0 = ev["bias_by_mean_tercile"][arm]["high"], ev["bias_by_mean_tercile"]["Q0"]["high"]
        fa, ca = abs(hi_a["floor_bias"]), abs(hi_a["ceiling_bias"])
        f0, c0 = abs(hi_0["floor_bias"]), abs(hi_0["ceiling_bias"])
        r3_strict = (fa < f0) and (ca <= c0 + 1e-9)
        r3_joint = ((fa + ca) < (f0 + c0)) and (fa - f0 <= tol) and (ca - c0 <= tol)
        r4 = a["frac_rows_crossed"] == 0.0
        res[arm] = {
            "rule_1_more_calibrated_players": bool(r1),
            "rule_2_pinball_majority_BOTH_tails": bool(r2),
            "rule_3_strict": bool(r3_strict),
            "rule_3_joint": bool(r3_joint),
            "rule_4_no_crossed_quantiles": bool(r4),
            "high_tercile_abs_bias_sum": {"Q0": f0 + c0, "arm": fa + ca},
            "SUPPORTED_STRICT": bool(r1 and r2 and r3_strict and r4),
            "SUPPORTED_JOINT": bool(r1 and r2 and r3_joint and r4),
        }
    for key in ("STRICT", "JOINT"):
        res[f"H1_{key}"] = bool(res["Q1"][f"SUPPORTED_{key}"] or res["Q2"][f"SUPPORTED_{key}"])
        res[f"preferred_{key}"] = ("Q1" if res["Q1"][f"SUPPORTED_{key}"]
                                   else ("Q2" if res["Q2"][f"SUPPORTED_{key}"] else None))
    return res


def main() -> int:
    prereg = Path("docs/research/PRISM_WALK_FORWARD_QUANTILE_PREREGISTRATION_V1.md")
    if not prereg.exists():
        raise RuntimeError("preregistration missing; this study may not be scored unfrozen")
    text = prereg.read_text(encoding="utf-8")
    for token in ("2016–2025", "0.25 × that market's production σ", "2026 is EXCLUDED"):
        if token not in text:
            raise RuntimeError(f"preregistration no longer carries frozen value {token!r}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)   # before any per-market CSV write
    report = {"preregistration": prereg.as_posix(),
              "walk_forward": {"first_fit_season": FIRST_FIT_SEASON,
                               "min_prior_seasons": MIN_PRIOR_SEASONS,
                               "target_seasons": list(TARGET_SEASONS)},
              "markets": {}, "sportsbook_inputs_used": False}

    for market, spec in MARKETS.items():
        panels, diags = {}, []
        for yr in range(FIRST_FIT_SEASON, max(TARGET_SEASONS) + 1):
            p, d = build_panel(yr, market)
            panels[yr] = p
            diags.append(d)
        pred, coefs_by_season = walk_forward(panels, market)
        pred.to_csv(str(OUT_PRED).format(market=market), index=False)
        ev, _ = _score(pred)
        decision = decide_dual(ev, spec["production_sigma"])

        eras = {}
        for name, (lo, hi) in ERAS.items():
            sub = pred[pred["season"].between(lo, hi)]
            if sub["pid"].nunique() < 20:
                continue
            e, _ = _score(sub)
            eras[name] = {
                "seasons": [lo, hi], "rows": int(len(sub)), "players": int(e["players_scored"]),
                **{a: {"frac_well_calibrated": e[a]["frac_well_calibrated"],
                       "coverage_median": e[a]["coverage_median"]} for a in ARMS},
                "Q1_p20_improved": e["Q1"]["frac_players_pin20_improved"],
                "Q1_p80_improved": e["Q1"]["frac_players_pin80_improved"],
                "Q2_p20_improved": e["Q2"]["frac_players_pin20_improved"],
                "Q2_p80_improved": e["Q2"]["frac_players_pin80_improved"],
                "decision": decide_dual(e, spec["production_sigma"]),
            }
        report["markets"][market] = {
            "production_sigma": spec["production_sigma"],
            "walk_forward_rows": int(len(pred)),
            "walk_forward_players": int(pred["pid"].nunique()),
            "seasons_scored": sorted(int(s) for s in pred["season"].unique()),
            "coefficients_by_target_season": coefs_by_season,
            "diagnostics": diags, "pooled": ev, "decision": decision, "by_era": eras,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")

    for market, r in report["markets"].items():
        ev, dec = r["pooled"], r["decision"]
        print(f"\n{'='*74}\n{market.upper()}  (production sigma={r['production_sigma']})")
        print(f"  walk-forward: {r['walk_forward_rows']} rows, {r['walk_forward_players']} players, "
              f"seasons {min(r['seasons_scored'])}-{max(r['seasons_scored'])}, "
              f"{ev['players_scored']} players scored")
        for arm in ARMS:
            a = ev[arm]
            print(f"  {arm}: well-calibrated {a['frac_well_calibrated']:.1%}  "
                  f"coverage median {a['coverage_median']:.3f}  crossed {a['frac_rows_crossed']:.1%}")
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
                  f"r3strict={d['rule_3_strict']} r3joint={d['rule_3_joint']} "
                  f"r4={d['rule_4_no_crossed_quantiles']} -> "
                  f"strict={'YES' if d['SUPPORTED_STRICT'] else 'no'} "
                  f"joint={'YES' if d['SUPPORTED_JOINT'] else 'no'}")
        print(f"  >>> {market}: H1 STRICT={dec['H1_STRICT']} ({dec['preferred_STRICT']})  "
              f"H1 JOINT={dec['H1_JOINT']} ({dec['preferred_JOINT']})")
        print("  BY ERA (era stability is required, not optional):")
        for name, e in r["by_era"].items():
            print(f"    {name}: n={e['players']:4d} | wc Q0 {e['Q0']['frac_well_calibrated']:.0%} "
                  f"Q1 {e['Q1']['frac_well_calibrated']:.0%} Q2 {e['Q2']['frac_well_calibrated']:.0%} "
                  f"| Q2 p20 {e['Q2_p20_improved']:.0%} p80 {e['Q2_p80_improved']:.0%} "
                  f"| joint Q1={e['decision']['Q1']['SUPPORTED_JOINT']} Q2={e['decision']['Q2']['SUPPORTED_JOINT']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
