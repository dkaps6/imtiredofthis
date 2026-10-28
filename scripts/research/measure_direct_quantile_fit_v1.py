"""Direct per-tail quantile fit vs production's constant-sigma normal.

Executes PRISM_DIRECT_QUANTILE_PREREGISTRATION_V1, frozen at abe11db5 before any
scoring. Arms, estimator, windows, thresholds and the four-part decision rule are
quoted from that document and none may be changed to produce a winner.

The prior study (PRISM_PLAYER_DISPERSION_RESULTS_V1) proved production's intervals
contain 50.0% of outcomes against a 60% target, and that scaling a single sigma by the
player's projected mean raises well-calibrated players 41.7% -> 52.1%. It failed
because a symmetric normal centred on the mean forces p20 and p80 equidistant, so
widening fixed the floor for 77.1% of players while breaking the ceiling for 68.8%.

This study removes the distribution family. Each tail is fitted directly by its own
pinball loss, so the asymmetry between floor and ceiling is free.

Evaluation is PER PLAYER. Averaged MAE is disallowed as primary: it is insensitive to
dispersion by construction, which is how the constant survived unexamined.

No sportsbook variable is read at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.research.measure_player_dispersion_v1 import (
    CALIBRATED_TOLERANCE, MIN_PRIOR_GAMES_WITH_TARGET, MIN_PRIOR_TARGETS,
    PRODUCTION_SIGMA, TARGET_COVERAGE, Z20, Z80, _load, _pinball, state_label,
)

OUT_JSON = Path("data/research/direct_quantile/direct_quantile_fit_v1.json")
OUT_ROWS = Path("data/research/direct_quantile/direct_quantile_fit_v1_players.csv")

# ---- frozen windows (preregistration section 2) ----
FIT_SEASONS = (2020, 2021, 2022)
HOLDOUT_SEASONS = (2017, 2018, 2019)       # never inspected; the only decisive evaluation
REPORTED_SEASONS = (2023, 2024, 2025)      # contaminated; cannot change the verdict
FIRST_SCORABLE_WEEK = 4
MIN_HOLDOUT_GAMES = 8                      # frozen this time, per section 3
ARMS = ("Q0", "Q1", "Q2")
TAILS = (("p20", 0.20), ("p80", 0.80))


def last_week(season: int) -> int:
    """Regular season ran to week 17 before 2021 and week 18 from 2021."""
    return 17 if season < 2021 else 18


def build_panel(season: int) -> tuple[pd.DataFrame, dict]:
    w = _load("weekly", season)
    w = w[w["week"].between(1, last_week(season))].copy()
    pos = "position" if "position" in w.columns else "position_group"
    w["pos"] = w[pos].astype(str).str.upper()
    w = w[w["pos"].isin(["WR", "TE"])].copy()
    name = "player_display_name" if "player_display_name" in w.columns else "player_name"
    w["player"] = w[name].astype(str)
    w["pid"] = w["player_id"].astype(str)
    for c in ("targets", "receiving_yards", "receptions", "carries"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)

    tt = w.groupby(["team", "week"])["targets"].sum().rename("team_targets").reset_index()
    w = w.merge(tt, on=["team", "week"], how="left")
    w["tgt_share"] = w["targets"] / w["team_targets"].replace(0, np.nan)

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
    rows = []
    for pid, grp in usable.groupby("pid"):
        grp = grp.sort_values("week")
        for _, cur in grp.iterrows():
            week = int(cur["week"])
            if week < FIRST_SCORABLE_WEEK or cur["state"] == "PARTIAL_GAME" or cur["targets"] < 1:
                continue
            prior = grp[grp["week"] < week]
            pt = float(prior["targets"].sum())
            if pt < MIN_PRIOR_TARGETS or int((prior["targets"] > 0).sum()) < MIN_PRIOR_GAMES_WITH_TARGET:
                continue
            n_prior = int(len(prior))
            rows.append({
                "season": season, "pid": pid, "player": cur["player"], "pos": cur["pos"], "week": week,
                "mu_hat": (float(prior["receiving_yards"].sum()) / pt) * (pt / n_prior),
                "centrality": float(pd.to_numeric(prior["tgt_share"], errors="coerce").mean()),
                "y": float(cur["receiving_yards"]),
            })
    panel = pd.DataFrame(rows).dropna(subset=["mu_hat", "centrality"])
    return panel, {"season": season, "last_week": last_week(season),
                   "state_counts": w["state"].value_counts().to_dict(),
                   "panel_rows": int(len(panel)),
                   "panel_players": int(panel["pid"].nunique()) if len(panel) else 0}


def _design(df: pd.DataFrame, arm: str) -> np.ndarray:
    cols = [np.ones(len(df)), df["mu_hat"].to_numpy(float)]
    if arm == "Q2":
        cols.append(df["centrality"].to_numpy(float))
    return np.column_stack(cols)


def fit_tail(fit: pd.DataFrame, arm: str, q: float) -> np.ndarray:
    """Minimise this tail's OWN pinball loss, so the two tails are independent."""
    X, y = _design(fit, arm), fit["y"].to_numpy(float)

    def loss(b: np.ndarray) -> float:
        return float(_pinball(y, X @ b, q).mean())

    # least-squares warm start, then Nelder-Mead: the pinball loss is piecewise
    # linear, so a gradient method is inappropriate at these dimensions
    start, *_ = np.linalg.lstsq(X, y, rcond=None)
    start = start.copy()
    start[0] += (q - 0.5) * 2.0 * PRODUCTION_SIGMA
    best = minimize(loss, start, method="Nelder-Mead",
                    options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    return best.x


def predict(df: pd.DataFrame, arm: str, coefs: dict) -> dict:
    if arm == "Q0":
        mu = df["mu_hat"].to_numpy(float)
        return {"p20": mu + Z20 * PRODUCTION_SIGMA, "p80": mu + Z80 * PRODUCTION_SIGMA}
    X = _design(df, arm)
    return {name: X @ coefs[arm][name] for name, _ in TAILS}


def evaluate(df: pd.DataFrame, preds: dict, min_games: int = MIN_HOLDOUT_GAMES):
    d = df.copy()
    for arm in ARMS:
        p20, p80 = preds[arm]["p20"], preds[arm]["p80"]
        d[f"p20_{arm}"], d[f"p80_{arm}"] = p20, p80
        d[f"in60_{arm}"] = ((d["y"] >= p20) & (d["y"] <= p80)).astype(float)
        d[f"pin20_{arm}"] = _pinball(d["y"].to_numpy(float), p20, 0.20)
        d[f"pin80_{arm}"] = _pinball(d["y"].to_numpy(float), p80, 0.80)
        d[f"crossed_{arm}"] = (p20 >= p80).astype(float)

    agg = {"games": ("y", "size"), "mu": ("mu_hat", "mean"), "player": ("player", "first")}
    for arm in ARMS:
        agg[f"cov_{arm}"] = (f"in60_{arm}", "mean")
        agg[f"pin20_{arm}"] = (f"pin20_{arm}", "mean")
        agg[f"pin80_{arm}"] = (f"pin80_{arm}", "mean")
        agg[f"pred20_{arm}"] = (f"p20_{arm}", "mean")
        agg[f"pred80_{arm}"] = (f"p80_{arm}", "mean")
    per = d.groupby("pid").agg(**agg)
    per = per[per["games"] >= min_games]

    out: dict = {"players_scored": int(len(per)), "rows": int(len(d)), "min_holdout_games": min_games}
    for arm in ARMS:
        err = (per[f"cov_{arm}"] - TARGET_COVERAGE).abs()
        out[arm] = {
            "frac_well_calibrated": float((err <= CALIBRATED_TOLERANCE).mean()),
            "coverage_median": float(per[f"cov_{arm}"].median()),
            "coverage_iqr": [float(per[f"cov_{arm}"].quantile(.25)), float(per[f"cov_{arm}"].quantile(.75))],
            "coverage_deciles": [float(per[f"cov_{arm}"].quantile(q / 10)) for q in range(1, 10)],
            "interval_width_median": float((per[f"pred80_{arm}"] - per[f"pred20_{arm}"]).median()),
            "frac_rows_crossed": float(d[f"crossed_{arm}"].mean()),
        }
    for arm in ("Q1", "Q2"):
        out[arm]["frac_players_pin20_improved"] = float((per[f"pin20_{arm}"] < per["pin20_Q0"]).mean())
        out[arm]["frac_players_pin80_improved"] = float((per[f"pin80_{arm}"] < per["pin80_Q0"]).mean())

    per["tercile"] = pd.qcut(per["mu"], 3, labels=["low", "mid", "high"])
    emp = d.groupby("pid")["y"].agg(e20=lambda s: s.quantile(.20), e80=lambda s: s.quantile(.80))
    per = per.join(emp)
    bias = {}
    for arm in ARMS:
        t = pd.DataFrame({"tercile": per["tercile"],
                          "fb": per[f"pred20_{arm}"] - per["e20"],
                          "cb": per[f"pred80_{arm}"] - per["e80"]})
        g = t.groupby("tercile", observed=True).mean()
        bias[arm] = {str(k): {"floor_bias": float(v["fb"]), "ceiling_bias": float(v["cb"])}
                     for k, v in g.iterrows()}
    out["bias_by_mean_tercile"] = bias
    return out, per


def decide(ev: dict) -> dict:
    """Frozen four-part rule; all must hold on the naive holdout."""
    res = {}
    for arm in ("Q1", "Q2"):
        a, q0 = ev[arm], ev["Q0"]
        r1 = a["frac_well_calibrated"] > q0["frac_well_calibrated"]
        r2 = a["frac_players_pin20_improved"] > 0.5 and a["frac_players_pin80_improved"] > 0.5
        hi_a = ev["bias_by_mean_tercile"][arm]["high"]
        hi_0 = ev["bias_by_mean_tercile"]["Q0"]["high"]
        r3 = (abs(hi_a["floor_bias"]) < abs(hi_0["floor_bias"])
              and abs(hi_a["ceiling_bias"]) <= abs(hi_0["ceiling_bias"]) + 1e-9)
        r4 = a["frac_rows_crossed"] == 0.0
        res[arm] = {"rule_1_more_calibrated_players": bool(r1),
                    "rule_2_pinball_majority_BOTH_tails": bool(r2),
                    "rule_3_high_mean_floor_better_ceiling_not_worse": bool(r3),
                    "rule_4_no_crossed_quantiles": bool(r4),
                    "SUPPORTED": bool(r1 and r2 and r3 and r4)}
    res["H1_SUPPORTED"] = bool(res["Q1"]["SUPPORTED"] or res["Q2"]["SUPPORTED"])
    res["preferred_arm"] = ("Q1" if res["Q1"]["SUPPORTED"] else
                            ("Q2" if res["Q2"]["SUPPORTED"] else None))
    return res


def main() -> int:
    prereg = Path("docs/research/PRISM_DIRECT_QUANTILE_PREREGISTRATION_V1.md")
    if not prereg.exists():
        raise RuntimeError("preregistration missing; this study may not be scored unfrozen")
    text = prereg.read_text(encoding="utf-8")
    for token in ("2017, 2018, 2019", "±0.10", "26.0", "p20_hat < p80_hat"):
        if token not in text:
            raise RuntimeError(f"preregistration no longer carries frozen value {token!r}")

    panels, diags = {}, []
    for yr in FIT_SEASONS + HOLDOUT_SEASONS + REPORTED_SEASONS:
        p, d = build_panel(yr)
        panels[yr] = p
        diags.append(d)

    fit = pd.concat([panels[y] for y in FIT_SEASONS], ignore_index=True)
    coefs = {arm: {name: fit_tail(fit, arm, q) for name, q in TAILS} for arm in ("Q1", "Q2")}

    def score(df: pd.DataFrame, min_games: int = MIN_HOLDOUT_GAMES):
        return evaluate(df, {arm: predict(df, arm, coefs) for arm in ARMS}, min_games)

    hold = pd.concat([panels[y] for y in HOLDOUT_SEASONS], ignore_index=True)
    ev, per = score(hold)
    decision = decide(ev)

    report = {
        "preregistration": prereg.as_posix(),
        "fit_seasons": list(FIT_SEASONS), "holdout_seasons": list(HOLDOUT_SEASONS),
        "fit_rows": int(len(fit)),
        "fitted_coefficients": {a: {k: [float(x) for x in v] for k, v in c.items()}
                                for a, c in coefs.items()},
        "diagnostics": diags,
        "holdout_primary": ev,
        "decision": decision,
        "sensitivity_min_holdout_games": {
            str(g): {a: score(hold, g)[0][a]["frac_well_calibrated"] for a in ARMS} for g in (6, 10)
        },
        "reported_not_decisive": {str(y): score(panels[y])[0] for y in REPORTED_SEASONS if len(panels[y])},
        "sportsbook_inputs_used": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    per.reset_index().to_csv(OUT_ROWS, index=False)

    print(f"fit rows={len(fit)} {FIT_SEASONS}  |  holdout {HOLDOUT_SEASONS}: "
          f"{ev['rows']} rows, {ev['players_scored']} players\n")
    print("fitted tails (each by its OWN pinball loss, so asymmetry is free):")
    for arm in ("Q1", "Q2"):
        for name, _ in TAILS:
            c = coefs[arm][name]
            extra = f" + {c[2]:+.2f}*centrality" if len(c) > 2 else ""
            print(f"  {arm} {name}: {c[0]:+8.3f} + {c[1]:.4f}*mu{extra}")
    print(f"\nPER-PLAYER CALIBRATION, target {TARGET_COVERAGE}:")
    for arm in ARMS:
        a = ev[arm]
        print(f"  {arm}: well-calibrated {a['frac_well_calibrated']:.1%}  "
              f"coverage median {a['coverage_median']:.3f} IQR "
              f"{a['coverage_iqr'][0]:.2f}-{a['coverage_iqr'][1]:.2f}  "
              f"median width {a['interval_width_median']:.1f}  crossed {a['frac_rows_crossed']:.1%}")
    for arm in ("Q1", "Q2"):
        print(f"  {arm}: players improved -- p20 {ev[arm]['frac_players_pin20_improved']:.1%}, "
              f"p80 {ev[arm]['frac_players_pin80_improved']:.1%}")
    print("\nfloor/ceiling bias by mean tercile:")
    for arm in ARMS:
        b = ev["bias_by_mean_tercile"][arm]
        print(f"  {arm}: " + "  ".join(
            f"{k} fb={b[k]['floor_bias']:+6.1f} cb={b[k]['ceiling_bias']:+6.1f}"
            for k in ("low", "mid", "high")))
    print("\nFROZEN DECISION RULE:")
    for arm in ("Q1", "Q2"):
        d = decision[arm]
        print(f"  {arm}: r1={d['rule_1_more_calibrated_players']} "
              f"r2={d['rule_2_pinball_majority_BOTH_tails']} "
              f"r3={d['rule_3_high_mean_floor_better_ceiling_not_worse']} "
              f"r4={d['rule_4_no_crossed_quantiles']} "
              f"-> {'SUPPORTED' if d['SUPPORTED'] else 'not supported'}")
    print(f"\n  H1 SUPPORTED: {decision['H1_SUPPORTED']}   preferred arm: {decision['preferred_arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
