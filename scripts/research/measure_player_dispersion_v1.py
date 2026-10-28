"""Player-specific dispersion vs the per-market constant.

Executes PRISM_PLAYER_DISPERSION_PREREGISTRATION_V1, frozen at cd8117d0 before any
scoring. Every threshold, arm, estimator, window and decision rule below is quoted
from that document; none may be changed to produce a winner.

Production prices every receiver with `sigma = SIGMA["rec_yards"] = 26.0` and a normal
CDF (pricing_v2.py:244), so no player property enters dispersion at all and each
player's floor and ceiling are forced equidistant from his mean.

H1: dispersion as a function of the player's own projected mean and role centrality
calibrates individual floors and ceilings better than that constant.

Evaluation is PER PLAYER. Averaged MAE is disallowed as a primary metric because it is
mathematically insensitive to dispersion -- a model can have every mean right and every
floor and ceiling wrong and score identically.

No sportsbook variable is read at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_JSON = Path("data/research/player_dispersion/player_dispersion_v1.json")
OUT_ROWS = Path("data/research/player_dispersion/player_dispersion_v1_players.csv")
CACHE = Path("/tmp/claude-0/-home-user-imtiredofthis/1701f8f3-afc9-57ea-bbc6-cce4dc6b5fbd/scratchpad/nfl_cache")

# ---- frozen windows (preregistration section 3) ----
FIT_SEASONS = (2020, 2021)
HOLDOUT_SEASON = 2022                      # never inspected; the only decisive evaluation
REPORTED_SEASONS = (2023, 2024, 2025)      # contaminated; cannot change the verdict
# ---- frozen thresholds (sections 4, 6) ----
MIN_PRIOR_TARGETS = 20
MIN_PRIOR_GAMES_WITH_TARGET = 3
FIRST_SCORABLE_WEEK = 4
LAST_WEEK = 18
SIGMA_FLOOR = 5.0
PRODUCTION_SIGMA = 26.0
PARTIAL_SNAP_FRACTION = 0.50
# ---- frozen evaluation (section 7) ----
Z20, Z80 = -0.8416212335729143, 0.8416212335729143
TARGET_COVERAGE = 0.60
CALIBRATED_TOLERANCE = 0.10
# NOT fixed by the preregistration -- disclosed as an implementation choice, with
# sensitivity reported at 6 and 10.
MIN_HOLDOUT_GAMES = 8
ARMS = ("S0", "S1", "S2")


def _load(kind: str, season: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{kind}_{season}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import nflreadpy as nfl
    fn = {"weekly": nfl.load_player_stats, "snaps": nfl.load_snap_counts}[kind]
    df = fn(seasons=[season]).to_pandas()
    df.to_parquet(p)
    return df


def state_label(targets: float, carries: float, snaps, snap_norm) -> str:
    """AGENTS.md anomaly gate: label before the observation becomes evidence."""
    if snaps is None or not np.isfinite(snaps):
        return "UNRESOLVED_CONTEXT" if (targets == 0 and carries == 0) else "HEALTHY_FULL_SNAPS_UNKNOWN"
    if snaps <= 0 and targets == 0 and carries == 0:
        return "DNP_INACTIVE"
    if snap_norm and np.isfinite(snap_norm) and snap_norm > 0 and snaps < PARTIAL_SNAP_FRACTION * snap_norm:
        return "PARTIAL_GAME"
    return "HEALTHY_FULL"


def build_panel(season: int) -> tuple[pd.DataFrame, dict]:
    w = _load("weekly", season)
    w = w[w["week"].between(1, LAST_WEEK)].copy()
    pos = "position" if "position" in w.columns else "position_group"
    w["pos"] = w[pos].astype(str).str.upper()
    w = w[w["pos"].isin(["WR", "TE"])].copy()
    name = "player_display_name" if "player_display_name" in w.columns else "player_name"
    w["player"] = w[name].astype(str)
    w["pid"] = w["player_id"].astype(str)
    for c in ("targets", "receiving_yards", "receptions", "carries"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)

    # role centrality: share of his team's targets that week
    tt = w.groupby(["team", "week"])["targets"].sum().rename("team_targets").reset_index()
    w = w.merge(tt, on=["team", "week"], how="left")
    w["tgt_share"] = w["targets"] / w["team_targets"].replace(0, np.nan)

    try:
        s = _load("snaps", season)
        s = s[["player", "week", "offense_snaps"]].copy()
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
            mu = (float(prior["receiving_yards"].sum()) / pt) * (pt / n_prior)
            rows.append({
                "season": season, "pid": pid, "player": cur["player"], "pos": cur["pos"], "week": week,
                "mu_hat": mu,
                "centrality": float(pd.to_numeric(prior["tgt_share"], errors="coerce").mean()),
                "y": float(cur["receiving_yards"]),
            })
    panel = pd.DataFrame(rows).dropna(subset=["mu_hat", "centrality"])
    return panel, {"season": season, "state_counts": w["state"].value_counts().to_dict(),
                   "panel_rows": int(len(panel)),
                   "panel_players": int(panel["pid"].nunique()) if len(panel) else 0}


def _terms(df: pd.DataFrame, arm: str) -> np.ndarray | None:
    if arm == "S0":
        return None
    cols = [np.ones(len(df)), df["mu_hat"].to_numpy(float)]
    if arm == "S2":
        cols.append(df["centrality"].to_numpy(float))
    return np.column_stack(cols)


def fit_sigma(fit: pd.DataFrame, arm: str) -> np.ndarray | None:
    """Frozen estimator: OLS of |y - mu| on the arm's terms, scaled by sqrt(pi/2)."""
    X = _terms(fit, arm)
    if X is None:
        return None
    resid = np.abs(fit["y"].to_numpy(float) - fit["mu_hat"].to_numpy(float))
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return beta


def predict_sigma(df: pd.DataFrame, arm: str, beta) -> np.ndarray:
    if arm == "S0":
        return np.full(len(df), PRODUCTION_SIGMA)
    raw = (_terms(df, arm) @ beta) * np.sqrt(np.pi / 2.0)
    return np.maximum(raw, SIGMA_FLOOR)


def _pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> np.ndarray:
    d = y - q_pred
    return np.maximum(q * d, (q - 1.0) * d)


def evaluate(df: pd.DataFrame, sigmas: dict, min_games: int = MIN_HOLDOUT_GAMES) -> dict:
    d = df.copy()
    for arm in ARMS:
        s = sigmas[arm]
        d[f"p20_{arm}"] = d["mu_hat"] + Z20 * s
        d[f"p80_{arm}"] = d["mu_hat"] + Z80 * s
        d[f"in60_{arm}"] = ((d["y"] >= d[f"p20_{arm}"]) & (d["y"] <= d[f"p80_{arm}"])).astype(float)
        d[f"pin20_{arm}"] = _pinball(d["y"].to_numpy(float), d[f"p20_{arm}"].to_numpy(float), 0.20)
        d[f"pin80_{arm}"] = _pinball(d["y"].to_numpy(float), d[f"p80_{arm}"].to_numpy(float), 0.80)
        d[f"sigma_{arm}"] = s
        d[f"floored_{arm}"] = (s <= SIGMA_FLOOR + 1e-12).astype(float)

    agg = {"games": ("y", "size"), "mu": ("mu_hat", "mean"), "player": ("player", "first")}
    for arm in ARMS:
        agg[f"cov_{arm}"] = (f"in60_{arm}", "mean")
        agg[f"pin20_{arm}"] = (f"pin20_{arm}", "mean")
        agg[f"pin80_{arm}"] = (f"pin80_{arm}", "mean")
        agg[f"sigma_{arm}"] = (f"sigma_{arm}", "mean")
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
            "sigma_median": float(per[f"sigma_{arm}"].median()),
            "sigma_range": [float(per[f"sigma_{arm}"].min()), float(per[f"sigma_{arm}"].max())],
            "frac_rows_on_sigma_floor": float(d[f"floored_{arm}"].mean()),
        }
    for arm in ("S1", "S2"):
        out[arm]["frac_players_pin20_improved"] = float((per[f"pin20_{arm}"] < per["pin20_S0"]).mean())
        out[arm]["frac_players_pin80_improved"] = float((per[f"pin80_{arm}"] < per["pin80_S0"]).mean())

    # floor/ceiling bias by mean tercile -- the star-compression table that must flatten
    per["tercile"] = pd.qcut(per["mu"], 3, labels=["low", "mid", "high"])
    emp = d.groupby("pid")["y"].agg(p20=lambda s: s.quantile(.20), p80=lambda s: s.quantile(.80))
    per = per.join(emp)
    bias = {}
    for arm in ARMS:
        pred20 = per["mu"] + Z20 * per[f"sigma_{arm}"]
        pred80 = per["mu"] + Z80 * per[f"sigma_{arm}"]
        t = pd.DataFrame({"tercile": per["tercile"], "fb": pred20 - per["p20"], "cb": pred80 - per["p80"]})
        g = t.groupby("tercile", observed=True).mean()
        bias[arm] = {str(k): {"floor_bias": float(v["fb"]), "ceiling_bias": float(v["cb"])}
                     for k, v in g.iterrows()}
    out["bias_by_mean_tercile"] = bias
    return out, per


def lognormal_secondary(df: pd.DataFrame, sigma: np.ndarray) -> dict:
    """Preregistration section 6 secondary: REPORTED, cannot change the verdict.

    Tests the diagnosis behind rule 2's failure. Centring a SYMMETRIC normal on the
    mean forces p20 and p80 equidistant, but the data puts the floor further from the
    mean than the ceiling -- which is what a right-skewed distribution looks like when
    you centre it on its MEAN rather than its median, since the mean sits above the
    median. A lognormal matched to the same mean and sd reproduces that asymmetry
    without changing either.
    """
    mu = df["mu_hat"].to_numpy(float)
    y = df["y"].to_numpy(float)
    ok = mu > 0
    cv2 = np.where(ok, (sigma / np.where(ok, mu, 1.0)) ** 2, np.nan)
    s_ln = np.sqrt(np.log1p(cv2))
    m_ln = np.log(np.where(ok, mu, 1.0)) - 0.5 * s_ln ** 2
    p20 = np.exp(m_ln + Z20 * s_ln)
    p80 = np.exp(m_ln + Z80 * s_ln)
    d = pd.DataFrame({"pid": df["pid"].to_numpy(), "y": y, "p20": p20, "p80": p80, "ok": ok})
    d = d[d["ok"]]
    d["in60"] = ((d["y"] >= d["p20"]) & (d["y"] <= d["p80"])).astype(float)
    d["pin20"] = _pinball(d["y"].to_numpy(), d["p20"].to_numpy(), 0.20)
    d["pin80"] = _pinball(d["y"].to_numpy(), d["p80"].to_numpy(), 0.80)
    per = d.groupby("pid").agg(games=("y", "size"), cov=("in60", "mean"),
                               pin20=("pin20", "mean"), pin80=("pin80", "mean"))
    per = per[per["games"] >= MIN_HOLDOUT_GAMES]
    err = (per["cov"] - TARGET_COVERAGE).abs()
    return {
        "players_scored": int(len(per)),
        "frac_well_calibrated": float((err <= CALIBRATED_TOLERANCE).mean()),
        "coverage_median": float(per["cov"].median()),
        "coverage_iqr": [float(per["cov"].quantile(.25)), float(per["cov"].quantile(.75))],
        "mean_downside_distance": float((df.loc[ok, "mu_hat"].to_numpy() - p20[ok]).mean()),
        "mean_upside_distance": float((p80[ok] - df.loc[ok, "mu_hat"].to_numpy()).mean()),
        "note": "REPORTED ONLY -- preregistered secondary, cannot change the verdict",
    }


def decide(ev: dict) -> dict:
    """Frozen four-part rule; all must hold on the clean holdout."""
    res = {}
    for arm in ("S1", "S2"):
        a, s0 = ev[arm], ev["S0"]
        r1 = a["frac_well_calibrated"] > s0["frac_well_calibrated"]
        r2 = a["frac_players_pin20_improved"] > 0.5 and a["frac_players_pin80_improved"] > 0.5
        hb_a = abs(ev["bias_by_mean_tercile"][arm]["high"]["floor_bias"])
        hb_0 = abs(ev["bias_by_mean_tercile"]["S0"]["high"]["floor_bias"])
        r3 = hb_a < hb_0
        r4 = a["frac_rows_on_sigma_floor"] <= 0.05 and a["sigma_range"][0] > 0
        res[arm] = {"rule_1_more_calibrated_players": bool(r1),
                    "rule_2_pinball_majority_both_tails": bool(r2),
                    "rule_3_star_compression_reduced": bool(r3),
                    "rule_4_sigma_well_behaved": bool(r4),
                    "SUPPORTED": bool(r1 and r2 and r3 and r4)}
    res["H1_SUPPORTED"] = bool(res["S1"]["SUPPORTED"] or res["S2"]["SUPPORTED"])
    res["preferred_arm"] = ("S1" if res["S1"]["SUPPORTED"] else
                            ("S2" if res["S2"]["SUPPORTED"] else None))
    return res


def main() -> int:
    prereg = Path("docs/research/PRISM_PLAYER_DISPERSION_PREREGISTRATION_V1.md")
    if not prereg.exists():
        raise RuntimeError("preregistration missing; this study may not be scored unfrozen")
    text = prereg.read_text(encoding="utf-8")
    for token in ("2020–2021", "**2022**", "26.0", "±0.10"):
        if token not in text:
            raise RuntimeError(f"preregistration no longer carries frozen value {token!r}")

    panels, diags = {}, []
    for yr in FIT_SEASONS + (HOLDOUT_SEASON,) + REPORTED_SEASONS:
        p, d = build_panel(yr)
        panels[yr] = p
        diags.append(d)

    fit = pd.concat([panels[y] for y in FIT_SEASONS], ignore_index=True)
    betas = {arm: fit_sigma(fit, arm) for arm in ARMS}

    def score(df: pd.DataFrame, min_games: int = MIN_HOLDOUT_GAMES):
        sig = {arm: predict_sigma(df, arm, betas[arm]) for arm in ARMS}
        return evaluate(df, sig, min_games)

    hold = panels[HOLDOUT_SEASON]
    ev, per = score(hold)
    decision = decide(ev)

    report = {
        "preregistration": prereg.as_posix(),
        "fit_seasons": list(FIT_SEASONS), "holdout_season": HOLDOUT_SEASON,
        "fit_rows": int(len(fit)),
        "fitted_coefficients": {a: (None if betas[a] is None else [float(v) for v in betas[a]])
                                for a in ARMS},
        "diagnostics": diags,
        "holdout_primary": ev,
        "decision": decision,
        "sensitivity_min_holdout_games": {
            str(g): {a: score(hold, g)[0][a]["frac_well_calibrated"] for a in ARMS} for g in (6, 10)
        },
        "reported_not_decisive": {str(y): score(panels[y])[0] for y in REPORTED_SEASONS
                                  if len(panels[y])},
        "lognormal_secondary_reported_only": {
            arm: lognormal_secondary(hold, predict_sigma(hold, arm, betas[arm])) for arm in ARMS
        },
        "sportsbook_inputs_used": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    per.reset_index().to_csv(OUT_ROWS, index=False)

    print(f"fit rows={len(fit)} ({FIT_SEASONS})   holdout {HOLDOUT_SEASON}: "
          f"{ev['rows']} rows, {ev['players_scored']} players scored\n")
    print("fitted sigma forms:")
    print(f"  S1: sigma = {betas['S1'][0]:.3f} + {betas['S1'][1]:.4f}*mu          (x sqrt(pi/2))")
    print(f"  S2: sigma = {betas['S2'][0]:.3f} + {betas['S2'][1]:.4f}*mu + {betas['S2'][2]:.3f}*centrality")
    print(f"\nPER-PLAYER CALIBRATION on {HOLDOUT_SEASON} (target coverage {TARGET_COVERAGE}):")
    for arm in ARMS:
        a = ev[arm]
        print(f"  {arm}: well-calibrated {a['frac_well_calibrated']:.1%}  "
              f"coverage median {a['coverage_median']:.3f} IQR {a['coverage_iqr'][0]:.2f}-{a['coverage_iqr'][1]:.2f}  "
              f"sigma median {a['sigma_median']:.1f} range {a['sigma_range'][0]:.1f}-{a['sigma_range'][1]:.1f}")
    for arm in ("S1", "S2"):
        print(f"  {arm}: players improved -- p20 {ev[arm]['frac_players_pin20_improved']:.1%}, "
              f"p80 {ev[arm]['frac_players_pin80_improved']:.1%}")
    print("\nfloor/ceiling bias by mean tercile (want flat, esp. high-mean floor):")
    for arm in ARMS:
        b = ev["bias_by_mean_tercile"][arm]
        print(f"  {arm}: " + "  ".join(
            f"{k} fb={b[k]['floor_bias']:+5.1f} cb={b[k]['ceiling_bias']:+5.1f}" for k in ("low", "mid", "high")))
    print("\nFROZEN DECISION RULE:")
    for arm in ("S1", "S2"):
        d = decision[arm]
        print(f"  {arm}: r1={d['rule_1_more_calibrated_players']} r2={d['rule_2_pinball_majority_both_tails']} "
              f"r3={d['rule_3_star_compression_reduced']} r4={d['rule_4_sigma_well_behaved']} "
              f"-> {'SUPPORTED' if d['SUPPORTED'] else 'not supported'}")
    print("\nLOGNORMAL SECONDARY (reported only, cannot change the verdict):")
    for arm in ARMS:
        ln = report["lognormal_secondary_reported_only"][arm]
        print(f"  {arm}: well-calibrated {ln['frac_well_calibrated']:.1%}  "
              f"coverage median {ln['coverage_median']:.3f}  "
              f"asymmetry: down {ln['mean_downside_distance']:.1f} vs up {ln['mean_upside_distance']:.1f}")
    print(f"\n  H1 SUPPORTED: {decision['H1_SUPPORTED']}   preferred arm: {decision['preferred_arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
