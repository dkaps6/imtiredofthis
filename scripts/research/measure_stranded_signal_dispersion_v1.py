#!/usr/bin/env python3
"""Dispersion of the stranded opponent signals, 2026 weeks 2-7.

Design frozen in `docs/research/PRISM_STRANDED_SIGNAL_DISPERSION_PREREGISTRATION_V1.md`
(commit `fb803940`, before this measurement). User-authorized 2026-10-27.

The cleared transmission trace established that `def_rush_epa` is carried onto
`TeamContext` and read by no rule, and that `rush_stuff_rate` and
yards-before-contact are referenced by no production module at all. It did not
establish how much matchup information that discards. This measures
**dispersion and rank stability only** and cannot show that consuming any
signal would improve a projection.

CHRONOLOGY FAILS CLOSED. Every opponent aggregate for target week N is built
from weeks 1..N-1 only, play-weighted. Week 1 is excluded rather than
backfilled. `_assert_strictly_prior()` raises if any aggregate draws on a play
at week >= N.

Outcome-blind for the target week: reads defensive team aggregates from PRIOR
weeks plus the schedule. No player result, box score, line, odds or price.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = 2026
TARGET_WEEKS = range(2, 8)      # week 1 has no prior evidence

EXPLOSIVE_RUSH_YDS = 10         # conventional nflverse definitions, fixed
EXPLOSIVE_PASS_YDS = 20
STUFF_MAX_YDS = 0

SIGNALS = ("def_rush_epa", "def_pass_epa", "success_rate_def",
           "explosive_rush_rate_allowed", "explosive_pass_rate_allowed",
           "rush_stuff_rate")

# Pre-committed in the pre-registration.
MIN_RANK_CORR = 0.0             # must be positive to be MATERIAL
TRIVIAL_IQR = {                 # below this the signal is NARROW
    "def_rush_epa": 0.05, "def_pass_epa": 0.05, "success_rate_def": 0.02,
    "explosive_rush_rate_allowed": 0.02, "explosive_pass_rate_allowed": 0.02,
    "rush_stuff_rate": 0.02,
}

FORBIDDEN_PLAYER = ("player", "receiver", "rusher", "passer", "fantasy")


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl
    b = nfl.load_pbp(seasons=[SEASON])
    b = b.to_pandas() if hasattr(b, "to_pandas") else b
    need = {"week", "defteam", "posteam", "play_type", "epa", "success",
            "yards_gained"}
    missing = need - set(b.columns)
    if missing:
        raise RuntimeError(f"2026 pbp missing {sorted(missing)}")
    d = b[sorted(need)].copy()
    # No player-level column may enter this study.
    bad = [c for c in d.columns
           if any(k in c.lower() for k in FORBIDDEN_PLAYER)]
    if bad:
        raise RuntimeError(f"player-level columns present: {bad}")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d["epa"] = pd.to_numeric(d["epa"], errors="coerce")
    d["success"] = pd.to_numeric(d["success"], errors="coerce")
    d["yards_gained"] = pd.to_numeric(d["yards_gained"], errors="coerce")
    return d.loc[d["week"].notna() & d["defteam"].notna()].copy()


def _aggregate(plays: pd.DataFrame) -> pd.DataFrame:
    run = plays.loc[plays["play_type"].eq("run")]
    pas = plays.loc[plays["play_type"].eq("pass")]
    out = pd.DataFrame(index=sorted(plays["defteam"].dropna().unique()))
    out.index.name = "defense"
    out["def_rush_epa"] = run.groupby("defteam")["epa"].mean()
    out["def_pass_epa"] = pas.groupby("defteam")["epa"].mean()
    out["success_rate_def"] = plays.groupby("defteam")["success"].mean()
    out["explosive_rush_rate_allowed"] = run.groupby("defteam")["yards_gained"] \
        .apply(lambda s: (s >= EXPLOSIVE_RUSH_YDS).mean())
    out["explosive_pass_rate_allowed"] = pas.groupby("defteam")["yards_gained"] \
        .apply(lambda s: (s >= EXPLOSIVE_PASS_YDS).mean())
    out["rush_stuff_rate"] = run.groupby("defteam")["yards_gained"] \
        .apply(lambda s: (s <= STUFF_MAX_YDS).mean())
    out["prior_plays"] = plays.groupby("defteam").size()
    return out.reset_index()


def _assert_strictly_prior(plays: pd.DataFrame, target_week: int) -> None:
    mx = pd.to_numeric(plays["week"], errors="coerce").max()
    if pd.notna(mx) and mx >= target_week:
        raise RuntimeError(
            f"chronology violation: aggregate for week {target_week} draws on "
            f"a play at week {mx}")


def build_as_of(pbp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wk in TARGET_WEEKS:
        prior = pbp.loc[pbp["week"].lt(wk)]
        _assert_strictly_prior(prior, wk)
        if prior.empty:
            continue
        agg = _aggregate(prior)
        agg["target_week"] = wk
        rows.append(agg)
    if not rows:
        raise RuntimeError("no as-of aggregates built")
    return pd.concat(rows, ignore_index=True)


def dispersion(as_of: pd.DataFrame) -> dict:
    out = {}
    for sig in SIGNALS:
        per_week = []
        for wk, g in as_of.groupby("target_week"):
            v = pd.to_numeric(g[sig], errors="coerce").dropna()
            if len(v) < 8:
                continue
            per_week.append({
                "target_week": int(wk), "defenses": int(len(v)),
                "sd": round(float(v.std(ddof=1)), 6),
                "iqr": round(float(v.quantile(.75) - v.quantile(.25)), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6),
                "range": round(float(v.max() - v.min()), 6),
            })
        # rank stability between consecutive target weeks
        corrs = []
        wks = sorted(as_of["target_week"].unique())
        for a, b in zip(wks, wks[1:]):
            ga = as_of.loc[as_of["target_week"].eq(a)].set_index("defense")[sig]
            gb = as_of.loc[as_of["target_week"].eq(b)].set_index("defense")[sig]
            j = pd.concat([ga, gb], axis=1, keys=["a", "b"]).dropna()
            if len(j) >= 8:
                corrs.append(round(float(j["a"].corr(j["b"], method="spearman")), 4))
        med_iqr = float(np.median([p["iqr"] for p in per_week])) if per_week else np.nan
        med_corr = float(np.median(corrs)) if corrs else np.nan
        trivial = TRIVIAL_IQR[sig]
        if not per_week or np.isnan(med_iqr):
            label = "NO_DATA"
        elif med_iqr <= trivial:
            label = "NARROW"
        elif np.isnan(med_corr) or med_corr <= MIN_RANK_CORR:
            label = "UNSTABLE"
        else:
            label = "MATERIAL_DISPERSION"
        out[sig] = {
            "per_target_week": per_week,
            "median_iqr": round(med_iqr, 6) if per_week else None,
            "trivial_iqr_threshold": trivial,
            "consecutive_week_rank_corr": corrs,
            "median_rank_corr": round(med_corr, 4) if corrs else None,
            "label": label,
        }
    return out


# Addendum V2, frozen at 5618937d. Disjoint windows sharing no play, so the
# correlation is a genuine out-of-sample persistence estimate rather than the
# overlapping-history autocorrelation V1 reported.
EARLY_WEEKS, LATE_WEEKS = (1, 2, 3), (4, 5, 6, 7)
MIN_PLAYS_PER_HALF = 40
PERSIST_STRONG, PERSIST_WEAK = 0.40, 0.20

RUSH_FAMILY = ("def_rush_epa", "explosive_rush_rate_allowed", "rush_stuff_rate")
PASS_FAMILY = ("def_pass_epa", "explosive_pass_rate_allowed")

# Blocker 3: provenance per signal, not one blanket "stranded" label.
PROVENANCE = {
    "def_rush_epa": ("TRANSPORTED_BUT_STRANDED",
                     "populated onto TeamContext by context_bridge_v3 and read "
                     "by no rule (cleared transmission trace)"),
    "def_pass_epa": ("TRANSPORTED_AND_CONSUMED_ON_THE_QB_ROUTE",
                     "read by qb_pass_synthesis_v1, the promoted M89/M90 path"),
    "success_rate_def": ("READ_BY_PRODUCTION",
                         "rules_v2.success_diff(offense, defense) reads "
                         "defense.success_rate_def, called in "
                         "project_game_script; whether it changes a named "
                         "player's final projection is a SEPARATE routing "
                         "question, since pass_share is hardcoded at 0.57"),
    "explosive_rush_rate_allowed": ("NEW_PBP_DERIVED_CANDIDATE",
                                    "computed here; production has a column "
                                    "named explosive_play_rate_allowed which is "
                                    "transported-but-stranded, but lineage from "
                                    "it to this construct is NOT established"),
    "explosive_pass_rate_allowed": ("NEW_PBP_DERIVED_CANDIDATE",
                                    "as above; lineage NOT established"),
    "rush_stuff_rate": ("NEW_PBP_DERIVED_CANDIDATE",
                        "production has a column of this name referenced by no "
                        "module; lineage from it to this construct is NOT "
                        "established"),
}


def _family_mask(plays: pd.DataFrame, sig: str) -> pd.Series:
    if sig in RUSH_FAMILY:
        return plays["play_type"].eq("run")
    if sig in PASS_FAMILY:
        return plays["play_type"].eq("pass")
    return pd.Series(True, index=plays.index)


def disjoint_persistence(pbp: pd.DataFrame) -> dict:
    """Spearman rho between two windows that share no play."""
    early = pbp.loc[pbp["week"].isin(EARLY_WEEKS)]
    late = pbp.loc[pbp["week"].isin(LATE_WEEKS)]
    assert set(EARLY_WEEKS).isdisjoint(LATE_WEEKS), "windows overlap"
    ea, la = _aggregate(early), _aggregate(late)
    out = {}
    for sig in SIGNALS:
        m_e, m_l = _family_mask(early, sig), _family_mask(late, sig)
        ce = early.loc[m_e].groupby("defteam").size()
        cl = late.loc[m_l].groupby("defteam").size()
        ok = set(ce[ce >= MIN_PLAYS_PER_HALF].index) & \
             set(cl[cl >= MIN_PLAYS_PER_HALF].index)
        a = ea.set_index("defense")[sig]
        b = la.set_index("defense")[sig]
        j = pd.concat([a, b], axis=1, keys=["early", "late"]).dropna()
        j = j.loc[j.index.isin(ok)]
        excluded = sorted(set(a.index) - set(j.index))
        if len(j) < 8:
            out[sig] = {"eligible_defenses": int(len(j)),
                        "label": "INSUFFICIENT_ELIGIBLE"}
            continue
        rho = float(j["early"].corr(j["late"], method="spearman"))
        out[sig] = {
            "eligible_defenses": int(len(j)),
            "excluded_defenses": excluded,
            "min_plays_per_half": MIN_PLAYS_PER_HALF,
            "disjoint_spearman_rho": round(rho, 4),
            "label": ("PERSISTENT" if rho >= PERSIST_STRONG
                      else "WEAKLY_PERSISTENT" if rho >= PERSIST_WEAK
                      else "NOT_PERSISTENT"),
        }
    return out


def faced_dispersion(as_of: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    """How much the matchup an offense confronts changes week to week."""
    sched = (pbp.loc[pbp["week"].isin(list(TARGET_WEEKS))]
             .groupby(["posteam", "week"])["defteam"]
             .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
             .reset_index().rename(columns={"week": "target_week",
                                            "defteam": "defense"}))
    j = sched.merge(as_of, on=["target_week", "defense"], how="left")
    out = {}
    for sig in SIGNALS:
        per_off = []
        for off, g in j.groupby("posteam"):
            v = pd.to_numeric(g[sig], errors="coerce").dropna()
            if len(v) < 3:
                continue
            per_off.append(float(v.max() - v.min()))
        out[sig] = {
            "offenses": len(per_off),
            "median_faced_range": round(float(np.median(per_off)), 6)
            if per_off else None,
            "max_faced_range": round(float(np.max(per_off)), 6)
            if per_off else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/stranded_dispersion"))
    a = ap.parse_args()

    pbp = load_pbp()
    as_of = build_as_of(pbp)
    disp = dispersion(as_of)
    faced = faced_dispersion(as_of, pbp)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "stranded_signal_dispersion_v1"
    as_of.to_csv(a.out_dir / f"{stem}_as_of.csv", index=False)
    status = {
        "version": "STRANDED_SIGNAL_DISPERSION_V1",
        "preregistration": ("docs/research/"
                            "PRISM_STRANDED_SIGNAL_DISPERSION_PREREGISTRATION_V1.md"),
        "authorized_by": "USER_2026_10_27",
        "scientific_boundary": "DISPERSION_AND_STABILITY_ONLY_NO_FITTING_NO_SCORING",
        "season": SEASON,
        "target_weeks": list(TARGET_WEEKS),
        "week1_excluded_no_prior_evidence": True,
        "chronology": "every aggregate for week N uses weeks 1..N-1 only; asserted",
        # Blocker 4: "no outcomes" was absolute and wrong. load_pbp reads
        # realized epa/success/yards_gained from played games.
        "target_week_outcomes_inspected": False,
        "prior_game_play_results_used": True,
        "outcome_language_note": (
            "prior-game play results (epa, success, yards_gained) are used ONLY "
            "to construct as-of defensive statistics. No target-week outcome is "
            "read and no model is scored. The V1 phrasing 'outcomes inspected: "
            "none' was absolute and is withdrawn (audit 6449349551 blocker 4)."),
        "model_scoring_performed": False,
        "player_level_columns_read": False,
        "sportsbook_inputs_used": 0,
        "plays": int(len(pbp)),
        "definitions": {
            "explosive_rush": f"yards_gained >= {EXPLOSIVE_RUSH_YDS}",
            "explosive_pass": f"yards_gained >= {EXPLOSIVE_PASS_YDS}",
            "stuff": f"yards_gained <= {STUFF_MAX_YDS}",
            "note": "conventional nflverse definitions, fixed not tuned",
        },
        "signal_provenance": {k: {"class": v[0], "note": v[1]}
                              for k, v in PROVENANCE.items()},
        "NOT_SIX_STRANDED_PRODUCTION_FIELDS": (
            "audit 6449349551 blocker 3. These are not six discarded production "
            "columns. success_rate_def IS read by production via "
            "rules_v2.success_diff inside project_game_script; def_pass_epa is "
            "consumed on the promoted QB route; and the explosive/stuff "
            "constructs are NEW PBP-derived candidates whose lineage to any "
            "production column is NOT established. Only def_rush_epa is "
            "transported-but-stranded on the cleared evidence."),
        "disjoint_persistence": disjoint_persistence(pbp),
        "RETRACTED_OVERLAPPING_PERSISTENCE": (
            "the V1 consecutive-week rank correlations (0.764-0.883) compare "
            "cumulative windows sharing all but one week, so they are "
            "overlapping-history autocorrelations and are forced by sample "
            "reuse. 'Persistent properties of a defense' and 'not noise' are "
            "WITHDRAWN as established. See disjoint_persistence for the "
            "replacement, frozen at 5618937d before it was run."),
        "IQR_CUTOFFS_NOT_PREREGISTERED": (
            "TRIVIAL_IQR (0.05 EPA, 0.02 rates) was never in the frozen V1 "
            "design, which said only 'non-trivial'. MATERIAL_DISPERSION is "
            "POST-PREREGISTRATION OPERATIONALIZATION, not a predeclared "
            "verdict. The numeric dispersion results stand."),
        "cross_sectional_dispersion": disp,
        "faced_dispersion": faced,
        "labels_are_dispersion_only": (
            "MATERIAL_DISPERSION / UNSTABLE / NARROW describe spread and rank "
            "stability. They are NOT evidence of predictive value and no "
            "threshold, coefficient or sign is proposed."),
        "cannot_establish": [
            "that consuming any stranded signal would improve accuracy",
            "anything about the box/coverage gates on 2026 (participation-locked)",
            "that production's def_rush_epa is constructed like this one",
        ],
    }
    (a.out_dir / f"{stem}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"=== STRANDED SIGNAL DISPERSION — {SEASON} target weeks "
          f"{TARGET_WEEKS.start}-{TARGET_WEEKS.stop - 1} ===")
    print(f"plays: {len(pbp)}  (week 1 excluded: no prior evidence)\n")
    print(f"{'signal':30s} {'med IQR':>9s} {'rank r':>7s} {'faced range':>12s}  label")
    for sig in SIGNALS:
        d, f = disp[sig], faced[sig]
        mi = d["median_iqr"]
        mc = d["median_rank_corr"]
        fr = f["median_faced_range"]
        print(f"{sig:30s} {mi if mi is None else f'{mi:9.4f}'} "
              f"{mc if mc is None else f'{mc:7.3f}'} "
              f"{fr if fr is None else f'{fr:12.4f}'}  {d['label']}")
    print("\n=== DISJOINT PERSISTENCE (weeks 1-3 vs 4-7, no shared play) ===")
    dp = status["disjoint_persistence"]
    print(f"{'signal':30s} {'rho':>7s} {'n':>4s}  label")
    for sig in SIGNALS:
        d = dp[sig]
        r = d.get("disjoint_spearman_rho")
        print(f"{sig:30s} {'n/a' if r is None else f'{r:7.3f}'} "
              f"{d['eligible_defenses']:4d}  {d['label']}")
    print("\nThe V1 consecutive-week correlations are WITHDRAWN as evidence of")
    print("persistence: they compared windows sharing all but one week.")
    print("Dispersion and rank persistence only. No predictive claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
