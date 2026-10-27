#!/usr/bin/env python3
"""Measure opponent gate reachability on 2025, per the frozen pre-registration.

Design: `docs/research/PRISM_GATE_PREVALENCE_PREREGISTRATION_V1.md`.
User-authorized 2026-10-27. Frontier named by GPT audit `6446595260`.

Outcome-blind: reads `defenders_in_box`, `defense_man_zone_type`, team identity,
week and `play_type`. No score, result, line, odds or price is read, and the
script asserts its own input columns.

Nothing is fitted, scored or promoted, and no threshold is proposed. The labels
NEAR_CONSTANT / NEAR_DEAD / DISCRIMINATING describe **reachability only** and
were fixed in the pre-registration before any per-team distribution was seen.

WHAT THIS IS NOT. Production normally PREFERS Sharp Football over the
participation path for box rates, so these figures describe what the
participation-derived fields would give, not necessarily the values production
used in a given week. This is a 2025 distributional prior, not a 2026 claim.

PRE-REGISTRATION UNDER-SPECIFICATION, DISCLOSED. The frozen design said "per
team-week" without stating whether a team-week rate is computed from that week's
plays alone or as the season-to-date value a pregame run would actually hold.
Both are reported rather than one being chosen after the fact. The cumulative
prior-weeks arm (`AS_OF_PREGAME`) is the production-faithful one, since a
pregame value for week N is built from weeks 1..N-1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SEASON = 2025
MAX_WEEK = 18

# Production's own definitions, from make_team_form.compute_personnel_rates.
LIGHT_MAX = 6   # light = defenders_in_box <= 6
HEAVY_MIN = 8   # heavy = defenders_in_box >= 8

# Production's gate conditions, from rules_v2.matchup_multipliers.
GATES = {
    "light_box_rate": (">=", 0.60),
    "heavy_box_rate": (">=", 0.60),
    "coverage_zone_rate": (">=", 0.60),
    "coverage_man_rate": (">=", 0.50),
}

# Pre-committed in the pre-registration, before any per-team distribution.
NEAR_CONSTANT = 0.90
NEAR_DEAD = 0.10

FORBIDDEN = ("score", "result", "epa", "wp", "line", "odds", "price", "spread",
             "total", "win")


def load_plays() -> pd.DataFrame:
    import nflreadpy as nfl

    def _pd(x):
        return x.to_pandas() if hasattr(x, "to_pandas") else x

    part = _pd(nfl.load_participation(seasons=[SEASON]))
    need = {"nflverse_game_id", "play_id", "possession_team",
            "defenders_in_box", "defense_man_zone_type"}
    missing = need - set(part.columns)
    if missing:
        raise RuntimeError(f"participation missing {sorted(missing)}")
    part = part[sorted(need)].copy()

    pbp = _pd(nfl.load_pbp(seasons=[SEASON]))
    pneed = {"game_id", "play_id", "defteam", "posteam", "week", "play_type"}
    pmissing = pneed - set(pbp.columns)
    if pmissing:
        raise RuntimeError(f"pbp missing {sorted(pmissing)}")
    pbp = pbp[sorted(pneed)].copy()

    m = part.merge(pbp, left_on=["nflverse_game_id", "play_id"],
                   right_on=["game_id", "play_id"], how="inner")
    # Outcome-blindness is asserted, not assumed.
    bad = [c for c in m.columns
           if any(k in c.lower() for k in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"joined frame carries outcome columns {bad}")

    m["week"] = pd.to_numeric(m["week"], errors="coerce")
    m = m.loc[m["week"].between(1, MAX_WEEK)].copy()
    m["box"] = pd.to_numeric(m["defenders_in_box"], errors="coerce")
    mz = m["defense_man_zone_type"].fillna("").astype(str).str.strip().str.upper()
    m["is_zone"] = mz.eq("ZONE_COVERAGE")
    m["is_man"] = mz.eq("MAN_COVERAGE")
    m["cov_classified"] = m["is_zone"] | m["is_man"]
    m["off"] = m["posteam"].map(canon_team)
    m["deff"] = m["defteam"].map(canon_team)
    m["scrimmage"] = m["play_type"].isin(["pass", "run"])
    return m


def weekly_rates(m: pd.DataFrame, team_col: str, scrimmage_only: bool,
                 cov_classified_only: bool,
                 literal_production_nulls: bool = False) -> pd.DataFrame:
    d = m.loc[m["scrimmage"]] if scrimmage_only else m
    d = d.loc[d[team_col].ne("")]
    # Audit 6448026674 finding 3. compute_personnel_rates() evaluates
    # `(to_numeric(box) <= 6).astype(float)` over EVERY participation row, and
    # NaN <= 6 is False, so a missing box counts in the denominator as
    # "not light". V1 of this script dropped missing-box rows from numerator
    # AND denominator, so the arm was not literally source-faithful. The
    # literal treatment is implemented here rather than argued immaterial.
    if literal_production_nulls:
        box = d
        light = box["box"].le(LIGHT_MAX)          # NaN -> False, kept in denom
        heavy = box["box"].ge(HEAVY_MIN)
    else:
        box = d.loc[d["box"].notna()]
        light = box["box"].le(LIGHT_MAX)
        heavy = box["box"].ge(HEAVY_MIN)
    box = box.assign(_light=light.astype(float), _heavy=heavy.astype(float))
    g = box.groupby([team_col, "week"], dropna=False)
    out = pd.DataFrame({
        "light_box_rate": g["_light"].mean(),
        "heavy_box_rate": g["_heavy"].mean(),
        "box_plays": g["_light"].size(),
        "box_null_rows": g["box"].apply(lambda s: int(s.isna().sum())),
    })
    c = d.loc[d["cov_classified"]] if cov_classified_only else d
    gc = c.groupby([team_col, "week"], dropna=False)
    cov = pd.DataFrame({
        "coverage_zone_rate": gc["is_zone"].mean(),
        "coverage_man_rate": gc["is_man"].mean(),
        "cov_plays": gc["is_zone"].size(),
    })
    out = out.join(cov, how="outer").reset_index()
    return out.rename(columns={team_col: "team"})


def as_of_pregame(weekly: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date value a pregame run for week N would actually hold.

    Rebuilt from prior weeks only, play-weighted, so it is the value the gate is
    evaluated against rather than a single noisy week.
    """
    rows = []
    for team, g in weekly.sort_values("week").groupby("team"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            prior = g.iloc[:i]
            if prior.empty:
                continue
            bw = prior["box_plays"].fillna(0)
            cw = prior["cov_plays"].fillna(0)
            def _w(col, wt):
                v = pd.to_numeric(prior[col], errors="coerce")
                ok = v.notna() & wt.gt(0)
                return float((v[ok] * wt[ok]).sum() / wt[ok].sum()) if ok.any() \
                    else np.nan
            rows.append({
                "team": team, "week": int(g.loc[i, "week"]),
                "light_box_rate": _w("light_box_rate", bw),
                "heavy_box_rate": _w("heavy_box_rate", bw),
                "coverage_zone_rate": _w("coverage_zone_rate", cw),
                "coverage_man_rate": _w("coverage_man_rate", cw),
                "prior_box_plays": float(bw.sum()),
                "prior_cov_plays": float(cw.sum()),
            })
    return pd.DataFrame(rows)


def gate_summary(df: pd.DataFrame) -> dict:
    out = {}
    for field, (op, thr) in GATES.items():
        # A missing column must not crash: df.get() returns None and
        # pd.to_numeric(None) yields a scalar nan, which has no .dropna().
        col = df[field] if field in getattr(df, "columns", []) else None
        v = (pd.to_numeric(col, errors="coerce").dropna()
             if col is not None else pd.Series(dtype="float64"))
        if v.empty:
            out[field] = {"team_weeks": 0, "label": "NO_DATA"}
            continue
        fires = (v >= thr) if op == ">=" else (v > thr)
        frac = float(fires.mean())
        out[field] = {
            "condition": f"{field} {op} {thr}",
            "team_weeks": int(len(v)),
            "team_weeks_firing": int(fires.sum()),
            "fraction_firing": round(frac, 4),
            "median_value": round(float(v.median()), 4),
            "p10": round(float(v.quantile(0.10)), 4),
            "p90": round(float(v.quantile(0.90)), 4),
            "label": ("NEAR_CONSTANT" if frac > NEAR_CONSTANT
                      else "NEAR_DEAD" if frac < NEAR_DEAD
                      else "DISCRIMINATING"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/gate_prevalence"))
    a = ap.parse_args()

    m = load_plays()
    arms, frames = {}, []
    for scrim in (False, True):
        for team_col, orient in (("off", "OFFENSE_FACED"),
                                 ("deff", "DEFENSE_DEPLOYED")):
            for covcls in (False, True):
                arm = (f"{'SCRIMMAGE_ONLY' if scrim else 'PRODUCTION_FAITHFUL_UNFILTERED'}"
                       f"__{orient}__{'COVERAGE_CLASSIFIED' if covcls else 'ALL_ROWS'}")
                wk = weekly_rates(m, team_col, scrim, covcls,
                                  literal_production_nulls=not scrim)
                ao = as_of_pregame(wk)
                wk["arm"], ao["arm"] = arm, arm
                wk["basis"], ao["basis"] = "SINGLE_WEEK", "AS_OF_PREGAME"
                frames += [wk, ao]
                arms[arm] = {"SINGLE_WEEK": gate_summary(wk),
                             "AS_OF_PREGAME": gate_summary(ao)}

    # Audit 6448026674 finding 1: the time basis is a NINTH dimension that the
    # preregistration left open. Report the label across every arm x basis so a
    # claim of robustness cannot quietly mean "within one basis".
    cross = {}
    for field in GATES:
        labs: dict[str, int] = {}
        for arm, bases in arms.items():
            for basis, summ in bases.items():
                labs[summ[field]["label"]] = labs.get(summ[field]["label"], 0) + 1
        cross[field] = {
            "labels_across_arm_x_basis": labs,
            "unconditional": len(labs) == 1,
            "n_combinations": sum(labs.values()),
        }

    detail = pd.concat(frames, ignore_index=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "opponent_gate_prevalence_v1"
    detail.to_csv(a.out_dir / f"{stem}_team_week.csv", index=False)

    status = {
        "version": "OPPONENT_GATE_PREVALENCE_V1",
        "preregistration": "docs/research/PRISM_GATE_PREVALENCE_PREREGISTRATION_V1.md",
        "authorized_by": "USER_2026_10_27; frontier named by GPT_6446595260",
        "scientific_boundary": "DESCRIPTIVE_REACHABILITY_ONLY_NO_FITTING_NO_SCORING",
        "outcomes_inspected": False,
        "sportsbook_inputs_used": 0,
        "season_measured": SEASON,
        "is_a_2026_claim": False,
        "weeks": [1, MAX_WEEK],
        "plays_joined": int(len(m)),
        "production_definitions": {
            "light": f"defenders_in_box <= {LIGHT_MAX}",
            "heavy": f"defenders_in_box >= {HEAVY_MIN}",
            "source": "make_team_form.compute_personnel_rates",
        },
        "prereg_labels": {"NEAR_CONSTANT": f"> {NEAR_CONSTANT}",
                          "NEAR_DEAD": f"< {NEAR_DEAD}"},
        "caveat_sharp_preferred": (
            "production PREFERS Sharp Football over the participation path for "
            "box rates, so these are the participation-derived values, not "
            "necessarily what production used in a given week"),
        "WITHDRAWN_SHARP_DENOMINATOR_INFERENCE": (
            "V1 concluded from the single source-conditioned 2026 W7 KC zone "
            "decode that production's Sharp-fed coverage fields behave like the "
            "COVERAGE_CLASSIFIED arm. Withdrawn per audit 6448026674 finding 4: "
            "the ALL_ROWS arm does fire rarely but nonzero (2/512 on "
            "scrimmage/defense-deployed), Sharp is a different provider whose "
            "denominator is unknown, and one inferred gate cannot identify a "
            "provider convention. Sharp's denominator and provenance are "
            "UNESTABLISHED."),
        "caveat_orientation": (
            "participation has NO defensive team column; production groups by "
            "possession_team, so its value is the box an OFFENSE faced while "
            "rules_v2 reads it as a defensive tendency. Both orientations are "
            "reported; which is correct is a semantic question for audit."),
        "disclosed_prereg_underspecification": (
            "the frozen design said 'per team-week' without fixing whether a "
            "team-week rate is a single week or the season-to-date value a "
            "pregame run would hold. Both are reported; AS_OF_PREGAME is the "
            "production-faithful basis."),
        "cross_basis_label_agreement": cross,
        "unconditional_gates": sorted(
            f for f, v in cross.items() if v["unconditional"]),
        "basis_dependent_gates": sorted(
            f for f, v in cross.items() if not v["unconditional"]),
        "HEADLINE_SCOPE": (
            "a gate is reported non-informative UNCONDITIONALLY only if its "
            "label is identical across all arm x basis combinations. Any other "
            "gate's label is conditional on the analysis choices named in the "
            "arm, and the time basis is one of those choices, not a universal "
            "result. AS_OF_PREGAME is preferred as the production-faithful "
            "planning basis -- a pregame value for week N is built from weeks "
            "1..N-1 -- and single-week rates additionally carry sampling noise "
            "from far fewer plays, which mechanically pushes a rate across a "
            "threshold. That is a reason to prefer it, not a reason to call "
            "its result basis-independent."),
        "gates_NOT_measured_here": [
            "middle_open_rate >= 0.50 (te_target_mult, slot_target_mult)",
            "pressure mismatch > 0.075 (rb_rec_target_mult, pass_eff_mult, "
            "sack_mult, int_mult, volatility_mult)",
        ],
        "arms": arms,
    }
    (a.out_dir / f"{stem}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"=== OPPONENT GATE PREVALENCE — {SEASON} weeks 1-{MAX_WEEK} ===")
    print(f"plays joined: {len(m)}\n")
    key = ("SCRIMMAGE_ONLY__DEFENSE_DEPLOYED__COVERAGE_CLASSIFIED",
           "PRODUCTION_FAITHFUL_UNFILTERED__OFFENSE_FACED__ALL_ROWS")
    for arm in key:
        print(f"--- {arm}  [AS_OF_PREGAME]")
        for f, s in arms[arm]["AS_OF_PREGAME"].items():
            if s.get("label") == "NO_DATA":
                print(f"    {f:22s} NO DATA"); continue
            print(f"    {f:22s} med={s['median_value']:.4f} "
                  f"fires {s['fraction_firing']:6.1%} of "
                  f"{s['team_weeks']:4d} team-weeks  -> {s['label']}")
        print()
    print("Labels describe REACHABILITY only, with cutoffs fixed in the")
    print("pre-registration. No threshold, coefficient or sign is proposed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
