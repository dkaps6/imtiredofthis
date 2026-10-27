"""Rate-scale gate bug: path attribution and one-variable effect sizing (W7 2026).

Audit 6456181944 approved an isolated, source-pinned, read-only shadow replay to
certify the player-mean consequence of the rate-scale guard before any merge.
Executing requirement 2 ("baseline parity first") produced a disposition the
proposal did not anticipate: the guard as committed is NOT on the code path that
produces player target share, so it cannot have a player-mean consequence.

Three findings, each independently checkable by the functions below.

1. PATH ATTRIBUTION. ``simulation_rules`` imports ``load_model_contexts`` from
   ``context_bridge`` (v1), which reads ``team_form.csv`` + ``cb_coverage_team.csv``
   directly. It never reads ``team_context_v3.csv``. ``apply_rules_to_metrics``
   additionally discards the teams dict (``_, players = load_model_contexts()``)
   and reads the TeamContext baked into each PlayerContext. A guard installed at
   the ``team_context_v3`` boundary therefore cannot change ``rules_tgt_share``.

2. ROOT CAUSE. ``make_team_form.PERCENTY_COLS`` is this repository's own existing
   percent->fraction mechanism, applied at Sharp ingest. It contains
   light_box_rate / heavy_box_rate / blitz_rate -- which is exactly why those
   arrive as fractions -- and omits coverage_man_rate / coverage_zone_rate /
   middle_open_rate / middle_closed_rate, which is exactly why those arrive as
   percents. The bug is a four-entry omission from an allowlist, one layer
   upstream of where the guard was built.

3. EFFECT SIZE, one variable, same fixture both arms. Correcting the scale removes
   19 false ``middle_open_rate >= 0.50`` fires (32/32 -> 13/32). Because
   ``slot_target_mult`` is structurally unreachable (PRISM_SLOT_ROLE_UNREACHABLE_V1),
   the gate's only live effect is ``te_target_mult *= 1.10``. The measured
   consequence is exactly one multiplier on exactly one position.

DISCLOSED LIMITATIONS -- this run is a counterfactual, not a production replay:

  * Baseline parity against the archive FAILS and is not repairable from this
    fixture. ``preserve_week7_opportunity_trace_v1`` subsets columns
    (``usecols=keep``), which dropped ``neutral_pace_last5`` and ``plays_est`` --
    both consumed by context_bridge v1 (lines 98, 100). ``rules_plays_est``
    differs from the archive by a per-team CONSTANT (within-team std exactly 0.0
    across all 28 slate teams), the signature of a differing team-context input,
    and ``rules_tgt_share`` differs universally as a cascade. Absolute levels here
    are therefore NOT production levels. The delta is still a clean differential
    because both arms consume the identical fixture and differ in one variable.
  * ``rules_ypt`` / ``rules_ypc`` are all-NaN in both arms (the fixture lacks the
    ypt/ypc inputs), so their invariance is UNMEASURED, not demonstrated.
  * Entitlement-stage teammate redistribution is not measured here. At the rules
    stage there is no room conservation, so the TE reduction is not yet offset.

Sportsbook separation: no market column is read. The fixture was built under a
denylist that excluded market/sportsbook columns, and ``market`` is absent from it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling import context_bridge as CB
from scripts.modeling import simulation_rules

FIXTURE = Path("data/research/week7_opportunity_trace")
PREFIX = "week7_trace__data__"
OUT_JSON = Path("data/research/rate_scale_path_attribution_v1.json")

# The four fields Sharp supplies as percents that PERCENTY_COLS omits.
MISSING_FROM_PERCENTY = (
    "coverage_man_rate",
    "coverage_zone_rate",
    "middle_open_rate",
    "middle_closed_rate",
)
# Fraction thresholds in rules_v2.matchup_multipliers.
GATES = (("middle_open_rate", 0.50), ("coverage_man_rate", 0.50), ("coverage_zone_rate", 0.60))
# Invariants that must not move under a pure rate-scale correction.
MUST_NOT_MOVE = ("rules_plays_est", "rules_pass_rate", "rules_rush_share")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE / f"{PREFIX}{name}.csv", low_memory=False)


def assert_source_contract() -> dict:
    """Fail closed if any structural fact this trace reports has since changed."""
    sim = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    mtf = Path("scripts/make_team_form.py").read_text(encoding="utf-8")
    cb1 = Path("scripts/modeling/context_bridge.py").read_text(encoding="utf-8")
    rules = Path("scripts/modeling/rules_v2.py").read_text(encoding="utf-8")

    if "from scripts.modeling.context_bridge import load_model_contexts" not in sim:
        raise RuntimeError(
            "simulation_rules no longer imports load_model_contexts from context_bridge (v1); "
            "finding 1 (path attribution) must be re-derived before it may be cited."
        )
    if "_, players = load_model_contexts()" not in sim:
        raise RuntimeError(
            "apply_rules_to_metrics no longer discards the teams dict; the claim that gate state "
            "reaches rules only via PlayerContext.offense/.defense must be re-derived."
        )
    if "team_context_v3" in cb1:
        raise RuntimeError("context_bridge v1 now references team_context_v3; path attribution is stale.")

    start = mtf.index("PERCENTY_COLS")
    block = mtf[start:mtf.index("}", start) + 1]
    still_missing = [c for c in MISSING_FROM_PERCENTY if f'"{c}"' not in block]
    if not still_missing:
        raise RuntimeError(
            "PERCENTY_COLS now contains every field this trace reports as omitted; the root cause "
            "has been fixed and these effect sizes describe a repaired build."
        )
    for field, thr in GATES:
        if f"{thr:.2f}" not in rules and str(thr) not in rules:
            raise RuntimeError(f"rules_v2 no longer carries the {field} threshold {thr}; re-derive gate state.")
    return {
        "simulation_rules_uses_context_bridge_v1": True,
        "apply_rules_to_metrics_discards_teams_dict": True,
        "context_bridge_v1_reads_team_context_v3": False,
        "percenty_cols_still_missing": still_missing,
    }


def path_attribution() -> dict:
    """Show team_context_v3 is absent from the rules path, by import graph."""
    sim = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    cb1 = Path("scripts/modeling/context_bridge.py").read_text(encoding="utf-8")
    readers = sorted(
        p.as_posix()
        for p in Path("scripts").rglob("*.py")
        if "context_bridge_v3" in p.read_text(encoding="utf-8") and "research" not in p.parts
    )
    return {
        "rules_stage_bridge": "scripts/modeling/context_bridge.py",
        "rules_stage_reads": sorted(
            f for f in ("team_form.csv", "cb_coverage_team.csv", "team_context_v3.csv") if f in cb1
        ),
        "guard_installed_in": "scripts/team_context_v3.py",
        "guard_on_rules_path": "team_context_v3" in sim or "team_context_v3" in cb1,
        "team_context_v3_consumers": readers,
    }


def _fix_scale(team_form: pd.DataFrame) -> pd.DataFrame:
    """Mirror make_team_form._prep_team_key_and_rates exactly: /100 iff max > 1.0."""
    out = team_form.copy()
    for col in MISSING_FROM_PERCENTY:
        if col not in out.columns:
            continue
        ser = pd.to_numeric(out[col], errors="coerce")
        if ser.notna().any() and ser.max(skipna=True) > 1.0:
            out[col] = ser / 100.0
    return out


def _arm(team_form: pd.DataFrame, fixture: dict) -> tuple[pd.DataFrame, dict]:
    """Build contexts from this team_form and run the real canonical rules transform."""
    teams = CB.build_team_contexts(team_form, fixture["coverage"])
    # Gate state reaches rules only through the TeamContext baked into each
    # PlayerContext, so players MUST be rebuilt from these teams.
    players = CB.build_player_contexts(
        fixture["player_form"], fixture["consensus"], teams,
        None, fixture["injuries"], None, fixture["team_week_map"],
    )
    original = simulation_rules.load_model_contexts
    simulation_rules.load_model_contexts = lambda *a, **k: (teams, players)
    try:
        priced = simulation_rules.apply_rules_to_metrics(
            fixture["metrics"].copy(), bayes_baseline=fixture["bayes"]
        )
    finally:
        simulation_rules.load_model_contexts = original
    return priced, teams


def gate_state(teams_asis: dict, teams_fixed: dict) -> dict:
    rows = {}
    for field, thr in GATES:
        a = [getattr(c, field) for c in teams_asis.values() if np.isfinite(getattr(c, field))]
        b = [getattr(c, field) for c in teams_fixed.values() if np.isfinite(getattr(c, field))]
        rows[field] = {
            "threshold": thr,
            "fires_as_is": int(sum(v >= thr for v in a)),
            "fires_fixed": int(sum(v >= thr for v in b)),
            "false_fires_removed": int(sum(v >= thr for v in a) - sum(v >= thr for v in b)),
            "teams_evaluated": len(a),
        }
    return rows


def one_variable_proof(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for col in MUST_NOT_MOVE:
        x = pd.to_numeric(a[col], errors="coerce")
        y = pd.to_numeric(b[col], errors="coerce")
        both = x.notna() & y.notna()
        out[col] = {
            "rows_compared": int(both.sum()),
            "max_abs_diff": float((x[both] - y[both]).abs().max()) if both.any() else 0.0,
            "verdict": "UNCHANGED" if both.any() and np.allclose(x[both], y[both], atol=1e-12) else
                       ("UNMEASURED_ALL_NAN" if not both.any() else "CHANGED"),
        }
    for col in ("rules_ypt", "rules_ypc"):
        x = pd.to_numeric(a[col], errors="coerce")
        y = pd.to_numeric(b[col], errors="coerce")
        both = x.notna() & y.notna()
        out[col] = {
            "rows_compared": int(both.sum()),
            "verdict": "UNMEASURED_ALL_NAN" if not both.any() else
                       ("UNCHANGED" if np.allclose(x[both], y[both], atol=1e-12) else "CHANGED"),
        }
    for col in ("rules_role", "position", "team", "opponent"):
        if col in a.columns and col in b.columns:
            out[col] = {"verdict": "UNCHANGED" if a[col].astype(str).equals(b[col].astype(str)) else "CHANGED"}
    return out


def effect_size(a: pd.DataFrame, b: pd.DataFrame, slate: set[str]) -> dict:
    d = pd.DataFrame({
        "position": a["position"].astype(str),
        "team": a["team"].astype(str),
        "role": a["rules_role"].astype(str),
        "player": a["player_clean_key"].astype(str),
        "as_is": pd.to_numeric(a["rules_tgt_share"], errors="coerce"),
        "fixed": pd.to_numeric(b["rules_tgt_share"], errors="coerce"),
    }).dropna(subset=["as_is", "fixed"])
    d["delta"] = d["fixed"] - d["as_is"]
    moved = d[d["delta"].abs() > 1e-12]
    rel = np.where(moved["as_is"] > 0, moved["delta"] / moved["as_is"], np.nan)
    by_pos = {
        str(pos): {"rows": int(len(grp)), "rows_moved": int((grp["delta"].abs() > 1e-12).sum())}
        for pos, grp in d.groupby("position")
    }
    unaffected = sorted(set(d["team"]) - set(moved["team"]))
    return {
        "rows_total": int(len(d)),
        "rows_moved": int(len(moved)),
        "positions_moved": sorted(moved["position"].unique().tolist()),
        "distinct_relative_multipliers": sorted({round(float(v), 6) for v in (1.0 + rel) if np.isfinite(v)}),
        "teams_moved": int(moved["team"].nunique()),
        "players_moved": int(moved["player"].nunique()),
        "mean_delta_among_moved": float(moved["delta"].mean()) if len(moved) else 0.0,
        "by_position": by_pos,
        "negative_control_unaffected_teams": len(unaffected),
        "negative_control_max_abs_delta": float(d[d["team"].isin(unaffected)]["delta"].abs().max())
        if unaffected else 0.0,
        "slate_teams": len(slate),
    }


def coverage_fallback_reach(team_form: pd.DataFrame, coverage: pd.DataFrame, slate: set[str]) -> dict:
    """Teams whose coverage falls back to percent-scaled team_form, and whether they play."""
    covered = set(coverage["team"].astype(str))
    fallback = sorted(set(team_form["team"].astype(str)) - covered)
    return {
        "fallback_teams": fallback,
        "fallback_teams_on_slate": sorted(set(fallback) & slate),
        "man_gate_false_fires_reach_slate": bool(set(fallback) & slate),
    }


def main() -> int:
    contract = assert_source_contract()
    fixture = {
        "metrics": _read("metrics_ready"),
        "bayes": _read("model_bayesian_diagnostics"),
        "coverage": _read("cb_coverage_team"),
        "player_form": _read("player_form"),
        "consensus": _read("player_form_consensus"),
        "team_week_map": _read("team_week_map"),
        "injuries": _read("injuries"),
    }
    team_form = _read("team_form")
    slate = set(fixture["team_week_map"]["team"].astype(str))

    as_is, teams_asis = _arm(team_form, fixture)
    fixed, teams_fixed = _arm(_fix_scale(team_form), fixture)

    report = {
        "source_contract": contract,
        "path_attribution": path_attribution(),
        "gate_state": gate_state(teams_asis, teams_fixed),
        "one_variable_proof": one_variable_proof(as_is, fixed),
        "effect_size": effect_size(as_is, fixed, slate),
        "coverage_fallback_reach": coverage_fallback_reach(team_form, fixture["coverage"], slate),
        "disclosed_limitations": [
            "baseline parity against the archive FAILS: the preserved fixture is a column subset "
            "missing neutral_pace_last5 and plays_est, both consumed by context_bridge v1; "
            "absolute levels here are counterfactual, only the same-fixture differential is clean",
            "rules_ypt/rules_ypc invariance is UNMEASURED (all-NaN in both arms), not demonstrated",
            "entitlement-stage teammate redistribution is not measured by this trace",
        ],
        "sportsbook_inputs_used": False,
    }

    moved = report["effect_size"]["rows_moved"]
    mults = report["effect_size"]["distinct_relative_multipliers"]
    if moved and report["effect_size"]["positions_moved"] != ["TE"]:
        raise RuntimeError(
            f"expected the live gate effect to be TE-only (slot_target_mult is unreachable); "
            f"got {report['effect_size']['positions_moved']}"
        )
    if moved and (len(mults) != 1 or abs(mults[0] - 1.0 / 1.10) > 1e-6):
        raise RuntimeError(
            f"expected exactly one relative multiplier 1/1.10 from removing the false middle_open "
            f"fire; got {mults}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON}")
    print(f"\nguard on rules path: {report['path_attribution']['guard_on_rules_path']}")
    print(f"PERCENTY_COLS still missing: {contract['percenty_cols_still_missing']}")
    for field, row in report["gate_state"].items():
        print(f"  {field:20s} >={row['threshold']}: {row['fires_as_is']}/{row['teams_evaluated']} "
              f"-> {row['fires_fixed']}/{row['teams_evaluated']} (false removed {row['false_fires_removed']})")
    es = report["effect_size"]
    print(f"\nrows moved: {es['rows_moved']}/{es['rows_total']}  positions={es['positions_moved']}  "
          f"multiplier={es['distinct_relative_multipliers']}")
    print(f"teams moved: {es['teams_moved']}  players moved: {es['players_moved']}  "
          f"negative-control teams: {es['negative_control_unaffected_teams']} "
          f"(max|delta|={es['negative_control_max_abs_delta']:.3e})")
    print(f"\ncoverage fallback teams: {report['coverage_fallback_reach']['fallback_teams']} "
          f"-> on slate: {report['coverage_fallback_reach']['fallback_teams_on_slate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
