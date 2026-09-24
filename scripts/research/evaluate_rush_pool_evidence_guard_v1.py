#!/usr/bin/env python3
"""Apply the frozen Rush Pool Evidence Guard V1 qualification gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VERSION = "RUSH_POOL_EVIDENCE_GUARD_V1"


def _read(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing season summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _m(s: dict, group: str, key: str) -> float:
    value = s["metrics"][group][key]
    if value is None:
        raise RuntimeError(f"missing metric {group}.{key}")
    return float(value)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-2024", type=Path, required=True)
    p.add_argument("--summary-2025", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--result-md", type=Path, required=True)
    a = p.parse_args()

    s24 = _read(a.summary_2024)
    s25 = _read(a.summary_2025)
    for s, year in ((s24, 2024), (s25, 2025)):
        if s.get("version") != VERSION or int(s.get("season")) != year:
            raise RuntimeError(f"wrong summary authority for {year}")
        if int(s.get("sportsbook_inputs_used", -1)) != 0:
            raise RuntimeError("sportsbook input integrity failure")
        if int(s.get("parameters_fit", -1)) != 0 or int(s.get("candidate_variants_scored", -1)) != 1:
            raise RuntimeError("frozen one-candidate contract violated")
        if not bool(s.get("week1_noop_verified")):
            raise RuntimeError("Week-1 no-op integrity failure")

    # Pooled MAE uses n-weighted absolute-error mass from the two independent years.
    n24 = int(s24["metrics"]["ALL"]["n"])
    n25 = int(s25["metrics"]["ALL"]["n"])
    bpool = (_m(s24, "ALL", "baseline_mae") * n24 + _m(s25, "ALL", "baseline_mae") * n25) / (n24 + n25)
    cpool = (_m(s24, "ALL", "candidate_mae") * n24 + _m(s25, "ALL", "candidate_mae") * n25) / (n24 + n25)
    pooled_improvement = bpool - cpool

    gates = {
        "all_mae_improves_2024": _m(s24, "ALL", "candidate_mae") < _m(s24, "ALL", "baseline_mae"),
        "all_mae_improves_2025": _m(s25, "ALL", "candidate_mae") < _m(s25, "ALL", "baseline_mae"),
        "rb_mae_improves_2024": _m(s24, "RB_FAMILY", "candidate_mae") < _m(s24, "RB_FAMILY", "baseline_mae"),
        "rb_mae_improves_2025": _m(s25, "RB_FAMILY", "candidate_mae") < _m(s25, "RB_FAMILY", "baseline_mae"),
        "pooled_all_mae_improvement_ge_002": pooled_improvement >= 0.02,
        "all_p90_nonworse_2024": _m(s24, "ALL", "candidate_p90") <= _m(s24, "ALL", "baseline_p90") + 1e-12,
        "all_p90_nonworse_2025": _m(s25, "ALL", "candidate_p90") <= _m(s25, "ALL", "baseline_p90") + 1e-12,
        "rb_p90_nonworse_2024": _m(s24, "RB_FAMILY", "candidate_p90") <= _m(s24, "RB_FAMILY", "baseline_p90") + 1e-12,
        "rb_p90_nonworse_2025": _m(s25, "RB_FAMILY", "candidate_p90") <= _m(s25, "RB_FAMILY", "baseline_p90") + 1e-12,
        "qb_mae_nonworse_2024": _m(s24, "QB", "candidate_mae") <= _m(s24, "QB", "baseline_mae") + 1e-12,
        "qb_mae_nonworse_2025": _m(s25, "QB", "candidate_mae") <= _m(s25, "QB", "baseline_mae") + 1e-12,
        "changed_rows_candidate_closer_2024": _m(s24, "ALL", "candidate_closer_rate") > 0.50,
        "changed_rows_candidate_closer_2025": _m(s25, "ALL", "candidate_closer_rate") > 0.50,
        "omitted_evidenced_decreases_2024": int(s24["candidate_omitted_evidenced_positive"]) < int(s24["baseline_omitted_evidenced_positive"]),
        "omitted_evidenced_decreases_2025": int(s25["candidate_omitted_evidenced_positive"]) < int(s25["baseline_omitted_evidenced_positive"]),
        "integrity_2024": int(s24["sportsbook_inputs_used"]) == 0 and bool(s24["week1_noop_verified"]),
        "integrity_2025": int(s25["sportsbook_inputs_used"]) == 0 and bool(s25["week1_noop_verified"]),
    }
    qualified = all(gates.values())
    disposition = f"{VERSION}_{'QUALIFIED' if qualified else 'FAILED_CLOSED'}"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "qualified": qualified,
        "production_changed": False,
        "parameters_fit": 0,
        "candidate_variants_scored": 1,
        "pooled": {
            "n": n24 + n25,
            "baseline_all_rush_att_mae": bpool,
            "candidate_all_rush_att_mae": cpool,
            "improvement": pooled_improvement,
        },
        "season_2024": s24,
        "season_2025": s25,
        "gates": gates,
        "stopping_rule": "No rescue, pool-size search, threshold search, position priority, depth-chart gate, rookie/QB exception, Bayesian retune, or 2026 outcome fit.",
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Rush Pool Evidence Guard V1 — Frozen Result",
        "",
        f"Disposition: **{disposition}**",
        "",
        f"Pooled ALL rush-attempt MAE: `{bpool:.6f} -> {cpool:.6f}` (improvement `{pooled_improvement:.6f}`)",
        "",
    ]
    for year, s in ((2024, s24), (2025, s25)):
        lines += [
            f"## {year}",
            "",
            f"- selector-changed team-games: `{int(s['selector_changed_team_games'])}` / `{int(s['team_games'])}`",
            f"- omitted evidenced positive players: `{int(s['baseline_omitted_evidenced_positive'])} -> {int(s['candidate_omitted_evidenced_positive'])}`",
            f"- mean position-prior-only carry mass: `{float(s['baseline_position_prior_only_mass_mean']):.6f} -> {float(s['candidate_position_prior_only_mass_mean']):.6f}`",
            f"- ALL MAE: `{_m(s,'ALL','baseline_mae'):.6f} -> {_m(s,'ALL','candidate_mae'):.6f}`",
            f"- ALL p90: `{_m(s,'ALL','baseline_p90'):.6f} -> {_m(s,'ALL','candidate_p90'):.6f}`",
            f"- RB/FB/HB MAE: `{_m(s,'RB_FAMILY','baseline_mae'):.6f} -> {_m(s,'RB_FAMILY','candidate_mae'):.6f}`",
            f"- RB/FB/HB p90: `{_m(s,'RB_FAMILY','baseline_p90'):.6f} -> {_m(s,'RB_FAMILY','candidate_p90'):.6f}`",
            f"- QB MAE: `{_m(s,'QB','baseline_mae'):.6f} -> {_m(s,'QB','candidate_mae'):.6f}`",
            f"- changed-row candidate closer rate: `{_m(s,'ALL','candidate_closer_rate'):.6f}`",
            "",
        ]
    lines += ["## Frozen gates", ""]
    for key, passed in gates.items():
        lines.append(f"- {key}: **{'PASS' if passed else 'FAIL'}**")
    lines += [
        "",
        "Production changed: **false**.",
        "",
        "If qualified, freeze a separate full-stack integration plan before editing production.",
        "If failed, this exact candidate is closed with no rescue.",
    ]
    a.result_md.parent.mkdir(parents=True, exist_ok=True)
    a.result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
