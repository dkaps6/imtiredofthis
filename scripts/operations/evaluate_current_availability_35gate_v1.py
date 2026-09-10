#!/usr/bin/env python3
"""Evaluate frozen Current Player Availability Full Slate gates 1-34.

Gate 35 is intentionally finalized only after Actions uploads the immutable
pre-result evidence and returns its artifact id/digest. No gate is optimized or
retuned here; definitions follow the previously frozen integration plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

DATA = Path("data")
OUT = DATA / "current_player_availability_35gate_preliminary.json"

CORE_HASHES = {
    "scripts/providers/ourlads_depth_status_v1.py": "c115816ea8aa4ba7150a635c3115546d43f94b3c",
    "scripts/build/build_current_player_availability_v1.py": "9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea",
    "scripts/providers/nfl_official_inactives_v1.py": "0d67316b6d7b7b9aa9d3637a07da4cd2b171639e",
    "scripts/validate_current_player_availability_timing_v1.py": "d67ae30ed6e837f62098671c499d05462fe9d837",
    "scripts/build/build_reconciled_active_roles_v1.py": "b2d05882cada048337f1f5f8b8db8ec7f9eef001",
}
PROTECTED_VERSIONS = {
    "te": "TE_R5P_PRODUCTION_MODEL_V1",
    "wr": "WR_R15_PRODUCTION_MODEL_V1",
    "r22": "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1",
    "r26": "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_V1",
}


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required evidence missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"required evidence has zero rows: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required evidence missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def b(v) -> bool:
    return bool(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-evidence", type=Path, required=True)
    ap.add_argument("--fixture-evidence", type=Path, required=True)
    args = ap.parse_args()

    static = read_json(args.static_evidence)
    fixtures = read_json(args.fixture_evidence)
    avail = read_csv(DATA / "current_player_availability.csv")
    active = read_csv(DATA / "roles_current_production_eligible_v1.csv")
    cert = read_csv(DATA / "current_player_availability_game_certification.csv")
    raw_roles = read_csv(DATA / "roles_ourlads.csv")
    form = read_csv(DATA / "player_form_consensus.csv")
    rb = read_csv(DATA / "rb_rush_synthesis_context.csv")
    candidate = read_json(DATA / "current_player_availability_candidate_summary.json")
    stack = read_json(DATA / "current_availability_football_stack_certification.json")
    ent = read_json(DATA / "target_entitlement_v1_audit.json")
    te = read_json(DATA / "te_r5p_full_slate_entitlement_audit.json")
    wr = read_json(DATA / "wr_r15_full_slate_entitlement_audit.json")
    qb = read_json(DATA / "qb_c2_production_integration_audit.json")
    r22 = read_json(DATA / "rb_receiving_tail_production_audit.json")
    r26 = read_json(DATA / "rb_r26_receptions_production_audit.json")
    r22trace = read_csv(DATA / "rb_receiving_tail_production_trace.csv")
    r26trace = read_csv(DATA / "rb_r26_receptions_production_trace.csv")

    unavailable = avail.loc[pd.to_numeric(avail.definitive_unavailable, errors="coerce").fillna(0).eq(1)].copy()
    bad = set(zip(unavailable.team.map(canon_team).astype(str), unavailable.player_clean_key.astype(str)))
    active_keys = set(zip(active.team.map(canon_team).astype(str), active.player_clean_key.astype(str)))
    form_keys = set(zip(form.team.map(canon_team).astype(str), form.player_clean_key.astype(str)))

    # Timing certification is the frozen authoritative schedule/timestamp ledger for
    # this candidate. schedule_2026.csv provides exact weekly game membership while
    # the certification carries the parsed kickoff UTC used by T-75 logic.
    sched = read_csv(DATA / "schedules" / "schedule_2026.csv")
    cert_kickoff = pd.to_datetime(cert.kickoff_utc, errors="coerce", utc=True)
    cert_teams = set(cert.away_team.map(canon_team).astype(str)) | set(cert.home_team.map(canon_team).astype(str))
    raw_role_teams = set(raw_roles.team.map(canon_team).dropna().astype(str))
    schedule_week = sched.loc[pd.to_numeric(sched.season, errors="coerce").eq(2026) & pd.to_numeric(sched.week, errors="coerce").eq(1)].copy()
    schedule_teams = set(schedule_week.away_team.map(canon_team).astype(str)) | set(schedule_week.home_team.map(canon_team).astype(str))

    gates: list[dict] = []
    def gate(n: int, name: str, passed: bool, evidence):
        gates.append({"gate": n, "name": name, "pass": bool(passed), "evidence": evidence})

    observed_core = {p: git_blob(Path(p)) for p in CORE_HASHES}
    gate(1, "locked availability core blobs", observed_core == CORE_HASHES, observed_core)
    gate(2, "protected trained model/artifact hashes unchanged", b(static.get("protected_model_artifacts_unchanged")), static)
    gate(3, "exact requested slate and parseable kickoffs", len(cert) == 16 and len(schedule_week) == 16 and cert_teams == schedule_teams == raw_role_teams and cert_kickoff.notna().all(), {"cert_games": len(cert), "schedule_games": len(schedule_week), "teams": len(cert_teams), "kickoff_parseable": bool(cert_kickoff.notna().all())})
    gate(4, "timestamped Ourlads covers all scheduled teams", raw_role_teams == cert_teams and "source_asof_utc" in avail.columns and avail.source_asof_utc.astype(str).str.strip().ne("").all(), {"roles_teams": len(raw_role_teams), "scheduled_teams": len(cert_teams)})
    prov_cols = [c for c in ["availability_authority", "final_availability_state", "source_asof_utc", "availability_generated_at_utc"] if c in active.columns]
    gate(5, "active-role provenance complete", len(prov_cols) == 4 and active[prov_cols].fillna("").astype(str).apply(lambda s: s.str.strip().ne("")).all().all(), {"provenance_columns": prov_cols, "rows": len(active)})
    gate(6, "no definitive unavailable in active roles", not bool(bad & active_keys), sorted(bad & active_keys))
    gate(7, "unavailable rows retained in audit artifact", len(unavailable) == int(candidate.get("definitive_unavailable", -1)) and len(unavailable) > 0, {"unavailable_rows": len(unavailable)})
    gate(8, "no duplicate active identity", not active.duplicated(["team", "player_clean_key"]).any(), {"rows": len(active)})
    qb_active = active.loc[active.position_group.astype(str).str.upper().eq("QB")].copy()
    qb1 = qb_active.role.astype(str).eq("QB1")
    per_qb1 = qb_active.assign(_qb1=qb1.astype(int)).groupby("team")._qb1.sum()
    gate(9, "QB1 uniqueness/completeness", bool((per_qb1 == 1).all()) and set(per_qb1.index) == set(active.team.unique()), {"teams": int(len(per_qb1)), "min_qb1": int(per_qb1.min()), "max_qb1": int(per_qb1.max())})
    ordinal_ok = True
    ordinal_bad = []
    for (team, grp), g in active.loc[active.position_group.astype(str).str.upper().isin(["QB", "RB", "FB", "TE"])].groupby(["team", "position_group"]):
        prefix = "RB" if str(grp).upper() in {"RB", "FB"} else str(grp).upper()
        ranks = sorted(int(x[len(prefix):]) for x in g.role.astype(str) if x.startswith(prefix) and x[len(prefix):].isdigit())
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            ordinal_ok = False; ordinal_bad.append({"team": team, "group": grp, "ranks": ranks})
    gate(10, "ordinal roles gap-free", ordinal_ok, ordinal_bad[:20])
    uncertain = avail.final_availability_state.astype(str).eq("UNCERTAIN")
    uncertain_keys = set(zip(avail.loc[uncertain, "team"].map(canon_team).astype(str), avail.loc[uncertain, "player_clean_key"].astype(str)))
    gate(11, "QUESTIONABLE/DOUBTFUL retained absent stronger evidence", uncertain_keys <= active_keys | {(t,k) for t,k in uncertain_keys if t not in set(active.team)}, {"uncertain": len(uncertain_keys)})
    complete = pd.to_numeric(avail.official_inactive_section_complete, errors="coerce").fillna(0).eq(1)
    gate(12, "official inactive absence only from complete sections", not ((~complete) & avail.final_availability_state.astype(str).eq("AVAILABLE_OFFICIAL_ACTIVE")).any(), {"complete_rows": int(complete.sum())})
    states = cert.certification_state.astype(str)
    elig = pd.to_numeric(cert.production_eligible, errors="coerce").fillna(0).eq(1)
    timing_ok = bool((~elig.loc[states.eq("REQUIRED_MISSING_FAIL_CLOSED")]).all()) if states.eq("REQUIRED_MISSING_FAIL_CLOSED").any() else True
    timing_ok &= bool(elig.loc[states.eq("NOT_YET_REQUIRED")].all()) if states.eq("NOT_YET_REQUIRED").any() else True
    gate(13, "T-75 fail-closed and not-yet-required semantics", timing_ok, states.value_counts().to_dict())
    snap = pd.to_datetime(cert.get("official_snapshot_asof_utc"), errors="coerce", utc=True)
    valid_snap = snap.notna()
    gate(14, "official snapshots strictly pre-kickoff", bool((snap.loc[valid_snap] < cert_kickoff.loc[valid_snap]).all()) if valid_snap.any() else True, {"snapshot_rows": int(valid_snap.sum())})
    gate(15, "PlayerForm excludes unavailable active opportunity", not bool(bad & form_keys), sorted(bad & form_keys))
    rb_keys = set(zip(rb.team.map(canon_team).astype(str), rb.player_clean_key.astype(str)))
    bad_rb = bad & rb_keys
    gate(16, "RB P3 no unavailable positive opportunity", not bad_rb, sorted(bad_rb))
    r26_keys = set(zip(r26trace.team.map(canon_team).astype(str), r26trace.player_clean_key.astype(str)))
    r26_bad = bad & r26_keys
    gate(17, "R26 no unavailable target/reception value", not r26_bad, sorted(r26_bad))
    r22_keys = set(zip(r22trace.team.map(canon_team).astype(str), r22trace.player_clean_key.astype(str)))
    r22_bad = bad & r22_keys
    gate(18, "R22 no unavailable receiving distribution", not r22_bad and stack.get("unavailable_players_with_simulation_arrays") == 0, {"trace_bad": sorted(r22_bad), "sim_bad": stack.get("unavailable_players_with_simulation_arrays")})
    gate(19, "QB production no unavailable starter/pass opportunity", not bool(bad & set(zip(qb_active.team.map(canon_team).astype(str), qb_active.player_clean_key.astype(str)))) and str(qb.get("disposition", "")).endswith("PASS"), {"qb_disposition": qb.get("disposition")})
    ent_trace = read_csv(DATA / "target_entitlement_v1_trace.csv")
    ent_keys = set(zip(ent_trace.team.map(canon_team).astype(str), ent_trace.player_clean_key.astype(str)))
    gate(20, "WR/TE entitlement excludes unavailable", not bool(bad & ent_keys), sorted(bad & ent_keys))
    conservation = all([
        bool(te.get("team_te_pool_preserved")), bool(te.get("non_te_entitlement_preserved")),
        bool(wr.get("m38_wr1_anchor_preserved")), bool(wr.get("wr2plus_pool_preserved")),
        bool(wr.get("wr_room_mass_preserved")), bool(wr.get("non_wr_entitlement_preserved")),
        bool(wr.get("team_total_player_entitlement_preserved")),
    ])
    gate(21, "M38/TE-R5P/WR-R15 conservation invariants", conservation, {"te": te, "wr": wr})
    gate(22, "team target entitlement conserved without injury percentage", conservation and int(stack.get("sportsbook_inputs_used", 1)) == 0, {"conservation": conservation, "sportsbook_inputs": stack.get("sportsbook_inputs_used")})
    versions = {"te": te.get("model_version"), "wr": wr.get("model_version"), "r22": r22.get("candidate"), "r26": r26.get("candidate")}
    versions_ok = versions == PROTECTED_VERSIONS
    gate(23, "promoted model versions protected", versions_ok, versions)
    sportsbook_zero = int(candidate.get("sportsbook_inputs_used", 1)) == 0 and int(stack.get("sportsbook_inputs_used", 1)) == 0 and not bool(ent.get("sportsbook_inputs_used", True))
    gate(24, "sportsbook inputs to availability/role/opportunity zero", sportsbook_zero, {"candidate": candidate.get("sportsbook_inputs_used"), "stack": stack.get("sportsbook_inputs_used"), "entitlement": ent.get("sportsbook_inputs_used")})
    gate(25, "downstream odds cannot resurrect ineligible players", b(static.get("downstream_universe_subset_guard_unchanged")), static.get("downstream_universe_subset_guard_unchanged"))
    candidate_ok = candidate.get("disposition") == "CURRENT_PLAYER_AVAILABILITY_NO_ODDS_FULL_SLATE_CANDIDATE_COMPLETED" and int(candidate.get("eligible_games", -1)) == 15 and int(candidate.get("withheld_games", -1)) == 1
    gate(26, "candidate no-odds Full Slate success", candidate_ok, candidate)
    gate(27, "static production-readiness availability wiring passes", b(static.get("availability_static_readiness_pass")), static)
    gate(28, "fixture RB1 OUT successor/current opportunity", b(fixtures.get("rb1_out_pass")), fixtures.get("rb1_out"))
    gate(29, "fixture QB1 inactive successor", b(fixtures.get("qb1_inactive_pass")), fixtures.get("qb1_inactive"))
    gate(30, "fixture WR/TE unavailable with conservation", b(fixtures.get("wr_te_unavailable_pass")), fixtures.get("wr_te_unavailable"))
    counts_ok = all(k in candidate for k in ["definitive_unavailable", "uncertain", "unknown", "withheld_games"])
    gate(31, "explicit real-source availability counts", counts_ok, {k: candidate.get(k) for k in ["definitive_unavailable", "uncertain", "unknown", "withheld_games"]})
    gate(32, "historical research unchanged", b(static.get("historical_research_unchanged")), static.get("historical_research_unchanged"))
    gate(33, "no unintended scientific model diff", b(static.get("scientific_model_diff_clean")), static.get("scientific_model_diff_clean"))
    gate(34, "live pricing only on eligible games if executed", b(static.get("live_pricing_not_executed")) or b(static.get("live_pricing_eligible_only")), {"not_executed": static.get("live_pricing_not_executed"), "eligible_only": static.get("live_pricing_eligible_only")})

    failures = [g for g in gates if not g["pass"]]
    payload = {
        "status": "PRELIMINARY_GATES_1_34_COMPLETE",
        "frozen_gate_count": 35,
        "evaluated_gate_count": 34,
        "passed_1_34": int(sum(g["pass"] for g in gates)),
        "failed_1_34": int(len(failures)),
        "all_1_34_pass": not failures,
        "gate35_pending_post_upload_lineage": True,
        "gates": gates,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"availability certification gates 1-34 failed: {[g['gate'] for g in failures]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
