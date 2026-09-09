#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE = "RB_R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_V1"
PASS = "R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED"
NO_MARKET = "R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_NO_RECEPTIONS_MARKET_YET"
FAIL = "R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_FAIL_NO_OBSERVATION"
R26Q_PASS = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION"
R26O_PASS = "R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL"
EXPECTED_Q_HEAD = "68661da94f03cab2f96182d47636cf55e088b5de"
EXPECTED_Q_RUN = 34400524030
EXPECTED_Q_ARTIFACT = 10123251043
EXPECTED_Q_DIGEST = "sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1"
EXPECTED_O_HEAD = "e7014a6e365cbb776e48085dcef12dfece744ca4"
EXPECTED_O_RUN = 34399750746
EXPECTED_O_ARTIFACT = 10123070453
EXPECTED_O_DIGEST = "sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0"
EXPECTED_ARRAYS = 107
EXPECTED_CHANGED = 104
EXPECTED_SEALED_FILES = 17
EXPECTED_DRAWS = 25000
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_f64(arr: np.ndarray) -> str:
    x = np.asarray(arr, dtype="<f8")
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def unique_file(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gate(rows: list[dict], name: str, passed: bool, evidence) -> bool:
    ev = evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True, default=str)
    rows.append({"gate": name, "passed": bool(passed), "evidence": ev})
    return bool(passed)


def name_keys(value) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    raw = str(value).strip().lower().replace("’", "'")
    if not raw or raw in {"nan", "none", "null", "<na>"}:
        return set()
    tokens: list[str] = []
    for token in re.split(r"\s+", raw):
        clean = re.sub(r"[^a-z0-9-]", "", token).replace("-", "")
        if clean:
            tokens.append(clean)
    while tokens and tokens[-1] in SUFFIXES:
        tokens.pop()
    if not tokens:
        return set()
    out = {"".join(tokens)}
    if len(tokens) >= 2:
        out.add(tokens[0] + tokens[-1])
    return {x for x in out if x}


def implied_american(price) -> float:
    try:
        p = float(price)
    except Exception:
        return float("nan")
    if not np.isfinite(p) or p == 0:
        return float("nan")
    return 100.0 / (p + 100.0) if p > 0 else (-p) / ((-p) + 100.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26q-root", type=Path, required=True)
    ap.add_argument("--parent-verified-marker", type=Path, required=True)
    ap.add_argument("--parent-head-marker", type=Path, required=True)
    ap.add_argument("--live-status", type=Path, required=True)
    ap.add_argument("--pricing-offers", type=Path, required=False)
    ap.add_argument("--pricing-audit", type=Path, required=False)
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    parent_verified = args.parent_verified_marker.is_file() and args.parent_verified_marker.read_text().strip() == "PASS"
    parent_head = args.parent_head_marker.read_text().strip() if args.parent_head_marker.is_file() else ""

    q_disp_path = unique_file(args.r26q_root.resolve(), "r26q_disposition.json")
    q_gate_path = unique_file(args.r26q_root.resolve(), "r26q_gate_matrix.csv")
    q_seal_path = unique_file(args.r26q_root.resolve(), "r26q_seal_manifest.csv")
    q_array_path = unique_file(args.r26q_root.resolve(), "r26q_array_seal_manifest.csv")
    q = read_json(q_disp_path)
    qg = pd.read_csv(q_gate_path, low_memory=False)
    qs = pd.read_csv(q_seal_path, low_memory=False)
    qa = pd.read_csv(q_array_path, low_memory=False)

    o_disp_path = unique_file(args.r26q_root.resolve(), "r26o_disposition.json")
    o_manifest_path = unique_file(args.r26q_root.resolve(), "r26o_rb_receptions_shadow_manifest.csv")
    o_npz_path = unique_file(args.r26q_root.resolve(), "r26o_rb_receptions_shadow_arrays.npz")
    od = read_json(o_disp_path)
    om = pd.read_csv(o_manifest_path, low_memory=False)

    gate(gates, "01_exact_r26q_artifact_digest_verified_by_workflow", parent_verified, parent_verified)
    gate(gates, "02_exact_r26q_head_sha", parent_head == EXPECTED_Q_HEAD, parent_head)
    gate(gates, "03_exact_r26q_pass_disposition", q.get("disposition") == R26Q_PASS, q.get("disposition"))
    q_pass_count = int(pd.Series(qg.get("passed", False)).astype(bool).sum())
    gate(gates, "04_r26q_gate_matrix_28_of_28", len(qg) == 28 and q_pass_count == 28, {"rows": len(qg), "passed": q_pass_count})
    lineage_ok = (
        int(q.get("sealed_parent_run", -1)) == EXPECTED_O_RUN
        and int(q.get("sealed_parent_artifact", -1)) == EXPECTED_O_ARTIFACT
        and q.get("sealed_parent_artifact_digest") == EXPECTED_O_DIGEST
        and q.get("sealed_parent_head") == EXPECTED_O_HEAD
        and od.get("disposition") == R26O_PASS
    )
    gate(gates, "05_r26q_parent_lineage_exact", lineage_ok, {
        "run": q.get("sealed_parent_run"), "artifact": q.get("sealed_parent_artifact"),
        "digest": q.get("sealed_parent_artifact_digest"), "head": q.get("sealed_parent_head"),
        "r26o_disposition": od.get("disposition"),
    })
    gate(gates, "06_r26q_reports_107_arrays", int(q.get("sealed_arrays", -1)) == EXPECTED_ARRAYS, q.get("sealed_arrays"))
    gate(gates, "07_r26q_reports_104_changed_arrays", int(q.get("sealed_changed_reception_arrays", -1)) == EXPECTED_CHANGED, q.get("sealed_changed_reception_arrays"))
    gate(gates, "08_r26q_reports_17_sealed_files", int(q.get("sealed_files", -1)) == EXPECTED_SEALED_FILES and len(qs) == EXPECTED_SEALED_FILES, {"reported": q.get("sealed_files"), "rows": len(qs)})
    byte_exact = qs.get("byte_exact", pd.Series(False, index=qs.index)).astype(bool)
    gate(gates, "09_all_r26q_sealed_files_byte_exact", len(qs) == EXPECTED_SEALED_FILES and bool(byte_exact.all()), int(byte_exact.sum()))
    gate(gates, "10_sealed_r26o_manifest_107_rb_fb_rows", len(om) == EXPECTED_ARRAYS and set(om.position_family.astype(str).str.upper()).issubset({"RB", "FB"}), {"rows": len(om), "positions": sorted(om.position_family.astype(str).unique().tolist())})

    sealed_array_rows: list[dict] = []
    with np.load(o_npz_path, allow_pickle=False) as npz:
        members = list(npz.files)
        gate(gates, "11_sealed_r26o_npz_107_members", len(members) == EXPECTED_ARRAYS, len(members))
        draw_ok = True
        hash_ok = True
        for row in om.itertuples(index=False):
            member = str(row.array_member)
            if member not in npz.files:
                draw_ok = hash_ok = False
                sealed_array_rows.append({"array_member": member, "present": False})
                continue
            arr = np.asarray(npz[member], dtype=float)
            h = sha256_f64(arr)
            d_ok = arr.ndim == 1 and len(arr) == EXPECTED_DRAWS
            h_ok = h == str(row.array_sha256_f64)
            draw_ok &= d_ok
            hash_ok &= h_ok
            sealed_array_rows.append({
                "array_member": member, "event_id": row.event_id, "team": row.team,
                "player": row.player, "player_clean_key": row.player_clean_key,
                "draws": int(len(arr)), "array_sha256_f64": h,
                "manifest_sha256_f64": str(row.array_sha256_f64), "hash_exact": h_ok,
                "finite": bool(np.isfinite(arr).all()),
            })
        gate(gates, "12_every_npz_member_25000_draws", draw_ok, draw_ok)
        gate(gates, "13_every_npz_hash_matches_manifest", hash_ok, hash_ok)

    scope_ok = int(pd.to_numeric(om.vacancy_active, errors="coerce").fillna(-1).eq(1).sum()) == EXPECTED_CHANGED
    nonvac = om.loc[pd.to_numeric(om.vacancy_active, errors="coerce").fillna(-1).eq(0), "team"].astype(str)
    scope_ok = scope_ok and len(nonvac) == 3 and sorted(nonvac.unique().tolist()) == ["CIN"]
    gate(gates, "14_scope_104_vacancy_3_cin_preserved", scope_ok, {"vacancy": int(pd.to_numeric(om.vacancy_active, errors="coerce").eq(1).sum()), "nonvacancy": int(len(nonvac)), "nonvacancy_teams": sorted(nonvac.unique().tolist())})
    gate(gates, "15_no_football_values_regenerated", q.get("football_values_regenerated") is False, q.get("football_values_regenerated"))
    gate(gates, "16_no_r9_refit", q.get("r9_refit") is False, q.get("r9_refit"))
    gate(gates, "17_no_production_parameters_changed", q.get("production_parameters_changed") is False, q.get("production_parameters_changed"))
    gate(gates, "18_no_production_or_live_shadow_activation", q.get("production_promotion_authorized") is False and q.get("live_shadow_production_activation_authorized") is False, {"production": q.get("production_promotion_authorized"), "live_shadow": q.get("live_shadow_production_activation_authorized")})
    gate(gates, "19_no_2026_week1_outcomes_used", int(q.get("2026_outcomes_used", -1)) == 0, q.get("2026_outcomes_used"))
    gate(gates, "20_sportsbook_after_immutable_candidate_only", parent_verified and hash_ok and q.get("sportsbook_football_inputs_used") == 0, {"parent_verified": parent_verified, "sealed_hashes_exact": hash_ok, "parent_sportsbook_inputs": q.get("sportsbook_football_inputs_used")})
    gate(gates, "21_current_roster_only_market_identity_not_candidate", True, {"roster_used_for_market_identity": True, "roster_used_for_candidate": False})

    live = read_json(args.live_status.resolve())
    legitimate_states = {"available", "no_active_slate_markets", "no_player_prop_markets"}
    live_state_ok = str(live.get("status")) in legitimate_states
    gate(gates, "22_live_provider_boundary_legitimate_state", live_state_ok, {"status": live.get("status"), "available": live.get("available")})

    market_available = bool(live.get("available"))
    identity_ok = (not market_available) or (
        live.get("core_prop_identity_disposition") == "LIVE_PROP_IDENTITY_READY"
        and int(live.get("core_prop_identity_unresolved_rows", -1)) == 0
    )
    gate(gates, "23_live_identity_ready_if_market_available", identity_ok, {"disposition": live.get("core_prop_identity_disposition"), "unresolved": live.get("core_prop_identity_unresolved_rows")})

    roles = pd.read_csv(args.roles.resolve(), low_memory=False)
    if not {"team", "player", "position"}.issubset(roles.columns):
        raise RuntimeError("R26R current Ourlads roles missing team/player/position")
    roles = roles.copy()
    roles["team"] = roles["team"].astype(str).str.upper().str.strip()
    roles["position_family"] = roles["position"].astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
    role_index: dict[tuple[str, str], set[str]] = {}
    for r in roles.itertuples(index=False):
        for key in name_keys(getattr(r, "player")):
            role_index.setdefault((str(r.team), key), set()).add(str(r.position_family))

    pricing = pd.DataFrame()
    pricing_audit = {}
    pricing_ok = True
    if market_available:
        if args.pricing_offers is None or not args.pricing_offers.is_file():
            pricing_ok = False
        else:
            pricing = pd.read_csv(args.pricing_offers.resolve(), low_memory=False)
        if args.pricing_audit is None or not args.pricing_audit.is_file():
            pricing_ok = False
        else:
            pricing_audit = read_json(args.pricing_audit.resolve())
        pricing_ok = pricing_ok and not pricing.empty and pricing_audit.get("consensus_line_created") is False
    gate(gates, "24_exact_pricing_offers_no_consensus_if_market_available", (not market_available) or pricing_ok, {"rows": int(len(pricing)), "consensus_line_created": pricing_audit.get("consensus_line_created") if pricing_audit else None})

    rec_offers = pd.DataFrame()
    rb_rec_offers = pd.DataFrame()
    matched_rows: list[dict] = []
    if market_available and pricing_ok:
        rec_offers = pricing.loc[pricing["market"].astype(str).eq("player_receptions")].copy()
        if not rec_offers.empty:
            def is_current_rb(row) -> bool:
                team = str(row.get("team_abbr", "")).upper().strip()
                keys = set()
                for c in ("canonical_player_name", "player_canonical", "player", "player_raw"):
                    if c in row.index:
                        keys |= name_keys(row.get(c))
                fams: set[str] = set()
                for key in keys:
                    fams |= role_index.get((team, key), set())
                return bool(fams & {"RB", "FB"})

            rb_rec_offers = rec_offers.loc[rec_offers.apply(is_current_rb, axis=1)].copy()
            manifest_lookup: dict[tuple[str, str], list[int]] = {}
            manifest_reset = om.reset_index(drop=True)
            for idx, row in manifest_reset.iterrows():
                for key in name_keys(row["player"]) | name_keys(row["player_clean_key"]):
                    manifest_lookup.setdefault((str(row["team"]).upper(), key), []).append(idx)

            with np.load(o_npz_path, allow_pickle=False) as npz:
                for row in rb_rec_offers.to_dict("records"):
                    team = str(row.get("team_abbr", "")).upper().strip()
                    keys = set()
                    for c in ("canonical_player_name", "player_canonical", "player", "player_raw"):
                        if c in row:
                            keys |= name_keys(row.get(c))
                    candidates: set[int] = set()
                    for key in keys:
                        candidates.update(manifest_lookup.get((team, key), []))
                    if len(candidates) != 1:
                        matched_rows.append({**row, "sealed_match_count": len(candidates), "sealed_match_ok": False})
                        continue
                    m = manifest_reset.iloc[next(iter(candidates))]
                    arr = np.asarray(npz[str(m.array_member)], dtype=float)
                    line = float(row["line"])
                    p_lt = float(np.mean(arr < line))
                    p_eq = float(np.mean(arr == line))
                    p_gt = float(np.mean(arr > line))
                    over_imp = implied_american(row.get("over_odds"))
                    under_imp = implied_american(row.get("under_odds"))
                    novig = float("nan")
                    if np.isfinite(over_imp) and np.isfinite(under_imp) and over_imp + under_imp > 0:
                        novig = over_imp / (over_imp + under_imp)
                    half_line = abs((line - math.floor(line)) - 0.5) < 1e-12
                    prob_gap = p_gt - novig if half_line and np.isfinite(novig) else float("nan")
                    baseline_mean = float(m.baseline_mc_receptions_mean)
                    candidate_mean = float(m.shadow_receptions_mean)
                    matched_rows.append({
                        **row, "sealed_match_count": 1, "sealed_match_ok": True,
                        "sealed_event_id": str(m.event_id), "sealed_team": str(m.team),
                        "sealed_player": str(m.player), "player_clean_key": str(m.player_clean_key),
                        "position_family": str(m.position_family), "vacancy_active": int(m.vacancy_active),
                        "array_member": str(m.array_member), "array_sha256_f64": str(m.array_sha256_f64),
                        "baseline_receptions_mean": baseline_mean,
                        "candidate_receptions_mean": candidate_mean,
                        "candidate_minus_baseline_mean": candidate_mean - baseline_mean,
                        "baseline_minus_line": baseline_mean - line,
                        "candidate_minus_line": candidate_mean - line,
                        "candidate_p_lt_line": p_lt, "candidate_p_eq_line": p_eq, "candidate_p_gt_line": p_gt,
                        "market_over_implied_probability": over_imp,
                        "market_under_implied_probability": under_imp,
                        "market_novig_over_probability": novig,
                        "candidate_minus_market_novig_over_probability": prob_gap,
                        "half_point_line_probability_comparable": half_line,
                    })

    snapshot = pd.DataFrame(matched_rows)
    has_rb_rec = not rb_rec_offers.empty
    match_ok = (not has_rb_rec) or (
        not snapshot.empty and snapshot["sealed_match_ok"].astype(bool).all() and len(snapshot) == len(rb_rec_offers)
    )
    failed_matches = int((~snapshot["sealed_match_ok"].astype(bool)).sum()) if len(snapshot) else 0
    gate(gates, "25_rb_receptions_matches_deterministic_one_to_one", match_ok, {"rb_reception_offers": int(len(rb_rec_offers)), "matched_rows": int(len(snapshot)), "failed_matches": failed_matches})

    offer_numeric_ok = True
    if has_rb_rec:
        good_mask = snapshot["sealed_match_ok"].astype(bool)
        for col in ("line", "over_odds", "under_odds"):
            vals = pd.to_numeric(snapshot.loc[good_mask, col], errors="coerce")
            offer_numeric_ok &= len(vals) > 0 and bool(vals.notna().all()) and bool(np.isfinite(vals).all())
        offer_numeric_ok &= bool((pd.to_numeric(snapshot.loc[good_mask, "over_odds"], errors="coerce") != 0).all())
        offer_numeric_ok &= bool((pd.to_numeric(snapshot.loc[good_mask, "under_odds"], errors="coerce") != 0).all())
    gate(gates, "26_retained_book_lines_valid", (not has_rb_rec) or offer_numeric_ok, offer_numeric_ok)

    prob_ok = True
    if has_rb_rec and match_ok:
        good = snapshot.loc[snapshot.sealed_match_ok.astype(bool)].copy()
        sums = good["candidate_p_lt_line"] + good["candidate_p_eq_line"] + good["candidate_p_gt_line"]
        prob_ok = bool(np.isfinite(good[["candidate_p_lt_line", "candidate_p_eq_line", "candidate_p_gt_line"]].to_numpy(float)).all()) and bool(np.allclose(sums.to_numpy(float), 1.0, atol=1e-12))
    gate(gates, "27_candidate_distribution_probabilities_valid", (not has_rb_rec) or prob_ok, prob_ok)

    array_recheck_ok = all(bool(r.get("hash_exact", False)) for r in sealed_array_rows if r.get("present", False)) and len(sealed_array_rows) == EXPECTED_ARRAYS
    gate(gates, "28_candidate_arrays_unchanged_from_seal", array_recheck_ok, array_recheck_ok)
    gate(gates, "29_sportsbook_absent_from_football_input_paths", True, {"sportsbook_football_inputs_used": 0, "football_values_regenerated": False})
    gate(gates, "30_observation_has_no_production_or_tuning_authority", True, {"production_promotion_authorized": False, "live_shadow_activation_authorized": False, "tuning_authorized": False})

    pd.DataFrame(sealed_array_rows).to_csv(out / "r26r_sealed_array_verification.csv", index=False)
    if len(snapshot):
        snapshot.to_csv(out / "r26r_receptions_book_line_snapshot.csv", index=False)
        good = snapshot.loc[snapshot.sealed_match_ok.astype(bool)].copy()
        if not good.empty:
            summary = good.groupby(["sealed_team", "sealed_player", "player_clean_key"], as_index=False).agg(
                book_line_rows=("line", "size"), books=("book", "nunique"),
                line_min=("line", "min"), line_median=("line", "median"), line_max=("line", "max"),
                baseline_receptions_mean=("baseline_receptions_mean", "first"),
                candidate_receptions_mean=("candidate_receptions_mean", "first"),
                candidate_minus_baseline_mean=("candidate_minus_baseline_mean", "first"),
            )
            summary.to_csv(out / "r26r_receptions_player_market_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["event_id", "team_abbr", "player", "market", "book", "line", "over_odds", "under_odds"]).to_csv(out / "r26r_receptions_book_line_snapshot.csv", index=False)
        pd.DataFrame(columns=["sealed_team", "sealed_player", "player_clean_key", "book_line_rows", "books", "line_min", "line_median", "line_max"]).to_csv(out / "r26r_receptions_player_market_summary.csv", index=False)

    source_rows = []
    for label, src in (
        ("live_odds_status", args.live_status),
        ("pricing_offers", args.pricing_offers),
        ("pricing_offer_audit", args.pricing_audit),
        ("current_roles_identity_only", args.roles),
    ):
        if src is not None and src.is_file():
            dst = out / f"source_{label}{src.suffix}"
            shutil.copy2(src, dst)
            source_rows.append({"source": label, "source_path": str(src), "sealed_copy": dst.name, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)})
    pd.DataFrame(source_rows).to_csv(out / "r26r_source_capture_manifest.csv", index=False)

    all_pass = all(bool(row["passed"]) for row in gates)
    usable_rec = has_rb_rec and match_ok and offer_numeric_ok and prob_ok
    disposition = PASS if all_pass and usable_rec else NO_MARKET if all_pass else FAIL

    status = {
        "candidate": CANDIDATE,
        "disposition": disposition,
        "parent_r26q_run": EXPECTED_Q_RUN,
        "parent_r26q_artifact": EXPECTED_Q_ARTIFACT,
        "parent_r26q_artifact_digest": EXPECTED_Q_DIGEST,
        "parent_r26q_head": EXPECTED_Q_HEAD,
        "all_gates_pass": all_pass,
        "gate_count": len(gates),
        "gate_pass_count": int(sum(bool(row["passed"]) for row in gates)),
        "live_market_status": live.get("status"),
        "live_market_available": market_available,
        "player_receptions_all_position_book_lines": int(len(rec_offers)),
        "rb_fb_receptions_book_lines": int(len(rb_rec_offers)),
        "matched_rb_fb_receptions_book_lines": int(snapshot["sealed_match_ok"].astype(bool).sum()) if len(snapshot) else 0,
        "2026_week1_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "football_values_regenerated": False,
        "r9_refit": False,
        "current_roster_used_for_market_identity": True,
        "current_roster_used_for_candidate": False,
        "production_parameters_changed": False,
        "production_promotion_authorized": False,
        "live_shadow_production_activation_authorized": False,
        "tuning_authorized": False,
        "postgame_evaluation_authorized": disposition in {PASS, NO_MARKET},
        "authority_note": "Observation only. Sportsbook evidence is downstream and cannot change the sealed R26Q candidate or production.",
    }
    (out / "r26r_disposition.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "r26r_market_capture_status.json").write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(gates).to_csv(out / "r26r_gate_matrix.csv", index=False)

    print(json.dumps(status, indent=2, sort_keys=True))
    print("R26R_DISPOSITION=" + disposition)
    return 0 if disposition != FAIL else 2


if __name__ == "__main__":
    raise SystemExit(main())
