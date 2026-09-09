from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE = "RB_R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_V1"
PASS_DISPOSITION = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION"
FAIL_DISPOSITION = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_FAIL_NO_OBSERVATION"
EXPECTED_R26O_DISPOSITION = "R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL"
EXPECTED_ROWS = 468
EXPECTED_RB_FB = 107
EXPECTED_CHANGED = 104
EXPECTED_DRAWS = 25000
EXPECTED_SEED = 42
EXPECTED_GATES = 38
EXPECTED_RESULT_KEYS = 2892
EXPECTED_HEAD = "e7014a6e365cbb776e48085dcef12dfece744ca4"

EVIDENCE_REL = Path("data/backtests/r26o_2026_week1_receptions_shadow_integration_compatibility_v1")
KEY_FILES = [
    EVIDENCE_REL / "r26o_disposition.json",
    EVIDENCE_REL / "r26o_gate_matrix.csv",
    EVIDENCE_REL / "r26o_rb_receptions_shadow_manifest.csv",
    EVIDENCE_REL / "r26o_rb_receptions_shadow_arrays.npz",
    EVIDENCE_REL / "r26o_full_result_exactness_audit.csv",
    Path("r26o_identity_staging_audit.json"),
    Path("r26o_plan.sha256"),
    Path("r26o_lock.sha256"),
    Path("r26o_code.sha256"),
    Path("r26o_stage_helper.sha256"),
    Path("r26o_repair_note.sha256"),
    Path("r26o_second_repair_note.sha256"),
    Path("r26o_dtype_wrapper.sha256"),
    Path("r26o_gate15_repair_note.sha256"),
    Path("r26o_audit_payload_wrapper.sha256"),
    Path("r26o_qb_selector.sha256"),
    Path("r26o_protected_clean.txt"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array_f64(a: np.ndarray) -> str:
    x = np.asarray(a, dtype=np.float64)
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def gate(rows: list[dict], name: str, passed: bool, evidence) -> None:
    rows.append({"gate": name, "passed": bool(passed), "evidence": json.dumps(evidence, sort_keys=True) if isinstance(evidence, (dict, list)) else str(evidence)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26o-root", required=True)
    ap.add_argument("--parent-verified-marker", required=True)
    ap.add_argument("--parent-head-marker", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.r26o_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = root / EVIDENCE_REL

    for rel in KEY_FILES:
        p = root / rel
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"missing required R26O evidence file: {rel}")

    parent_verified = Path(args.parent_verified_marker).read_text(encoding="utf-8").strip() == "PASS"
    parent_head = Path(args.parent_head_marker).read_text(encoding="utf-8").strip()

    disp = json.loads((ev / "r26o_disposition.json").read_text(encoding="utf-8"))
    gates_df = pd.read_csv(ev / "r26o_gate_matrix.csv")
    manifest = pd.read_csv(ev / "r26o_rb_receptions_shadow_manifest.csv")
    exactness = pd.read_csv(ev / "r26o_full_result_exactness_audit.csv")
    npz = np.load(ev / "r26o_rb_receptions_shadow_arrays.npz", allow_pickle=False)

    array_rows = []
    array_valid = True
    array_hash_exact = True
    manifest_by_member = manifest.set_index("array_member", drop=False)
    for member in npz.files:
        arr = np.asarray(npz[member], dtype=np.float64)
        finite = bool(np.isfinite(arr).all())
        nonnegative = bool((arr >= 0).all())
        integer = bool(np.max(np.abs(arr - np.rint(arr))) <= 1e-12) if len(arr) else False
        draws_ok = len(arr) == EXPECTED_DRAWS
        h = sha256_array_f64(arr)
        in_manifest = member in manifest_by_member.index
        mh = str(manifest_by_member.loc[member, "array_sha256_f64"]) if in_manifest else ""
        hash_ok = in_manifest and h == mh
        array_valid &= finite and nonnegative and integer and draws_ok
        array_hash_exact &= hash_ok
        row = manifest_by_member.loc[member] if in_manifest else None
        array_rows.append({
            "array_member": member,
            "event_id": "" if row is None else str(row.event_id),
            "team": "" if row is None else str(row.team),
            "player": "" if row is None else str(row.player),
            "player_clean_key": "" if row is None else str(row.player_clean_key),
            "position_family": "" if row is None else str(row.position_family),
            "vacancy_active": -1 if row is None else int(row.vacancy_active),
            "draws": int(len(arr)),
            "array_sha256_f64": h,
            "manifest_sha256_f64": mh,
            "hash_exact": bool(hash_ok),
            "finite": finite,
            "nonnegative": nonnegative,
            "integer_valued": integer,
        })

    changed = exactness.loc[exactness.changed.astype(bool)]
    forbidden = exactness.loc[exactness.forbidden_change.astype(bool)]
    nonrec_changed = changed.loc[changed.market.ne("receptions")]
    changed_nonrbfb = changed.loc[~changed.position_family.isin(["RB", "FB"])]
    vacancy = manifest.loc[manifest.vacancy_active.eq(1)]
    nonvac = manifest.loc[manifest.vacancy_active.eq(0)]
    mean_changed = (manifest.shadow_receptions_mean - manifest.baseline_mc_receptions_mean).abs() > 0

    seal_files = []
    sealed_root = out / "sealed_r26o"
    for rel in KEY_FILES:
        src = root / rel
        dst = sealed_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        src_sha = sha256_file(src)
        dst_sha = sha256_file(dst)
        seal_files.append({
            "path": str(rel),
            "bytes": int(src.stat().st_size),
            "source_sha256": src_sha,
            "sealed_sha256": dst_sha,
            "byte_exact": src_sha == dst_sha and src.stat().st_size == dst.stat().st_size,
            "source": "immutable_r26o_artifact_10123070453",
        })

    file_seal_exact = all(r["byte_exact"] for r in seal_files)

    gates: list[dict] = []
    gate(gates, "01_exact_r26o_artifact_digest_verified_by_workflow", parent_verified, parent_verified)
    gate(gates, "02_exact_r26o_head_sha", parent_head == EXPECTED_HEAD, parent_head)
    gate(gates, "03_exact_r26o_pass_disposition", disp.get("disposition") == EXPECTED_R26O_DISPOSITION, disp.get("disposition"))
    gate(gates, "04_all_structural_gates_pass", disp.get("all_structural_gates_pass") is True, disp.get("all_structural_gates_pass"))
    gate(gates, "05_prospective_seal_design_authorized", disp.get("prospective_seal_design_authorized") is True, disp.get("prospective_seal_design_authorized"))
    gate(gates, "06_production_rows_exact", int(disp.get("production_rows", -1)) == EXPECTED_ROWS, disp.get("production_rows"))
    gate(gates, "07_rb_fb_rows_exact", int(disp.get("rb_fb_rows", -1)) == EXPECTED_RB_FB, disp.get("rb_fb_rows"))
    gate(gates, "08_r26n_changed_rb_fb_exact", int(disp.get("r26n_changed_rb_fb_rows", -1)) == EXPECTED_CHANGED, disp.get("r26n_changed_rb_fb_rows"))
    gate(gates, "09_shadow_changed_array_count_exact", int(disp.get("shadow_changed_array_count", -1)) == EXPECTED_CHANGED, disp.get("shadow_changed_array_count"))
    gate(gates, "10_iterations_seed_exact", int(disp.get("iterations", -1)) == EXPECTED_DRAWS and int(disp.get("seed", -1)) == EXPECTED_SEED, {"iterations": disp.get("iterations"), "seed": disp.get("seed")})
    gate(gates, "11_r26o_gate_matrix_38_of_38", len(gates_df) == EXPECTED_GATES and gates_df.passed.astype(bool).all(), {"rows": len(gates_df), "passed": int(gates_df.passed.astype(bool).sum())})
    gate(gates, "12_manifest_107_rb_fb_rows", len(manifest) == EXPECTED_RB_FB and set(manifest.position_family.astype(str)).issubset({"RB", "FB"}), {"rows": len(manifest), "positions": sorted(manifest.position_family.astype(str).unique())})
    gate(gates, "13_scope_104_vacancy_3_cin_preserved", len(vacancy) == EXPECTED_CHANGED and len(nonvac) == 3 and set(nonvac.team.astype(str)) == {"CIN"} and int(mean_changed.sum()) == EXPECTED_CHANGED and not mean_changed.loc[nonvac.index].any(), {"vacancy": len(vacancy), "nonvacancy": len(nonvac), "nonvacancy_teams": sorted(nonvac.team.astype(str).unique()), "mean_changed_rows": int(mean_changed.sum())})
    gate(gates, "14_manifest_draws_exact", manifest.draws.eq(EXPECTED_DRAWS).all(), sorted(manifest.draws.unique().tolist()))
    gate(gates, "15_npz_member_count_exact", len(npz.files) == EXPECTED_RB_FB and set(npz.files) == set(manifest.array_member.astype(str)), {"npz": len(npz.files), "manifest": len(manifest)})
    gate(gates, "16_npz_arrays_valid", array_valid, array_valid)
    gate(gates, "17_npz_array_hashes_match_manifest", array_hash_exact, array_hash_exact)
    gate(gates, "18_exactness_zero_forbidden_changes", len(forbidden) == 0, len(forbidden))
    gate(gates, "19_exactness_104_receptions_rb_fb_changes", len(changed) == EXPECTED_CHANGED and len(nonrec_changed) == 0 and len(changed_nonrbfb) == 0, {"changed": len(changed), "nonreception": len(nonrec_changed), "non_rb_fb": len(changed_nonrbfb)})
    gate(gates, "20_nonreception_stack_exact", not exactness.loc[exactness.market.ne("receptions"), "changed"].astype(bool).any() and len(exactness) == EXPECTED_RESULT_KEYS, {"result_keys": len(exactness)})
    gate(gates, "21_2026_outcomes_zero", int(disp.get("2026_outcomes_used", -1)) == 0, disp.get("2026_outcomes_used"))
    gate(gates, "22_sportsbook_football_inputs_zero", int(disp.get("sportsbook_football_inputs_used", -1)) == 0, disp.get("sportsbook_football_inputs_used"))
    gate(gates, "23_same_week_depth_false", disp.get("same_week_depth_used") is False, disp.get("same_week_depth_used"))
    gate(gates, "24_r9_refit_false", disp.get("r9_refit") is False, disp.get("r9_refit"))
    gate(gates, "25_production_parameters_unchanged", disp.get("production_parameters_changed") is False, disp.get("production_parameters_changed"))
    gate(gates, "26_production_promotion_false", disp.get("production_promotion_authorized") is False, disp.get("production_promotion_authorized"))
    gate(gates, "27_live_shadow_activation_false", disp.get("live_shadow_production_activation_authorized") is False, disp.get("live_shadow_production_activation_authorized"))
    gate(gates, "28_r26q_is_read_only_seal", file_seal_exact, {"sealed_files": len(seal_files), "all_byte_exact": file_seal_exact})

    gate_df = pd.DataFrame(gates)
    all_pass = bool(gate_df.passed.all())
    disposition = PASS_DISPOSITION if all_pass else FAIL_DISPOSITION

    pd.DataFrame(seal_files).to_csv(out / "r26q_seal_manifest.csv", index=False)
    pd.DataFrame(array_rows).to_csv(out / "r26q_array_seal_manifest.csv", index=False)
    gate_df.to_csv(out / "r26q_gate_matrix.csv", index=False)

    payload = {
        "candidate": CANDIDATE,
        "disposition": disposition,
        "all_seal_gates_pass": all_pass,
        "sealed_parent_run": 34399750746,
        "sealed_parent_artifact": 10123070453,
        "sealed_parent_head": EXPECTED_HEAD,
        "sealed_parent_artifact_digest": "sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0",
        "sealed_files": len(seal_files),
        "sealed_arrays": len(npz.files),
        "sealed_changed_reception_arrays": len(changed),
        "iterations": EXPECTED_DRAWS,
        "seed": EXPECTED_SEED,
        "2026_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "same_week_depth_used": False,
        "r9_refit": False,
        "football_values_regenerated": False,
        "production_parameters_changed": False,
        "live_shadow_production_activation_authorized": False,
        "production_promotion_authorized": False,
        "observation_design_authorized": bool(all_pass),
        "authority_note": "PASS authorizes only immutable prospective observation/postgame evaluation of the sealed R26O receptions shadow candidate. Production remains unchanged.",
    }
    (out / "r26q_disposition.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("R26Q_DISPOSITION=" + disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
