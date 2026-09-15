from pathlib import Path

from scripts.operations import apply_current_availability_eligible_team_seam_v1 as seam


def _transform_r26_source(tmp_path: Path) -> str:
    source = seam.R26.read_text(encoding="utf-8")
    path = tmp_path / "rb_r26_adapter.py"
    path.write_text(source, encoding="utf-8")
    return seam.transform(
        path,
        [
            (seam.R26_IMPORT_ANCHOR, seam.R26_IMPORT, "R26 helper import"),
            (seam.R26_OLD, seam.R26_NEW, "R26 team coverage guard"),
            (seam.R26_CONTROL_OLD, seam.R26_CONTROL_NEW, "R26 optional current-slate CIN entitlement control"),
            (seam.R26_TRACE_CONTROL_OLD, seam.R26_TRACE_CONTROL_NEW, "R26 optional current-slate CIN trace control"),
            (seam.R26_AUDIT_CONTROL_OLD, seam.R26_AUDIT_CONTROL_NEW, "R26 current-slate control audit"),
        ],
        write=False,
    )


def test_partial_current_slate_does_not_require_withheld_cin_control(tmp_path):
    transformed = _transform_r26_source(tmp_path)

    assert 'validate_current_team_set(' in transformed
    assert 'else float("inf")' not in transformed
    assert 'if cin_trace.empty or' not in transformed
    assert 'control_team_present = bool(cin.any())' in transformed
    assert 'if not cin_trace.empty and (' in transformed
    assert '"control_team_present_in_current_universe": bool(control_team_present)' in transformed

    # Frozen R26 science/assets remain untouched by the production-only seam.
    assert 'EXPECTED_VACANCY_TEAMS = {' in transformed
    assert 'CONTROL_TEAM = "CIN"' in transformed
    assert 'EXPECTED_MODEL_SHA256 = "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba"' in transformed
    assert 'EXPECTED_ROOM_STATE_SHA256 = "27ad7bad8fcdfa6b0b1090994e0c0d2bdc4c0e45209d2409abc9574b5d354258"' in transformed

    compile(transformed, str(tmp_path / "rb_r26_adapter.py"), "exec")


def test_control_still_fails_if_present_and_changed(tmp_path):
    transformed = _transform_r26_source(tmp_path)

    # When CIN is actually in the certified current universe, both original
    # invariance checks remain live; only absence is treated as not applicable.
    assert 'max_cin_entitlement_delta = float(rb.loc[cin, "entitlement_delta"].abs().max()) if control_team_present else 0.0' in transformed
    assert 'max_cin_entitlement_delta > 1e-12' in transformed
    assert 'cin_trace.rb_r26_receptions_applied.any()' in transformed
    assert 'cin_trace.final_minus_baseline_receptions_mean.abs().max() > 1e-12' in transformed
