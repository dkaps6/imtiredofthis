from pathlib import Path


def test_full_slate_qb_c2_inline_audit_exempts_only_final_board_quarantine():
    text = Path(".github/workflows/full-slate.yml").read_text(encoding="utf-8")
    assert "QB C2 candidate lineage missing from pass-yard rows" not in text
    assert "FINAL_BOARD_QUARANTINE" in text
    assert "publishable_qb=qb.loc[~quarantine].copy()" in text
    assert "QB C2 candidate lineage missing from publishable pass-yard rows" in text
    assert "final-board-quarantined QB rows incorrectly claim C2 specialist consumption" in text
    assert "final-board-quarantined QB rows incorrectly carry C2 candidate lineage" in text
