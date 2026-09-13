from pathlib import Path

from scripts.operations.apply_current_availability_eligible_team_seam_v1 import (
    FULL,
    FULL_EVENT_ANCHOR,
    FULL_EVENT_NEW,
    FULL_IMPORT,
    FULL_IMPORT_ANCHOR,
    FULL_NEW,
    FULL_OLD,
    transform,
)


def test_seam_relaxes_canonical_game_count_to_a_floor_not_exact_match():
    """Regression guard for a real production incident (2026-09-13): once the
    team-coverage check (patched immediately above this one) allows the
    football simulation to legitimately cover more teams than are currently
    certified-eligible, the derived canonical-game count is a superset too --
    that must not be fatal. Only covering FEWER games than the eligible set
    requires is a real bug.
    """
    assert "observed_games < expected_games" in FULL_EVENT_NEW
    assert "observed_games != expected_games" not in FULL_EVENT_NEW


def test_seam_still_applies_cleanly_to_the_real_target_file(tmp_path, monkeypatch):
    """Exercise the actual transform against a scratch copy of the real
    target file, proving the frozen source anchors this seam depends on
    still exist and the resulting code is syntactically valid."""
    import scripts.operations.apply_current_availability_eligible_team_seam_v1 as seam_mod

    scratch = tmp_path / "run_pricing_with_full_roster_universe_v1.py"
    scratch.write_text(FULL.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(seam_mod, "FULL", scratch)

    text = transform(
        scratch,
        [
            (FULL_IMPORT_ANCHOR, FULL_IMPORT, "import"),
            (FULL_OLD, FULL_NEW, "team coverage guard"),
            (FULL_EVENT_ANCHOR, FULL_EVENT_NEW, "canonical game coverage guard"),
        ],
        write=True,
    )
    compile(text, str(scratch), "exec")
    assert "observed_games < expected_games" in scratch.read_text(encoding="utf-8")
