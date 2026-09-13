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


def test_seam_requires_exact_canonical_game_count():
    """Once current availability certifies the football team set, the pricing
    simulation must cover exactly those teams' games.  Extra already-started or
    otherwise withheld games are not part of the current football universe."""
    assert "observed_games != expected_games" in FULL_EVENT_NEW
    assert "observed_games < expected_games" not in FULL_EVENT_NEW


def test_seam_still_applies_cleanly_to_the_real_target_file(tmp_path, monkeypatch):
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
    assert "observed_games != expected_games" in scratch.read_text(encoding="utf-8")
