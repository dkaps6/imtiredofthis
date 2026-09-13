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
    """The current football universe and its canonical games come from the same
    certified current-role authority. Extra games indicate universe widening and
    must fail closed rather than being accepted as a superset."""
    assert "observed_games != expected_games" in FULL_EVENT_NEW
    assert "observed_games < expected_games" not in FULL_EVENT_NEW


def test_seam_still_applies_cleanly_to_real_pricing_builder(tmp_path, monkeypatch):
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
    patched = scratch.read_text(encoding="utf-8")
    assert "validate_current_team_set" in patched
    assert "observed_games != expected_games" in patched
