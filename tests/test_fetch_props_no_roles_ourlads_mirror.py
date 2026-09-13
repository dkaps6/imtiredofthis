from pathlib import Path

SOURCE = Path("scripts/fetch_props_oddsapi.py").read_text(encoding="utf-8")


def test_fetch_props_no_longer_mirrors_resolved_roles_onto_canonical_roles_ourlads():
    """Regression guard for a real production incident (2026-09-13): this CLI
    used to shutil.copyfile whatever roles source ROLES_CSV/--roles-csv
    resolved to (in production, the kickoff-timing-gated
    roles_current_production_eligible_v1.csv) onto data/roles_ourlads.csv --
    silently overwriting the documented-immutable raw Ourlads scrape that
    many other production scripts (PlayerForm, repair_live_prop_identity_v1,
    coverage_v2, ...) read directly. That crashed a live run with
    "Ourlads roster identity index missing current-event teams" once the
    kickoff wave withheld enough teams for the gated file to be small.
    """
    assert "shutil.copyfile" not in SOURCE
    assert "Mirrored roles CSV" not in SOURCE
    assert "\nimport shutil\n" not in SOURCE
