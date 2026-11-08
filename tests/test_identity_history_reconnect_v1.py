"""Guard the five name-variant overrides that reconnect prior-game history.

docs/research/PRISM_EXISTENCE_CEILING_IS_SYSTEM_WIDE_V1.md

Each of these players was being assigned `position_prior_only` Bayesian evidence -- i.e.
projected from a generic POSITION PRIOR rather than his own history -- purely because the
Ourlads spelling did not join the nflverse canonical name. That is the exact failure the
existing `Marvin Harrison -> Marvin Harrison Jr.` row was added for. If a row is removed,
that player silently loses his history again, so this pins them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.validate_manual_name_overrides_v1 import validate

# source spelling -> nflverse canonical, with the gsis_id the identity was verified against
RECONNECTED = {
    # Target is the ASCII form, NOT "Audric Estimé", and that is deliberate.
    # Four normalizers in this repo disagree on his acute e -- bayesian_v2._key
    # keeps it, repair_live_prop_identity._name_keys strips it,
    # player_identity_v3.player_name_key transliterates it, and
    # simulation_v2._player_key passes a stored key through unchanged -- so an
    # accented canonical target put four different keys into the artifacts and
    # 2026 Week 9 run 42487807719 failed with "full football universe missing
    # Bayesian context" for him after the odds were bought.
    #
    # History reconnection, which is what this file guards, is PRESERVED: the
    # companion row below maps the nflverse spelling to the same ASCII target,
    # so the roles side and the nflverse-logs side still canonicalize to one
    # name. test_accented_and_stripped_spellings_converge pins that.
    "Audric Estim": ("Audric Estime", "00-0039373"),
    "Deandre Carter": ("DeAndre Carter", "00-0031763"),
    "Drew Ogletree": ("Andrew Ogletree", "00-0037292"),
    "Hollywood Brown": ("Marquise Brown", "00-0035662"),
    "Ben Vansumeren": ("Ben VanSumeren", "00-0038489"),
}


@pytest.fixture(scope="module")
def overrides() -> pd.DataFrame:
    return pd.read_csv("data/manual_name_overrides.csv", encoding="utf-8")


def test_overrides_validate_and_round_trip():
    """The production loader calls this at run_player_form_v2_loader.py:320."""
    out = validate()
    assert out["round_trip_failures"] == 0
    assert out["rows"] >= len(RECONNECTED)


@pytest.mark.parametrize("source,expected", sorted(RECONNECTED.items()))
def test_history_reconnecting_override_is_present(overrides, source, expected):
    full, _gsis = expected
    row = overrides.loc[overrides["player_source_name"].astype(str).str.strip() == source]
    assert len(row) == 1, f"missing or duplicated override for {source!r}"
    assert row.iloc[0]["full_name"].strip() == full


@pytest.mark.parametrize("source", sorted(RECONNECTED))
def test_override_carries_its_verification_provenance(overrides, source):
    """A name override changes model identity, so it may not be unsourced."""
    row = overrides.loc[overrides["player_source_name"].astype(str).str.strip() == source].iloc[0]
    _full, gsis = RECONNECTED[source]
    assert gsis in str(row["verified_source"]), f"{source}: verified_source must cite its gsis_id"
    assert str(row["reason"]).strip()
    assert str(row["verified_date"]).strip()


def test_accented_and_stripped_spellings_converge():
    """Both spellings of Estime must canonicalize to one name, or history breaks.

    This is the substance the accented target used to provide. Reconnection
    needs the roles side and the nflverse side to agree -- it does NOT need the
    agreed name to carry the diacritic, and carrying it broke the pricing
    universe.
    """
    from scripts.utils.canonical_names import canonicalize_player_name

    targets = {canonicalize_player_name(v) for v in ("Audric Estim", "Audric Estim\u00e9", "Audric Estime")}
    assert targets == {"Audric Estime"}, f"spellings diverge: {sorted(targets)}"


def test_no_reconnect_target_carries_a_diacritic():
    """A canonical target must be ASCII, because the normalizers disagree otherwise."""
    for source, (full, _gsis) in RECONNECTED.items():
        assert full.isascii(), f"{source} -> {full!r} is not ASCII; see the Estime comment above"


def test_source_names_are_unique(overrides):
    norm = overrides["player_source_name"].astype(str).str.strip().str.lower()
    assert not norm.duplicated().any()
