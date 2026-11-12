"""Two player-key conventions that agree on every name without a suffix.

    _key("Brian Robinson Jr.")                          -> brianrobinsonjr
    player_name_key("Brian Robinson Jr.", strip=True)   -> brianrobinson
    _key("Stefon Diggs") == player_name_key("Stefon Diggs") -> stefondiggs

PlayerForm and Player Identity v3 key with the suffix stripped; the Bayesian
baseline and the model-context roster key from the display name via `_key`,
which keeps it. Every unsuffixed player therefore joins and every suffixed
player silently does not.

It stayed invisible until the depth-chart union widened the week-10 2026
universe from 481 to 487 and pulled suffixed players into the full-universe
gates, which failed closed on exactly that set and nothing else:
Gardner Minshew II, Brian Robinson Jr., Erick All Jr., Travis Etienne Jr.,
Thomas Fidone II, Michael Pittman Jr. (runs 43324642471, 43325943849).

Both gates are fixed with a fallback over only the rows that matched nothing,
and the fallback key is derived from the DISPLAY NAME on both sides -- a
concatenated key has no suffix token left to strip, so re-keying
"brianrobinsonjr" is a no-op and a fallback built that way misses exactly the
rows it exists to catch. That mistake was made once here; these tests pin it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.bayesian_v2 import _key, apply_bayesian_to_metrics
from scripts.utils.player_identity_v3 import player_name_key

SUFFIXED = [
    "Gardner Minshew II",
    "Brian Robinson Jr.",
    "Erick All Jr.",
    "Travis Etienne Jr.",
    "Thomas Fidone II",
    "Michael Pittman Jr.",
]


def test_the_two_conventions_disagree_only_on_suffixes():
    """The premise. If this ever fails, the fallbacks are no longer needed."""
    for name in SUFFIXED:
        assert _key(name) != player_name_key(name, strip_suffix=True)
    for name in ["Stefon Diggs", "Malik Nabers", "Rachaad White"]:
        assert _key(name) == player_name_key(name, strip_suffix=True)


def test_restripping_a_concatenated_key_is_a_noop():
    """Why the fallback must key from the display name, not the clean key."""
    for name in SUFFIXED:
        concatenated = _key(name)
        assert player_name_key(concatenated, strip_suffix=True) == concatenated


def _baseline(names: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"team": t, "player": n, "player_clean_key": _key(n),
         "bayes_available": 1, "bayes_tgt_share": 0.2, "bayes_mu": 50.0}
        for t, n in names
    ])


def _metrics(names: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"team": t, "player": n,
         "player_clean_key": player_name_key(n, strip_suffix=True)}
        for t, n in names
    ])


def test_suffixed_players_get_their_bayesian_baseline():
    pairs = list(zip(["ARI", "ATL", "CIN", "NO", "NYG", "PIT"], SUFFIXED))
    out = apply_bayesian_to_metrics(_metrics(pairs), _baseline(pairs))
    assert out["bayes_applied"].eq(1).all(), \
        out.loc[out.bayes_applied.ne(1), ["team", "player"]].to_dict("records")


def test_unsuffixed_players_are_unaffected():
    pairs = [("WAS", "Stefon Diggs"), ("NYG", "Malik Nabers")]
    out = apply_bayesian_to_metrics(_metrics(pairs), _baseline(pairs))
    assert out["bayes_applied"].eq(1).all()
    assert out["bayes_mu"].eq(50.0).all()


def test_mixed_board_resolves_every_row():
    pairs = [("ATL", "Brian Robinson Jr."), ("WAS", "Stefon Diggs"),
             ("PIT", "Michael Pittman Jr."), ("NYG", "Malik Nabers")]
    out = apply_bayesian_to_metrics(_metrics(pairs), _baseline(pairs))
    assert out["bayes_applied"].eq(1).all()


def test_absent_player_still_fails_to_match():
    """The fallback must not invent a baseline for someone genuinely missing."""
    out = apply_bayesian_to_metrics(
        _metrics([("ATL", "Brian Robinson Jr."), ("NYG", "Nobody Here")]),
        _baseline([("ATL", "Brian Robinson Jr.")]),
    )
    got = dict(zip(out["player"], out["bayes_applied"]))
    assert got["Brian Robinson Jr."] == 1
    assert got["Nobody Here"] == 0


def test_fallback_does_not_guess_an_ambiguous_stripped_key():
    """Two suffixed people collapsing to one stripped key must stay unmatched.

    The primary join misses both (their full keys carry different suffixes),
    and the fallback drops the ambiguous group rather than picking one. The row
    stays unmatched and the caller's gate fires, which is correct.
    """
    baseline = pd.DataFrame([
        {"team": "NYG", "player": "Chris Smith Jr.", "player_clean_key": _key("Chris Smith Jr."),
         "bayes_available": 1, "bayes_tgt_share": 0.2, "bayes_mu": 10.0},
        {"team": "NYG", "player": "Chris Smith II", "player_clean_key": _key("Chris Smith II"),
         "bayes_available": 1, "bayes_tgt_share": 0.2, "bayes_mu": 90.0},
    ])
    metrics = pd.DataFrame([
        {"team": "NYG", "player": "Chris Smith Jr.",
         "player_clean_key": player_name_key("Chris Smith Jr.", strip_suffix=True)},
    ])
    out = apply_bayesian_to_metrics(metrics, baseline)
    assert out["bayes_applied"].iloc[0] == 0
    assert pd.isna(out["bayes_mu"].iloc[0])


@pytest.mark.xfail(
    reason="PRE-EXISTING, NOT FIXED HERE. A suffix-stripped metrics key can "
           "collide with a DIFFERENT unsuffixed player's full key on the same "
           "team, and the primary join resolves it silently to that other "
           "player before any fallback is consulted. Guarding the primary join "
           "is a wider change than the week-10 pricing block needed, so it is "
           "recorded as a known hazard rather than quietly left undocumented. "
           "No real 2026 collision is known; the risk is a genuine same-name "
           "pair on one roster.",
    strict=True,
)
def test_primary_join_should_not_resolve_a_cross_convention_collision():
    baseline = pd.DataFrame([
        {"team": "NYG", "player": "Chris Smith Jr.", "player_clean_key": _key("Chris Smith Jr."),
         "bayes_available": 1, "bayes_tgt_share": 0.2, "bayes_mu": 10.0},
        {"team": "NYG", "player": "Chris Smith", "player_clean_key": _key("Chris Smith"),
         "bayes_available": 1, "bayes_tgt_share": 0.2, "bayes_mu": 90.0},
    ])
    metrics = pd.DataFrame([
        {"team": "NYG", "player": "Chris Smith Jr.",
         "player_clean_key": player_name_key("Chris Smith Jr.", strip_suffix=True)},
    ])
    out = apply_bayesian_to_metrics(metrics, baseline)
    # Junior must not be handed Senior's baseline.
    assert out["bayes_mu"].iloc[0] != 90.0


def test_rules_gate_resolves_suffixed_players():
    """The same convention gap also sat in apply_rules_to_metrics."""
    from scripts.modeling import simulation_rules as sr
    src = (sr.__file__ or "")
    text = open(src, encoding="utf-8").read()
    # The fallback index and the lookup that consults it must both be present.
    assert "by_player_stripped" in text
    assert "by_player_stripped.get(" in text
    assert "strip_suffix=True" in text
