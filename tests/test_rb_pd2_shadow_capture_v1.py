"""Section 7 no-op proofs for the RB PD2 forward-shadow capture hook.

`docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md` section 7 requires
four proofs before any live shadow lock:

1. hook-disabled path remains byte-identical to current priced output;
2. hook-enabled vs hook-disabled `model_proj`, `model_sd`, fair probability,
   edge and decision fields are identical;
3. shadow output exists only in the separate research artifact;
4. flag OFF is the default production path.

Proofs 1-3 are established here by running the *real* `scripts.run_pricing_v2.price`
twice over one controlled fixture -- once with the flag unset, once with it on --
and comparing the full output frames, not a chosen subset of columns.  Only the
two heavyweight external context loaders are substituted (they read live slate
CSVs that do not exist at unit-test time); everything from `simulate()` through
`adjusted_outcomes`, the empirical probability, edge and decision assembly is
the production code path.

A structural pass additionally proves the production edit cannot alter pricing
even on inputs this fixture does not reach.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.modeling.bayesian_v2 as bayesian_v2
import scripts.modeling.simulation_rules as simulation_rules
import scripts.run_pricing_v2 as run_pricing_v2
from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.research import rb_pd2_shadow_capture_v1 as shadow

SEASON = 2026
WEEK = 2  # Deliberately not Week 1: keeps the promoted P3 route out of the fixture.

PLAYERS = [
    # player, key, team, opponent, position, market, line
    ("Alpha Back", "alphaback", "CHI", "CAR", "RB", "rush_yards", 62.5),
    ("Bravo Back", "bravoback", "CAR", "CHI", "RB", "rush_yards", 48.5),
    ("Charlie Wide", "charliewide", "CHI", "CAR", "WR", "rec_yards", 55.5),
]
EVENT_ID = "2026_02_CAR_CHI"


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------
# `materialize_pricing_offers_v1.py` expands the compact live layer to one row
# per book/line, so the fixture must too -- a one-row-per-player fixture cannot
# exercise the deduplication contract at all.
BOOKS = [("draftkings", "DraftKings", 0.0), ("fanduel", "FanDuel", -1.5), ("betmgm", "BetMGM", 1.5)]


def _metrics_frame() -> pd.DataFrame:
    rows = []
    for player, key, team, opp, pos, market, line in PLAYERS:
      for book, book_title, line_offset in BOOKS:
        rows.append({
            "season": SEASON,
            "week": WEEK,
            "event_id": EVENT_ID,
            "player": player,
            "player_clean_key": key,
            "team": team,
            "opponent": opp,
            "position": pos,
            "position_group": pos,
            "market": market,
            "line": line + line_offset,
            "over_odds": -110,
            "under_odds": -110,
            "book": book,
            "book_title": book_title,
            "commence_time": "2026-09-17T17:00:00Z",
            "tgt_share": 0.12 if pos == "RB" else 0.26,
            "rush_share": 0.58 if pos == "RB" else 0.0,
            "ypt": 6.4,
            "ypc": 4.4,
            "catch_rate": 0.72,
        })
    return pd.DataFrame(rows)


def _diagnostics_frame(prefix: str) -> pd.DataFrame:
    """ML / State diagnostics share one schema, differing only in column prefix."""
    targets = {
        "rush_yards": {"alphaback": 61.0, "bravoback": 47.0, "charliewide": np.nan},
        "rec_yards": {"alphaback": 9.0, "bravoback": 7.0, "charliewide": 56.0},
        "receptions": {"alphaback": 1.6, "bravoback": 1.2, "charliewide": 4.4},
        "rush_att": {"alphaback": 14.0, "bravoback": 11.0, "charliewide": np.nan},
        "rush_rec_yards": {"alphaback": 70.0, "bravoback": 54.0, "charliewide": np.nan},
        "pass_yards": {"alphaback": np.nan, "bravoback": np.nan, "charliewide": np.nan},
        "anytime_td": {"alphaback": 0.42, "bravoback": 0.31, "charliewide": 0.28},
    }
    rows = []
    for player, key, team, _opp, _pos, _market, _line in PLAYERS:
        row = {
            "team": team,
            "player_clean_key": key,
            f"{prefix}_available": 1,
            f"{prefix}_method": f"{prefix}_fixture",
            f"{prefix}_training_cutoff": f"{SEASON - 1}-W18",
        }
        for target, by_key in targets.items():
            row[f"{prefix}_{target}"] = by_key[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _bayes_baseline() -> pd.DataFrame:
    rows = []
    for _player, key, team, _opp, pos, _market, _line in PLAYERS:
        rows.append({
            "team": team,
            "player_clean_key": key,
            "bayes_available": 1,
            "bayes_evidence_state": "fixture",
            "bayes_tgt_share": 0.12 if pos == "RB" else 0.26,
            "bayes_rush_share": 0.58 if pos == "RB" else 0.0,
            "bayes_ypt": 6.4,
            "bayes_ypc": 4.4,
            "bayes_ypa": 7.1,
            "bayes_receptions_per_target": 0.72,
        })
    return pd.DataFrame(rows)


def _contexts():
    teams = {
        team: TeamContext(
            team=team,
            season=SEASON,
            success_rate_off=0.45,
            success_rate_def=0.44,
            neutral_pace=27.5,
            plays_est=63.0,
            proe=0.01,
            coverage_man_rate=0.30,
            coverage_zone_rate=0.70,
            def_pass_epa=0.02,
            def_rush_epa=-0.03,
        )
        for team in ("CHI", "CAR")
    }
    players = [
        PlayerContext(
            player=player,
            team=team,
            opponent=opp,
            season=SEASON,
            week=WEEK,
            position=pos,
            role=pos,
            game_id=EVENT_ID,
            features={
                "tgt_share": 0.12 if pos == "RB" else 0.26,
                "rush_share": 0.58 if pos == "RB" else 0.0,
                "ypt": 6.4,
                "ypc": 4.4,
                "ypa": 7.1,
                "catch_rate": 0.72,
            },
            offense=teams[team],
            defense=teams[opp],
        )
        for player, _key, team, opp, pos, _market, _line in PLAYERS
    ]
    return teams, players


@pytest.fixture()
def priced_fixture(tmp_path, monkeypatch):
    """Point production pricing at an isolated data/output tree."""
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    research = tmp_path / "research"
    data.mkdir()
    outputs.mkdir()

    _metrics_frame().to_csv(data / "metrics_ready.csv", index=False)
    _diagnostics_frame("ml").to_csv(data / "model_ml_diagnostics.csv", index=False)
    _diagnostics_frame("state").to_csv(data / "model_state_diagnostics.csv", index=False)

    monkeypatch.setattr(run_pricing_v2, "DATA", data)
    monkeypatch.setattr(run_pricing_v2, "OUTPUTS", outputs)
    monkeypatch.setattr(run_pricing_v2, "OUT", outputs / "props_priced_clean.csv")
    monkeypatch.setattr(run_pricing_v2, "RULE_INPUTS", data / "model_rule_simulation_inputs.csv")
    monkeypatch.setattr(run_pricing_v2, "ML_DIAGNOSTICS", data / "model_ml_diagnostics.csv")
    monkeypatch.setattr(run_pricing_v2, "STATE_DIAGNOSTICS", data / "model_state_diagnostics.csv")
    monkeypatch.setattr(run_pricing_v2, "WEATHER_PATH", data / "weather_week.csv")

    monkeypatch.setattr(bayesian_v2, "load_bayesian_baseline", lambda *a, **k: _bayes_baseline())
    monkeypatch.setattr(simulation_rules, "load_model_contexts", lambda *a, **k: _contexts())

    monkeypatch.delenv(shadow.FLAG, raising=False)
    shadow.reset()
    yield {"data": data, "outputs": outputs, "research": research}
    shadow.reset()


# --------------------------------------------------------------------------
# Proof 4 -- the default is OFF
# --------------------------------------------------------------------------
def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(shadow.FLAG, raising=False)
    assert shadow.capture_enabled() is False
    for value in ("", " ", "0", "false", "no", "off", "OFF"):
        monkeypatch.setenv(shadow.FLAG, value)
        assert shadow.capture_enabled() is False, value
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(shadow.FLAG, value)
        assert shadow.capture_enabled() is True, value


# --------------------------------------------------------------------------
# Proofs 1 and 2 -- real end-to-end pricing, flag OFF vs flag ON
# --------------------------------------------------------------------------
def test_flag_on_and_off_price_identically_end_to_end(priced_fixture, monkeypatch):
    monkeypatch.delenv(shadow.FLAG, raising=False)
    off = run_pricing_v2.price(SEASON)
    assert not off.empty
    assert set(off["market"]) == {"rush_yards", "rec_yards"}

    off_csv = (priced_fixture["outputs"] / "off.csv")
    off.to_csv(off_csv, index=False)

    monkeypatch.setenv(shadow.FLAG, "1")
    monkeypatch.setattr(shadow, "DEFAULT_ROOT", priced_fixture["research"])
    on = run_pricing_v2.price(SEASON)

    on_csv = (priced_fixture["outputs"] / "on.csv")
    on.to_csv(on_csv, index=False)

    # Proof 1/2 in its strongest form: the entire priced frame, every column,
    # byte-identical -- not merely the five fields section 7 enumerates.
    assert off_csv.read_bytes() == on_csv.read_bytes()
    pd.testing.assert_frame_equal(off, on)

    # ...and the enumerated fields explicitly, so a future column rename cannot
    # quietly hollow out the check above.
    for column in ("model_proj", "model_sd", "fair_prob", "edge", "decision"):
        if column in off.columns:
            pd.testing.assert_series_equal(off[column], on[column])


def _strip_hook(source: str) -> str:
    """Return the pricing module with every section 7 statement mechanically removed.

    The proof-1 baseline is derived from the *live* file rather than a frozen
    snapshot, so this check keeps testing the real production source as it
    evolves instead of silently comparing against stale history.
    """
    tree = ast.parse(source)
    removed = 0

    def is_hook(node) -> bool:
        if isinstance(node, ast.Assign):
            return any(isinstance(t, ast.Name) and t.id == "shadow" for t in node.targets)
        if isinstance(node, ast.If):
            dumped = ast.dump(node.test)
            return "'shadow'" in dumped or shadow.FLAG in dumped
        return False

    def strip(node) -> None:
        nonlocal removed
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            kept = [n for n in block if not is_hook(n)]
            removed += len(block) - len(kept)
            setattr(node, field, kept)
            for child in kept:
                strip(child)

    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "price":
            strip(fn)

    assert removed == 4, f"expected to strip exactly 4 hook statements, stripped {removed}"
    return ast.unparse(ast.fix_missing_locations(tree))


def _load_stripped_pricing_module(tmp_path, monkeypatch, fixture):
    path = tmp_path / "run_pricing_v2_nohook.py"
    path.write_text(_strip_hook(Path("scripts/run_pricing_v2.py").read_text()))
    spec = importlib.util.spec_from_file_location("run_pricing_v2_nohook", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pricing_v2_nohook"] = module
    spec.loader.exec_module(module)

    data, outputs = fixture["data"], fixture["outputs"]
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "OUTPUTS", outputs)
    monkeypatch.setattr(module, "OUT", outputs / "props_priced_clean.csv")
    monkeypatch.setattr(module, "RULE_INPUTS", data / "model_rule_simulation_inputs.csv")
    monkeypatch.setattr(module, "ML_DIAGNOSTICS", data / "model_ml_diagnostics.csv")
    monkeypatch.setattr(module, "STATE_DIAGNOSTICS", data / "model_state_diagnostics.csv")
    monkeypatch.setattr(module, "WEATHER_PATH", data / "weather_week.csv")
    return module


def test_hook_disabled_output_is_identical_to_the_hookless_module(tmp_path, monkeypatch, priced_fixture):
    """Proof 1 in its literal form: hook-free source vs hooked source, flag unset."""
    hookless = _load_stripped_pricing_module(tmp_path, monkeypatch, priced_fixture)

    monkeypatch.delenv(shadow.FLAG, raising=False)
    before = hookless.price(SEASON)
    after = run_pricing_v2.price(SEASON)

    before_csv = priced_fixture["outputs"] / "hookless.csv"
    after_csv = priced_fixture["outputs"] / "hooked_flag_off.csv"
    before.to_csv(before_csv, index=False)
    after.to_csv(after_csv, index=False)

    assert before_csv.read_bytes() == after_csv.read_bytes()
    pd.testing.assert_frame_equal(before, after)


# --------------------------------------------------------------------------
# Proof 3 -- shadow output exists only in the research artifact
# --------------------------------------------------------------------------
def _run_capture(priced_fixture, monkeypatch):
    monkeypatch.setenv(shadow.FLAG, "1")
    monkeypatch.setattr(shadow, "DEFAULT_ROOT", priced_fixture["research"])
    priced = run_pricing_v2.price(SEASON)
    run_pricing_v2.OUTPUTS.mkdir(parents=True, exist_ok=True)
    priced.to_csv(run_pricing_v2.OUT, index=False)

    sessions = sorted(priced_fixture["research"].iterdir())
    assert len(sessions) == 1, "one pricing invocation must produce exactly one session dir"
    records, receipt = shadow.load_session(sessions[0])
    return priced, sessions[0], records, receipt


def test_shadow_output_lands_only_in_the_research_artifact(priced_fixture, monkeypatch):
    _priced, session_dir, records, receipt = _run_capture(priced_fixture, monkeypatch)

    # Only the eligible RB rush_yards player-games, and every one of them.
    assert len(records) == 2
    assert {r["player_clean_key"] for r in records} == {"alphaback", "bravoback"}
    assert {r["market"] for r in records} == {"rush_yards"}
    assert all(r["season"] == SEASON and r["week"] == WEEK for r in records)
    assert all(r["baseline"]["draw_count"] > 0 for r in records)
    assert all(r["sportsbook_inputs_used_in_candidate"] is False for r in records)
    assert all(r["production_output_mutated"] is False for r in records)
    assert all(r["outcome_present_at_lock"] is False for r in records)
    assert receipt["valid"] is True

    # No sportsbook field may appear anywhere in the scientific artifact.
    forbidden = {"book", "book_title", "line", "vegas_line", "side", "over_odds",
                 "under_odds", "odds", "edge", "decision", "ev_roi"}
    for record in records:
        assert forbidden.isdisjoint(record.keys()), record.keys() & forbidden

    # No shadow field leaked into the production board.
    board = pd.read_csv(run_pricing_v2.OUT)
    leaked = [c for c in board.columns if "shadow" in c.lower() or "pd2" in c.lower()]
    assert leaked == []

    assert sorted(p.name for p in session_dir.iterdir()) == [
        "baseline_capture.jsonl", "baseline_draws.npz", "session_receipt.json",
    ]


# --------------------------------------------------------------------------
# Proof 5 -- book/line expansion collapses to one capture per football key
# --------------------------------------------------------------------------
def test_multi_book_rows_collapse_to_exactly_one_capture_per_football_key(priced_fixture, monkeypatch):
    _priced, _session_dir, records, receipt = _run_capture(priced_fixture, monkeypatch)

    # The fixture prices 3 books x 3 players = 9 rows, 6 of them eligible RB
    # rush_yards rows spanning 2 player-games.
    assert receipt["pricing_rows_seen"] == len(BOOKS) * 2
    assert receipt["duplicate_rows_collapsed"] == (len(BOOKS) - 1) * 2
    assert receipt["rows_written"] == 2
    assert len(records) == 2
    assert len({r["football_key"] for r in records}) == 2
    assert all(r["duplicate_pricing_rows"] == len(BOOKS) for r in records)

    # The football key itself must be sportsbook-independent.
    for record in records:
        for book, book_title, _offset in BOOKS:
            assert book not in record["football_key"]
            assert book_title not in record["football_key"]


def test_repeated_football_key_with_different_draws_is_an_integrity_failure(tmp_path):
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    row = pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR",
                     "player": "Alpha Back", "player_clean_key": "alphaback"})
    kw = dict(row=row, market="rush_yards", position="RB", season=SEASON, week=WEEK)

    assert shadow.capture(adjusted_outcomes=np.linspace(1.0, 100.0, 64), target_mean=50.0, mc_proj=49.0, **kw) is True
    # Identical repeat: collapsed, session stays valid.
    assert shadow.capture(adjusted_outcomes=np.linspace(1.0, 100.0, 64), target_mean=50.0, mc_proj=49.0, **kw) is False
    assert shadow.sentinels() == []
    # Same key, different draws: integrity failure.
    assert shadow.capture(adjusted_outcomes=np.linspace(2.0, 100.0, 64), target_mean=50.0, mc_proj=49.0, **kw) is False
    kinds = [s["kind"] for s in shadow.sentinels()]
    assert kinds == ["duplicate_football_key_mismatch"]
    assert shadow.finalize()["valid"] is False
    shadow.reset()


def test_repeated_football_key_with_different_target_mean_is_an_integrity_failure(tmp_path):
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    row = pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR",
                     "player": "Alpha Back", "player_clean_key": "alphaback"})
    draws = np.linspace(1.0, 100.0, 64)
    kw = dict(row=row, adjusted_outcomes=draws, market="rush_yards", position="RB", season=SEASON, week=WEEK)

    assert shadow.capture(target_mean=50.0, mc_proj=49.0, **kw) is True
    assert shadow.capture(target_mean=51.0, mc_proj=49.0, **kw) is False
    assert [s["kind"] for s in shadow.sentinels()] == ["duplicate_football_key_mismatch"]
    assert "target_mean" in shadow.sentinels()[0]["detail"]
    shadow.reset()


# --------------------------------------------------------------------------
# Proof: the exact array survives, losslessly
# --------------------------------------------------------------------------
def test_exact_draws_round_trip_losslessly_with_digest_verification(priced_fixture, monkeypatch):
    priced, session_dir, records, _receipt = _run_capture(priced_fixture, monkeypatch)

    for record in records:
        draws = shadow.load_draws(session_dir, record)
        assert draws.dtype == np.float64
        assert draws.size == record["baseline"]["draw_count"]
        # The persisted array must reproduce the recorded summary exactly, which
        # a digest alone could never establish.
        assert shadow._digest(draws) == record["baseline"]["draw_digest_sha256"]
        assert float(np.mean(draws)) == record["baseline"]["mean"]

        # ...and must be the array production actually priced from.
        row = priced.loc[
            priced["player_clean_key"].eq(record["player_clean_key"])
            & priced["market"].eq("rush_yards")
        ].iloc[0]
        assert float(np.mean(draws)) == pytest.approx(float(row["model_proj"]), rel=0, abs=1e-9)

        # Section 8 needs the draws elementwise; prove the transform can run.
        mu = float(np.mean(draws))
        candidate_raw = np.maximum(0.0, mu + 1.3 * (draws - mu))
        candidate = candidate_raw * (mu / float(np.mean(candidate_raw)))
        assert float(np.mean(candidate)) == pytest.approx(mu, abs=1e-8)
        assert candidate.size == draws.size


def test_load_draws_rejects_a_tampered_digest(priced_fixture, monkeypatch):
    _priced, session_dir, records, _receipt = _run_capture(priced_fixture, monkeypatch)
    tampered = dict(records[0])
    tampered["baseline"] = dict(tampered["baseline"], draw_digest_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        shadow.load_draws(session_dir, tampered)


# --------------------------------------------------------------------------
# Proof 6 -- a capture failure cannot touch the canonical priced output
# --------------------------------------------------------------------------
def test_capture_failure_leaves_production_untouched_and_invalidates_the_session(priced_fixture, monkeypatch):
    """The flag is ON during the authoritative pregame run, so this is the case
    that matters: research breakage must cost the study, never the board."""
    monkeypatch.delenv(shadow.FLAG, raising=False)
    clean = run_pricing_v2.price(SEASON)

    monkeypatch.setenv(shadow.FLAG, "1")
    monkeypatch.setattr(shadow, "DEFAULT_ROOT", priced_fixture["research"])

    def explode(_draws):
        raise ValueError("synthetic shadow failure")

    monkeypatch.setattr(shadow, "_summary", explode)

    broken = run_pricing_v2.price(SEASON)

    # Production is byte-identical despite every capture failing.
    pd.testing.assert_frame_equal(clean, broken)

    session_dir = sorted(priced_fixture["research"].iterdir())[0]
    receipt = json.loads((session_dir / "session_receipt.json").read_text())
    assert receipt["valid"] is False
    assert receipt["rows_written"] == 0
    # Both gates fire, and they are independent: every eligible row raised, and
    # the completeness check separately noticed every expected key is absent.
    kinds = [s["kind"] for s in receipt["sentinels"]]
    assert set(kinds) == {"capture_exception", "missing_expected_key"}
    assert kinds.count("capture_exception") == len(BOOKS) * 2
    assert kinds.count("missing_expected_key") == 1
    assert "synthetic shadow failure" in receipt["sentinels"][0]["detail"]
    assert len(receipt["missing_expected_keys"]) == 2

    with pytest.raises(RuntimeError, match="is invalid"):
        shadow.assert_session_valid(receipt)


def test_missing_expected_key_invalidates_the_session(priced_fixture, monkeypatch):
    monkeypatch.setenv(shadow.FLAG, "1")
    monkeypatch.setattr(shadow, "DEFAULT_ROOT", priced_fixture["research"])
    shadow.begin_session(season=SEASON, out_root=priced_fixture["research"])
    receipt = shadow.finalize(expected_football_keys={"2026|2|X|CHI|CAR|ghostback|rush_yards"})
    assert receipt["valid"] is False
    assert receipt["missing_expected_keys"] == ["2026|2|X|CHI|CAR|ghostback|rush_yards"]
    shadow.reset()


# --------------------------------------------------------------------------
# Completeness -- wired into the real finalizer, not just the API
# --------------------------------------------------------------------------
def test_expected_key_set_is_sportsbook_independent_and_matches_what_was_captured(priced_fixture, monkeypatch):
    _priced, _session_dir, records, receipt = _run_capture(priced_fixture, monkeypatch)

    metrics = pd.read_csv(priced_fixture["data"] / "metrics_ready.csv")
    expected = shadow.expected_keys(metrics, season=SEASON)

    # 3 books x 2 RBs in the frame collapse to 2 football keys.
    assert len(expected) == 2
    assert expected == {r["football_key"] for r in records}
    assert receipt["missing_expected_keys"] == []
    assert receipt["valid"] is True

    for book, book_title, _offset in BOOKS:
        assert not any(book in key or book_title in key for key in expected)


def test_a_row_skipped_before_the_hook_invalidates_the_real_pricing_session(priced_fixture, monkeypatch):
    """The failure class the completeness gate exists for.

    `run_pricing_v2.py:226-229` drops a row whose simulation lookup returned
    nothing, *before* the capture seam. That row produces neither a record nor a
    sentinel, so only an independently-derived expected set can notice it is
    gone. Reproduced here with the genuine mechanism rather than by stubbing
    `capture()`.
    """
    real_lookup = run_pricing_v2.lookup

    def lookup_missing_one(sims, row, raw_market):
        if str(row.get("player_clean_key")) == "bravoback" and str(raw_market) == "rush_yards":
            return None
        return real_lookup(sims, row, raw_market)

    monkeypatch.setattr(run_pricing_v2, "lookup", lookup_missing_one)

    # Flag OFF: production's own behaviour under the same lookup failure.
    monkeypatch.delenv(shadow.FLAG, raising=False)
    without_shadow = run_pricing_v2.price(SEASON)

    # Flag ON: identical production output, but the session must refuse to validate.
    monkeypatch.setenv(shadow.FLAG, "1")
    monkeypatch.setattr(shadow, "DEFAULT_ROOT", priced_fixture["research"])
    with_shadow = run_pricing_v2.price(SEASON)

    pd.testing.assert_frame_equal(without_shadow, with_shadow)
    assert "bravoback" not in set(
        with_shadow.loc[with_shadow["market"].eq("rush_yards"), "player_clean_key"]
    )
    assert "alphaback" in set(
        with_shadow.loc[with_shadow["market"].eq("rush_yards"), "player_clean_key"]
    )

    session_dir = sorted(priced_fixture["research"].iterdir())[0]
    records, receipt = shadow.load_session(session_dir)

    assert len(records) == 1
    assert records[0]["player_clean_key"] == "alphaback"
    assert len(receipt["missing_expected_keys"]) == 1
    assert "bravoback" in receipt["missing_expected_keys"][0]
    assert receipt["valid"] is False
    assert {s["kind"] for s in receipt["sentinels"]} == {"missing_expected_key"}

    with pytest.raises(RuntimeError, match="is invalid"):
        shadow.assert_session_valid(receipt)


# --------------------------------------------------------------------------
# Session isolation
# --------------------------------------------------------------------------
def test_two_sessions_in_one_interpreter_do_not_share_state(priced_fixture, monkeypatch):
    monkeypatch.setenv(shadow.FLAG, "1")
    monkeypatch.setattr(shadow, "DEFAULT_ROOT", priced_fixture["research"])

    run_pricing_v2.price(SEASON)
    run_pricing_v2.price(SEASON)

    sessions = sorted(priced_fixture["research"].iterdir())
    assert len(sessions) == 2
    ids = set()
    for session in sessions:
        records, receipt = shadow.load_session(session)
        # Each session holds its own two captures, not four.
        assert len(records) == 2
        assert receipt["rows_written"] == 2
        assert receipt["valid"] is True
        assert {r["session_id"] for r in records} == {receipt["session_id"]}
        ids.add(receipt["session_id"])
    assert len(ids) == 2


def test_a_crash_between_capture_and_finalize_cannot_leak_into_the_next_session(tmp_path):
    """begin_session replaces state unconditionally, which is what makes a
    half-finished invocation unable to contaminate the next one."""
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    row = pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR",
                     "player": "Ghost", "player_clean_key": "ghost"})
    assert shadow.capture(row=row, adjusted_outcomes=np.linspace(1.0, 9.0, 16), target_mean=5.0,
                          mc_proj=5.0, market="rush_yards", position="RB", season=SEASON, week=WEEK) is True
    assert shadow.pending() == 1

    # Pricing raises here; finalize never runs. The next invocation opens a session.
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    assert shadow.pending() == 0
    receipt = shadow.finalize()
    assert receipt["rows_written"] == 0
    assert receipt["valid"] is True
    shadow.reset()


def test_session_carries_run_provenance(priced_fixture, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "d" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "987654321")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_JOB", "full-slate")
    _priced, _session_dir, records, receipt = _run_capture(priced_fixture, monkeypatch)

    assert receipt["provenance"]["code_sha"] == "d" * 40
    assert receipt["provenance"]["workflow_run_id"] == "987654321"
    assert receipt["provenance"]["workflow_run_attempt"] == "2"
    assert receipt["provenance"]["workflow_job"] == "full-slate"

    # Per-record provenance needed for an immutable section 9 join.
    for record in records:
        assert record["session_id"] == receipt["session_id"]
        assert record["commence_time"] == "2026-09-17T17:00:00Z"
        assert np.isfinite(record["mc_proj"])
        assert record["captured_at_utc"].endswith("Z")
        assert record["baseline_lock_eligible"] is True


# --------------------------------------------------------------------------
# Structural proof -- the production edit cannot alter pricing on any input
# --------------------------------------------------------------------------
def _price_function() -> ast.FunctionDef:
    tree = ast.parse(Path("scripts/run_pricing_v2.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "price":
            return node
    raise AssertionError("price() not found in scripts/run_pricing_v2.py")


def test_guarded_hook_assigns_nothing_and_never_rebinds_the_draw_array():
    fn = _price_function()

    guards = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "shadow"
    ]
    assert len(guards) == 2, "expected exactly the capture guard and the flush guard"

    for guard in guards:
        for inner in ast.walk(guard):
            if isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                # `info` is the flush receipt; nothing else may be bound, and no
                # attribute/subscript target may be written at all.
                assert names <= {"info"}, f"guarded block assigns {names}"
                assert all(isinstance(t, ast.Name) for t in targets)

    # `adjusted_outcomes` is still bound exactly twice, both in the pre-existing
    # mean-alignment branch, and never inside a shadow guard.
    binds = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "adjusted_outcomes" for t in node.targets)
    ]
    assert len(binds) == 2
    guarded_binds = [n for guard in guards for n in ast.walk(guard) if n in binds]
    assert guarded_binds == []


def test_research_module_is_imported_only_under_the_flag():
    source = Path("scripts/run_pricing_v2.py").read_text()
    tree = ast.parse(source)

    # No module-level import of the research package.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", "") or ""
            assert "research" not in name
            assert all("research" not in a.name for a in node.names)

    fn = _price_function()
    imports = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts.research")
    ]
    assert len(imports) == 1

    # That import sits inside an `os.getenv(FLAG)` guard.
    enclosing = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If) and any(i in ast.walk(node) for i in imports)
    ]
    assert enclosing, "research import is not guarded"
    assert any(shadow.FLAG in ast.dump(node.test) for node in enclosing)


# --------------------------------------------------------------------------
# Capture-module invariants
# --------------------------------------------------------------------------
def test_capture_copies_and_never_mutates_or_aliases_the_source_array(tmp_path):
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    draws = np.linspace(10.0, 90.0, 256)
    original = draws.copy()
    row = pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR",
                     "player": "Alpha Back", "player_clean_key": "alphaback"})

    assert shadow.capture(
        row=row, adjusted_outcomes=draws, target_mean=float(draws.mean()), mc_proj=float(draws.mean()),
        market="rush_yards", position="RB", season=SEASON, week=WEEK,
    ) is True
    assert shadow.pending() == 1
    np.testing.assert_array_equal(draws, original)

    # Mutating the production array after capture must not change what was stored.
    key = next(iter(shadow._SESSION["records"]))
    before = shadow._SESSION["records"][key]["baseline"]["draw_digest_sha256"]
    stored = shadow._SESSION["arrays"][shadow._SESSION["records"][key]["array_key"]].copy()
    draws *= 2.0
    assert shadow._SESSION["records"][key]["baseline"]["draw_digest_sha256"] == before
    np.testing.assert_array_equal(shadow._SESSION["arrays"][shadow._SESSION["records"][key]["array_key"]], stored)
    shadow.reset()


@pytest.mark.parametrize("position,market", [
    ("WR", "rush_yards"), ("QB", "rush_yards"), ("TE", "rec_yards"),
    ("RB", "rec_yards"), ("RB", "rush_att"), ("RB", "rush_rec_yards"), ("RB", "anytime_td"),
])
def test_capture_refuses_out_of_scope_rows(position, market, tmp_path):
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    assert shadow.capture(
        row=pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR", "player_clean_key": "x"}),
        adjusted_outcomes=np.ones(8), target_mean=1.0, mc_proj=1.0,
        market=market, position=position, season=SEASON, week=WEEK,
    ) is False
    assert shadow.pending() == 0
    assert shadow.sentinels() == []
    shadow.reset()


@pytest.mark.parametrize("draws,kind", [
    (np.array([]), "bad_array_shape"),
    (np.array([1.0, np.nan]), "non_finite_draw"),
    (np.array([1.0, -2.0]), "negative_draw"),
])
def test_a_degenerate_array_records_a_sentinel_instead_of_raising(draws, kind, tmp_path):
    """Never raise: the flag is ON during the authoritative pregame pricing run."""
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    row = pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR", "player_clean_key": "x"})
    assert shadow.capture(row=row, adjusted_outcomes=draws, target_mean=1.0, mc_proj=1.0,
                          market="rush_yards", position="RB", season=SEASON, week=WEEK) is False
    assert [s["kind"] for s in shadow.sentinels()] == [kind]
    assert shadow.finalize()["valid"] is False
    shadow.reset()


def test_blank_identity_records_a_sentinel(tmp_path):
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    row = pd.Series({"event_id": "", "team": "CHI", "opponent": "CAR", "player_clean_key": "x"})
    assert shadow.capture(row=row, adjusted_outcomes=np.ones(8), target_mean=1.0, mc_proj=1.0,
                          market="rush_yards", position="RB", season=SEASON, week=WEEK) is False
    assert [s["kind"] for s in shadow.sentinels()] == ["blank_identity"]
    shadow.reset()


def test_non_finite_target_mean_is_recorded_but_marked_lock_ineligible(tmp_path):
    """A non-finite ensemble mean is a real production fallback, not a capture
    bug, so it must not invalidate the whole session -- but it must never become
    a section 9 lock either."""
    shadow.reset()
    shadow.begin_session(season=SEASON, out_root=tmp_path)
    row = pd.Series({"event_id": EVENT_ID, "team": "CHI", "opponent": "CAR",
                     "player": "Alpha Back", "player_clean_key": "alphaback"})
    assert shadow.capture(row=row, adjusted_outcomes=np.linspace(1.0, 9.0, 32), target_mean=float("nan"),
                          mc_proj=5.0, market="rush_yards", position="RB", season=SEASON, week=WEEK) is True
    receipt = shadow.finalize()
    assert receipt["valid"] is True
    assert receipt["rows_written"] == 1
    assert receipt["lock_eligible_rows"] == 0

    records, _ = shadow.load_session(Path(receipt["dir"]))
    assert records[0]["baseline_lock_eligible"] is False
    assert records[0]["baseline_lock_ineligible_reason"] == "non_finite_target_mean"
    # JSON must not carry a bare NaN.
    assert records[0]["target_mean"] is None
    shadow.reset()
