"""Guards for the W7 production-parity replay harness.

Its job is to refuse to produce evidence it cannot earn, so that is what the tests pin:
the pre-flight must catch an incomplete fixture, and parity must be a gate rather than a
diagnostic that can be talked past.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import replay_w7_production_parity_v1 as R


def test_it_runs_the_real_production_entry_point_not_the_simulator():
    """The whole point of the audit finding: no direct simulation_v2.simulate."""
    src = Path("scripts/research/replay_w7_production_parity_v1.py").read_text(encoding="utf-8")
    assert "scripts.run_pricing_with_full_roster_universe_v3" in src
    assert "from scripts.simulation_v2 import" not in src
    assert "simulate(" not in src.split("def main")[0].replace("_full_roster_simulate", "")


def test_preflight_requires_the_ml_and_state_prediction_columns():
    req = R.REQUIRED_INPUTS
    assert "ml_pass_yards" in req["data/model_ml_diagnostics.csv"]
    assert "state_available" in req["data/model_state_diagnostics.csv"]
    assert "line" in req["outputs/props_raw.csv"]


def test_preflight_fails_closed_on_a_column_subset(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    for rel, cols in R.REQUIRED_INPUTS.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        # identity columns only -- exactly how the preserved fixture is stripped
        pd.DataFrame({"player_clean_key": ["x"], "team": ["KC"], "market": ["rec_yards"],
                      "player": ["x"]}).to_csv(p, index=False)
    out = R.check_inputs(tmp_path)
    assert out["usable"] is False
    assert "data/model_ml_diagnostics.csv" in out["column_incomplete"]
    assert "denylist" in out["reason"]


def test_preflight_passes_when_every_required_column_is_present(tmp_path):
    for rel, cols in R.REQUIRED_INPUTS.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({c: ["x"] for c in cols}).to_csv(p, index=False)
    assert R.check_inputs(tmp_path)["usable"] is True


def test_parity_is_a_gate_and_rejects_a_near_miss():
    key = R.PARITY_KEY
    base = pd.DataFrame({"player_clean_key": [f"p{i}" for i in range(100)],
                         "market": ["rec_yards"] * 100,
                         "model_proj": np.linspace(10, 90, 100),
                         "model_sd": np.linspace(5, 30, 100),
                         "fair_prob": np.linspace(0.1, 0.9, 100)})
    assert R.parity(base.copy(), base.copy())["PARITY_ESTABLISHED"] is True

    drifted = base.copy()
    drifted.loc[:4, "model_proj"] += 0.5      # 5% of rows off -> must fail
    out = R.parity(drifted, base.copy())
    assert out["PARITY_ESTABLISHED"] is False
    assert out["columns"]["model_proj"]["match_rate"] < 0.99


def test_parity_accounts_for_rows_it_could_not_reproduce():
    base = pd.DataFrame({"player_clean_key": [f"p{i}" for i in range(100)],
                         "market": ["rec_yards"] * 100,
                         "model_proj": np.arange(100.0), "model_sd": np.arange(100.0),
                         "fair_prob": np.linspace(0.1, 0.9, 100)})
    partial = base.iloc[:50].copy()           # half the board missing
    out = R.parity(partial, base)
    assert out["archived_not_replayed"] == 50
    assert out["PARITY_ESTABLISHED"] is False, "coverage shortfall must fail the gate"


def test_the_replay_never_fetches_odds():
    src = Path("scripts/research/replay_w7_production_parity_v1.py").read_text(encoding="utf-8")
    assert "fetch_props" not in src and "oddsapi" not in src.lower()
    assert "market_track_record/raw_props" in src, "paid lines must come from the archive"
