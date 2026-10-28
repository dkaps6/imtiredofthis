"""Pin down which module actually prices production, so the SIGMA error cannot recur.

Twice in this research line a defect was located in a module the production workflow
does not call: first `team_context_v3.py` (not on the target-share path), then
`pricing_v2.py`'s SIGMA constant (not on the pricing path). These tests assert the real
path, so a future claim about "what production does" has something to check itself
against.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(".github/workflows/full-slate.yml")


def test_workflow_prices_via_the_full_roster_universe_entry_point():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "run_pricing_with_full_roster_universe_v3.py" in text, (
        "the pricing entry point the workflow runs has changed; any claim about "
        "production pricing must be re-derived"
    )


def test_production_chain_reaches_run_pricing_v2_the_real_sd_writer():
    v1 = Path("scripts/run_pricing_with_full_roster_universe_v1.py").read_text(encoding="utf-8")
    assert "import scripts.run_pricing_v2 as pricing" in v1
    rp = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    assert 'model_sd = float(np.std(adjusted_outcomes, ddof=1))' in rp, (
        "model_sd is no longer the empirical SD of the player's own simulated outcomes"
    )


def test_production_prices_from_the_empirical_mc_distribution_not_a_normal():
    rp = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    assert "p_over = float(np.mean(adjusted_outcomes > line))" in rp, (
        "production no longer derives p_over from the empirical MC outcomes; a "
        "distributional assumption may have been introduced"
    )


def test_the_SIGMA_constant_is_NOT_on_the_production_path():
    """The specific thing I got wrong. If this ever fails, the claim changes."""
    rp = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    imports = re.findall(r"from scripts\.pricing_v2 import ([^\n]+)", rp)
    assert imports, "run_pricing_v2 no longer imports from pricing_v2 at all"
    imported = {n.strip() for line in imports for n in line.split(",")}
    assert "SIGMA" not in imported, "SIGMA is now imported into the production pricer"
    assert not re.search(r"\bSIGMA\b", rp), "SIGMA is now referenced in the production pricer"


def test_only_known_modules_import_the_constant_sigma_module():
    """A new importer of pricing_v2 could put the constant back on the path."""
    allowed = {
        "scripts/run_pricing_v2.py",                                  # takes odds helpers only
        "scripts/research/backtest_2026_live_calibration_v1.py",      # research, reads it to compare
    }
    importers = set()
    for p in Path("scripts").rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        if re.search(r"from scripts\.pricing_v2 import|import scripts\.pricing_v2", t):
            importers.add(p.as_posix())
    unexpected = importers - allowed
    assert not unexpected, f"new importer(s) of the constant-sigma module: {sorted(unexpected)}"


def test_live_boards_show_player_varying_dispersion():
    """The empirical check that refuted the claim; keep it as a regression."""
    import pandas as pd
    board = Path("data/market_track_record/boards/2026_wk07.csv")
    if not board.exists():
        pytest.skip("week 7 board not archived in this checkout")
    d = pd.read_csv(board, low_memory=False)
    for market in ("rec_yards", "rush_yards", "receptions"):
        sub = pd.to_numeric(d.loc[d["market"].eq(market), "model_sd"], errors="coerce").dropna()
        if len(sub) < 50:
            continue
        assert sub.nunique() > 10, f"{market} model_sd collapsed to near-constant"
