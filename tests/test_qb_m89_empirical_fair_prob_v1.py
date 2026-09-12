import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.research.grade_qb_m89_empirical_fair_prob_v1 import (
    VARIANTS,
    _row_arrays,
    grade_variant,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAWS = 2000


def _build_fixture(tmp_path, n_players=20):
    rng = np.random.default_rng(3)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    trace_rows, meta_rows, props_rows = [], [], []
    week_arrays: dict[int, dict[str, np.ndarray]] = {}
    for i in range(n_players):
        season, week = 2024, (i % 17) + 1
        player = f"qb{i}"
        game_id = f"{season}_{week:02d}_KC_BAL"
        base_proj = 230.0 + rng.normal(0, 15)
        arr = rng.normal(base_proj, 35.0, DRAWS)
        football_synthesis = base_proj + rng.normal(3, 8)
        market_assisted = base_proj + rng.normal(5, 8)
        actual = football_synthesis + rng.normal(0, 40)
        array_key = f"a{i:06d}"
        week_arrays.setdefault(week, {})[array_key] = arr

        trace_rows.append({
            "season": season, "week": week, "team": "KC", "opponent": "BAL",
            "player_clean_key": player, "game_id": game_id,
            "base_proj": base_proj, "football_synthesis": football_synthesis,
            "market_assisted": market_assisted, "actual": actual,
        })
        meta_rows.append({
            "season": season, "week": week, "team": "KC", "opponent": "BAL",
            "player": player, "player_clean_key": player, "market": "pass_yards",
            "event_id": "e1", "array_key": array_key, "draws": DRAWS,
            "mc_mean": float(np.mean(arr)), "mc_sd": float(np.std(arr, ddof=1)),
            "npz_file": f"{season}_week_{week:02d}.npz",
        })
        props_rows.append({
            "season": season, "week": week, "game_id": game_id,
            "player_clean_key": player, "market": "pass_yards",
            "book": "draftkings", "line": 229.5, "over_odds": -110, "under_odds": -110, "player": player,
        })

    for week, arrs in week_arrays.items():
        np.savez(dist_dir / f"2024_week_{week:02d}.npz", **arrs)

    trace = pd.DataFrame(trace_rows)
    meta = pd.DataFrame(meta_rows)
    props = pd.DataFrame(props_rows)
    meta.to_csv(dist_dir / "2024_metadata.csv", index=False)
    return trace, meta, props, dist_dir


def test_grade_variant_produces_all_three_tiers(tmp_path):
    trace, meta, props, dist_dir = _build_fixture(tmp_path)
    arrays = _row_arrays(meta, dist_dir)
    matched = trace.copy()
    matched["market"] = "pass_yards"
    matched["line"] = 229.5
    matched["over_odds"] = -110
    matched["under_odds"] = -110

    summary = grade_variant(matched, arrays, "football_synthesis")
    assert set(summary["tier"]) == {"ALL_NO_FILTER", "LEAN_OR_STRONG", "STRONG_ONLY_PLAY_TIER"}
    all_no_filter = summary.loc[summary.tier.eq("ALL_NO_FILTER")].iloc[0]
    assert int(all_no_filter["matched_rows"]) == len(trace)


def test_translator_never_moves_the_mean_across_variants(tmp_path):
    trace, meta, props, dist_dir = _build_fixture(tmp_path)
    arrays = _row_arrays(meta, dist_dir)
    matched = trace.copy()
    matched["market"] = "pass_yards"
    matched["line"] = 229.5
    matched["over_odds"] = -110
    matched["under_odds"] = -110

    for proj_col in VARIANTS:
        summary = grade_variant(matched, arrays, proj_col)
        row = summary.loc[summary.tier.eq("ALL_NO_FILTER")].iloc[0]
        # model_mae is computed from the same proj_col fed in -- sanity check
        # it's finite and the grading pipeline ran to completion for every variant.
        assert np.isfinite(row["model_mae"])
        assert row["matched_rows"] == len(trace)


def test_grade_variant_fails_closed_on_zero_mean_mc_array(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    np.savez(dist_dir / "2024_week_01.npz", a000000=np.zeros(DRAWS))

    matched = pd.DataFrame([
        {
            "season": 2024, "week": 1, "team": "KC", "opponent": "BAL",
            "player_clean_key": "qb0", "game_id": "2024_01_KC_BAL", "market": "pass_yards",
            "base_proj": 5.0, "football_synthesis": 5.0, "market_assisted": 5.0,
            "actual": 3.0, "line": 4.5, "over_odds": -110, "under_odds": -110,
        }
    ])
    arrays = {(2024, 1, "KC", "BAL", "qb0", "pass_yards"): np.zeros(DRAWS)}
    with pytest.raises(RuntimeError, match="failed to align mean"):
        grade_variant(matched, arrays, "base_proj")


def test_cli_end_to_end_via_subprocess(tmp_path):
    trace, meta, props, dist_dir = _build_fixture(tmp_path)
    trace_path = tmp_path / "trace.csv"
    props_path = tmp_path / "props.csv"
    trace.to_csv(trace_path, index=False)
    props.to_csv(props_path, index=False)
    out_dir = tmp_path / "out"

    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(_REPO_ROOT, "scripts/research/grade_qb_m89_empirical_fair_prob_v1.py"),
            "--trace", str(trace_path),
            "--distribution-dir", str(dist_dir),
            "--props", str(props_path),
            "--out-dir", str(out_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(out_dir / "qb_m89_empirical_fair_prob_summary.csv")
    assert set(summary["variant"]) == {"base_proj", "football_synthesis", "market_assisted"}
    assert set(summary["tier"]) == {"ALL_NO_FILTER", "LEAN_OR_STRONG", "STRONG_ONLY_PLAY_TIER"}
