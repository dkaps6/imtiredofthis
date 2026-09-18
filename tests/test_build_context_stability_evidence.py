import importlib.util
from pathlib import Path
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/research/build_context_stability_evidence.py"
spec = importlib.util.spec_from_file_location("context_stability", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_adjacent_period_shift_is_entity_and_season_scoped():
    df = pd.DataFrame({
        "player_id": ["a","a","a","a","b","b"],
        "season": [2024,2024,2025,2025,2024,2024],
        "week": [1,2,1,2,1,2],
        "signal": [1.0,2.0,100.0,101.0,4.0,8.0],
    }).sort_values(["player_id","season","week"])
    x = df["signal"]
    prior = x.groupby([df["player_id"], df["season"]], sort=False).shift(1)
    # No carry across season boundary: only a-2024 W2, a-2025 W2, b-2024 W2 have priors.
    assert prior.notna().sum() == 3
    assert pd.isna(prior.iloc[2])


def test_duplicate_period_key_is_detectable():
    df = pd.DataFrame({
        "player_id": ["a","a"], "season": [2024,2024], "week": [1,1]
    })
    assert df.duplicated(["player_id","season","week"]).any()
