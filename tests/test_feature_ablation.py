import pandas as pd

from scripts.backtest.run_feature_ablation import mask_team_features, summarize


def test_mask_team_features_only_removes_requested_family():
    base = pd.DataFrame([{
        "team": "BUF", "coverage_man_rate": 0.4, "coverage_zone_rate": 0.6,
        "light_box_rate": 0.2, "heavy_box_rate": 0.3,
        "pressure_rate_generated": 0.35, "proe": 0.04,
    }])
    out = mask_team_features(base, "no_coverage")
    assert pd.isna(out.loc[0, "coverage_man_rate"])
    assert pd.isna(out.loc[0, "coverage_zone_rate"])
    assert out.loc[0, "light_box_rate"] == 0.2
    assert out.loc[0, "pressure_rate_generated"] == 0.35
    assert out.loc[0, "proe"] == 0.04


def test_summary_positive_delta_means_feature_helped_full_model():
    rows = []
    for variant, preds in [("full", [10.0, 20.0]), ("no_box", [13.0, 24.0])]:
        for actual, proj in zip([10.0, 20.0], preds):
            rows.append({"variant": variant, "market": "rush_yards", "actual": actual, "mc_proj": proj})
    out = summarize(pd.DataFrame(rows)).set_index("variant")
    assert out.loc["full", "mae"] == 0.0
    assert out.loc["no_box", "delta_mae_vs_full"] > 0
    assert out.loc["no_box", "feature_effect"] == "helps_full_model"
