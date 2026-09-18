import pandas as pd
import pytest

from scripts.data_frontier.advanced_data_signal_exploration_common_v1 import (
    confidence_tier,
    future_geometry_validation,
    pair_metrics,
    split_persistence,
)


def test_confidence_tier_boundaries():
    s = pd.Series([4, 5, 9, 10, 24, 25])
    assert confidence_tier(s, 5, 10, 25).tolist() == [
        "ABSTAIN", "LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH"
    ]


def test_pair_metrics_perfect_rank_relationship():
    df = pd.DataFrame({"pred":[1.0,2.0,3.0],"actual":[1.5,2.5,3.5]})
    m = pair_metrics(df,"pred","actual")
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(0.5)
    assert m["bias"] == pytest.approx(-0.5)
    assert m["spearman"] == pytest.approx(1.0)


def test_split_persistence_requires_minimum_observations():
    rows=[]
    for week in [1,2,3,4]:
        for value in [1.0,1.1,1.2]:
            rows.append({"week":week,"id":"A","metric":value+week/100})
    df=pd.DataFrame(rows)
    summary,_=split_persistence(
        df,
        keys=["id"],
        metrics=["metric"],
        early_mask=df["week"].le(2),
        late_mask=df["week"].ge(3),
        min_obs_each=5,
    )
    assert int(summary.iloc[0]["paired_profiles"]) == 1


def test_future_geometry_validation_uses_history_target_week_not_same_game():
    raw=pd.DataFrame({
        "week":[2,2,3,3],
        "id":["A","A","A","A"],
        "actual_metric":[10.0,12.0,20.0,22.0],
    })
    hist=pd.DataFrame({
        "target_week":[2,3],
        "id":["A","A"],
        "hist_metric":[11.0,21.0],
        "n":[5,6],
    })
    summary,matched=future_geometry_validation(
        raw=raw,
        history=hist,
        target_keys=["id"],
        target_week_col="week",
        history_target_week_col="target_week",
        history_pred_actual_pairs=[("hist_metric","actual_metric")],
        history_count_col="n",
        tier_breaks=(5,10,25),
    )
    all_row=summary.loc[summary["confidence_tier"].eq("ALL")].iloc[0]
    assert int(all_row["n"]) == 2
    assert all_row["mae"] == pytest.approx(0.0)
    assert len(matched) == 2
