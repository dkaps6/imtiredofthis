import pandas as pd
from types import SimpleNamespace
from scripts.backtest.run_pass_tendency_recalibration import candidate_pass_share, rolling_dropback_baselines


def test_rolling_baseline_excludes_target_week():
    df=pd.DataFrame([
        {"season":2024,"week":18,"team":"AAA","dropback_rate":0.60},
        {"season":2025,"week":1,"team":"AAA","dropback_rate":0.50},
        {"season":2025,"week":2,"team":"AAA","dropback_rate":0.90},
    ])
    rates,league=rolling_dropback_baselines(df,2025,2,2024)
    assert abs(rates["AAA"]-0.55)<1e-9
    assert abs(league-0.55)<1e-9


def test_historical_shrinkage_sits_between_team_and_league():
    offense=SimpleNamespace(proe=0.0)
    defense=SimpleNamespace()
    # monkey-simple neutral success inputs expected by helpers
    offense.success_rate_off=0.5; defense.success_rate_def=0.5
    cfg={"mode":"historical","shrink":0.50,"proe_weight":0.0,"state":0.0}
    share,*_=candidate_pass_share(offense,defense,cfg,0.65,0.55)
    assert abs(share-0.60)<1e-9


def test_partial_proe_is_smaller_than_full_proe_effect():
    offense=SimpleNamespace(proe=0.08,success_rate_off=0.5)
    defense=SimpleNamespace(success_rate_def=0.5)
    base={"mode":"historical","shrink":0.50,"state":0.0}
    q,*_=candidate_pass_share(offense,defense,{**base,"proe_weight":0.25},0.55,0.55)
    h,*_=candidate_pass_share(offense,defense,{**base,"proe_weight":0.50},0.55,0.55)
    assert abs(q-0.57)<1e-9
    assert abs(h-0.59)<1e-9
