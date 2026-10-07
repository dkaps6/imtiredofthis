import pandas as pd
import scripts.research.audit_player_individualization_v1 as m

def test_thresholds_are_mechanical():
    t,_=m.threshold_table()
    row=t.loc[t.metric.eq("ypt")].iloc[0]
    assert row.first_current_games_player_specific_gt_50pct==0
    assert row.first_current_games_current_alone_gt_group==6

def test_architecture_has_only_explicit_true_interaction():
    a=m.architecture_map()
    yes=set(a.loc[a.true_player_environment_interaction.eq("YES"),"component"])
    assert "historical_identity" in yes
    assert "injury_vacancy_redistribution" in yes
    assert "target_matchup_multiplier" not in yes
    assert "receiving_eff_matchup_multiplier" not in yes
