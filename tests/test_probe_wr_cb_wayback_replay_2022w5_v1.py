def test_probe_module_is_source_only():
    import inspect
    import scripts.research.probe_wr_cb_wayback_replay_2022w5_v1 as s
    text=inspect.getsource(s)
    assert "load_player_stats" not in text
    assert "parse_page(" not in text
    assert '"target_game_outcomes":False' in text
