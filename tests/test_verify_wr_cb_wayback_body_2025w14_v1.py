from scripts.research.verify_wr_cb_wayback_body_2025w14_v1 import sha1_b32
def test_sha1_known_payload():
    assert sha1_b32(b"abc")=="VGMT4NSHA2AWVOR6EVYXQUGCNSONBWE5"
def test_protected_script_contains_no_outcome_loader():
    import inspect
    import scripts.research.verify_wr_cb_wayback_body_2025w14_v1 as s
    src=inspect.getsource(s)
    banned=["load_player_stats","weeks1_3","graded","actual_source","sportsbook"]
    assert "target_game_outcomes\":False" in src
    for b in banned[:4]: assert b not in src


def test_scalar_publication_timestamp_guard_uses_pd_isna():
    import inspect
    import scripts.research.verify_wr_cb_wayback_body_2025w14_v1 as s
    src=inspect.getsource(s)
    assert "pub.notna()" not in src
    assert "pd.isna(pub)" in src
