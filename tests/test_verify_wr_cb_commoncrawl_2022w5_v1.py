from scripts.research.verify_wr_cb_commoncrawl_2022w5_v1 import _sha1_b32
def test_sha1_known_payload():
    assert _sha1_b32(b"abc")=="VGMT4NSHA2AWVOR6EVYXQUGCNSONBWE5"
def test_no_outcome_loaders_in_source_verifier():
    import inspect
    import scripts.research.verify_wr_cb_commoncrawl_2022w5_v1 as s
    text=inspect.getsource(s)
    for banned in ("load_player_stats","weeks1_3_full_graded","actual_source"):
        assert banned not in text
    assert '"target_game_outcomes":False' in text
