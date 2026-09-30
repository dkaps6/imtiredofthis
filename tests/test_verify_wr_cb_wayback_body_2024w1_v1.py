from scripts.research.verify_wr_cb_wayback_body_2024w1_v1 import (
    sha1_b32, pregame_row_timestamp_eligible,
)

def test_sha1_base32_known_payload_and_strict_game_clocks():
    assert sha1_b32(b"abc")=="VGMT4NSHA2AWVOR6EVYXQUGCNSONBWE5"
    assert not pregame_row_timestamp_eligible(
        "2024-09-05T12:00:00Z","2024-09-06T00:20:00Z")
    assert pregame_row_timestamp_eligible(
        "2024-09-05T12:00:00Z","2024-09-08T17:00:00Z")
    assert not pregame_row_timestamp_eligible("", "2024-09-08T17:00:00Z")
