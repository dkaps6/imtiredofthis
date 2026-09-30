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


def test_script_inventory_records_only_hashes_markers_and_json_shape():
    from bs4 import BeautifulSoup
    from scripts.research.verify_wr_cb_wayback_body_2024w1_v1 import script_structure_inventory
    soup = BeautifulSoup(
        '<script type="application/ld+json">'
        '{"@type":"Article","articleBody":"Wide Receiver vs Cornerback Matchup"}'
        '</script><script>window.__INITIAL_STATE__={"x":1}</script>',
        "html.parser",
    )
    inv = script_structure_inventory(soup)
    assert len(inv) == 2
    assert inv[0]["json_parseable"] is True
    assert "articlebody" in inv[0]["json_keys_of_interest"]
    assert "cornerback" in inv[0]["markers"]
    assert inv[0]["characters"] > 0 and len(inv[0]["sha256"]) == 64
    assert "Wide Receiver vs Cornerback Matchup" not in str(inv[0])
    assert "__initial_state__" in inv[1]["markers"]


def test_archived_article_body_inventory_hashes_structure_without_text():
    from bs4 import BeautifulSoup
    from scripts.research.verify_wr_cb_wayback_body_2024w1_v1 import (
        archived_article_body_inventory, _synthetic_page_from_archived_body,
    )
    body = (
        "<h2>Left WR vs Right CB</h2>"
        "<div>Wide Receiver</div><div>Team</div><div>DK / FD $</div>"
        "<div>Cornerback</div><div>Opp</div><div>Matchup</div>"
        "<div>Player One</div><div>IND</div><div>$1</div>"
        "<div>Corner One</div><div>TEN</div><div>Safe</div>"
    )
    soup = BeautifulSoup(
        '<script type="application/ld+json">'
        + __import__("json").dumps({"@type":"Article","articleBody":body})
        + "</script>",
        "html.parser",
    )
    bodies, inv = archived_article_body_inventory(soup)
    assert bodies == [body]
    assert inv[0]["contains_wr_cb_terms"] is True
    assert inv[0]["html_divs"] == 9
    assert len(inv[0]["sha256"]) == 64
    assert "Player One" not in str(inv[0])
    synthetic = _synthetic_page_from_archived_body(
        bodies[0], "2024-09-05T12:00:00Z"
    )
    assert 'article:published_time' in synthetic
