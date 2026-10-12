from scripts.research.audit_qb_team_pass_intent_common_crawl_archive_transport_v1i_r1 import (
    FRANCHISE_IDENTITY,
    _matches_identity,
)


def test_identity_vocab_has_32_teams():
    assert len(FRANCHISE_IDENTITY) == 32


def test_geographic_only_dallas_url_is_rejected():
    assert not _matches_identity(
        "https://www.dallasnews.com/arts-entertainment/define-dallas-fashion/",
        "DAL",
        "NYJ",
    )


def test_franchise_identity_url_is_accepted():
    assert _matches_identity(
        "https://www.dallasnews.com/sports/cowboys/2023/09/15/week-2-jets/",
        "DAL",
        "NYJ",
    )


def test_opponent_identity_url_is_accepted():
    assert _matches_identity(
        "https://publisher.test/nfl/jets-week-2-preview/",
        "DAL",
        "NYJ",
    )
