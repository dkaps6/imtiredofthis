from scripts.research.audit_gsis_2026_source_access_v1 import _safe_path, _sanitized_url


def test_safe_path_strips_query_fragment():
    assert _safe_path("https://www.nflgsis.com/GameStatsLive/Schedule?token=secret#x") == "/GameStatsLive/Schedule"


def test_safe_path_rejects_external_hosts():
    assert _safe_path("https://id.nfl.com/account/sign-in?secret=1") == ""


def test_sanitized_url_strips_query_fragment():
    assert _sanitized_url("https://www.nflgsis.com/GameStatsLive/Auth/?foo=bar#frag") == "https://www.nflgsis.com/GameStatsLive/Auth/"


def test_sanitized_url_keeps_host_path_only_for_external_auth():
    assert _sanitized_url("https://id.nfl.com/account/sign-in?state=secret") == "https://id.nfl.com/account/sign-in"
