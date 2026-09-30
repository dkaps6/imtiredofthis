from datetime import datetime,timezone
from scripts.research.discover_wr_cb_wayback_2024_2025_v1 import classify,same_url,ts

def d(x): return datetime.fromisoformat(x.replace("Z","+00:00"))

def test_exact_article_identity_and_time_classes():
    a="https://www.fantasyalarm.com/articles/nfl/wide-receivers/foo/123"
    assert same_url(a,"http://fantasyalarm.com/articles/nfl/wide-receivers/foo/123/")
    assert not same_url(a,a.replace("/123","/124"))
    first,last=d("2024-09-06T00:20:00Z"),d("2024-09-10T00:15:00Z")
    assert classify(d("2024-09-05T23:00:00Z"),first,last)=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify(d("2024-09-07T00:00:00Z"),first,last)=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify(d("2024-09-11T00:00:00Z"),first,last)=="POST_WEEK_INDEX_ONLY"
    assert ts("bad") is None
