from datetime import datetime, timezone
from scripts.research.probe_wr_cb_sparse_exact_cdx_v1 import classify, parse_payload

URL="https://www.fantasyalarm.com/articles/nfl/wide-receivers/example/123"

def test_parse_payload_keeps_only_exact_200_rows():
    payload=[
      ["timestamp","original","statuscode","digest"],
      ["20220922010000",URL,"200","ABC"],
      ["20220922020000",URL,"302","DEF"],
      ["20220922030000","https://example.com/x","200","GHI"],
    ]
    rows=parse_payload(payload,URL)
    assert rows==[{"timestamp":"20220922010000","digest":"ABC","original":URL}]

def test_classify_full_partial_post():
    first=datetime(2022,9,23,0,15,tzinfo=timezone.utc)
    last=datetime(2022,9,27,0,15,tzinfo=timezone.utc)
    assert classify(datetime(2022,9,22,tzinfo=timezone.utc),first,last)=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify(datetime(2022,9,24,tzinfo=timezone.utc),first,last)=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify(datetime(2022,9,28,tzinfo=timezone.utc),first,last)=="POST_WEEK_INDEX_ONLY"
