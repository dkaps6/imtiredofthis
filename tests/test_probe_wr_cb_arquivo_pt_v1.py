from datetime import datetime, timezone
from scripts.research.probe_wr_cb_arquivo_pt_v1 import classify, parse_payload

URL="https://www.fantasyalarm.com/articles/nfl/wide-receivers/example/123"

def test_parse_payload_keeps_only_exact_url_versions():
    payload={"response_items":[
      {"originalURL":URL,"tstamp":"20220922010000","digest":"ABC","status":"200","linkToArchive":"https://arquivo.pt/wayback/x","collection":"A"},
      {"originalURL":"https://example.com/other","tstamp":"20220922020000","digest":"DEF"},
      {"originalURL":URL,"tstamp":"bad","digest":"GHI"},
    ]}
    assert parse_payload(payload,URL)==[{
      "timestamp":"20220922010000","digest":"ABC","original":URL,
      "status":"200","link_to_archive":"https://arquivo.pt/wayback/x","collection":"A"
    }]

def test_classify_archive_timestamp():
    first=datetime(2022,9,23,0,15,tzinfo=timezone.utc)
    last=datetime(2022,9,27,0,15,tzinfo=timezone.utc)
    assert classify(datetime(2022,9,22,tzinfo=timezone.utc),first,last)=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify(datetime(2022,9,24,tzinfo=timezone.utc),first,last)=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify(datetime(2022,9,28,tzinfo=timezone.utc),first,last)=="POST_WEEK_INDEX_ONLY"
