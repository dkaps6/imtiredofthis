from datetime import datetime
from scripts.research.probe_wr_cb_wayback_available_v1 import parse_api,classify
def d(x):return datetime.fromisoformat(x.replace("Z","+00:00"))
def test_parse_available_and_unavailable():
    p={"archived_snapshots":{"closest":{"available":True,"status":"200","timestamp":"20240907005714","url":"https://web.archive.org/web/20240907005714/x"}}}
    snap,status=parse_api(p);assert status=="AVAILABLE" and snap["timestamp"]=="20240907005714"
    snap,status=parse_api({"archived_snapshots":{}});assert snap is None and "NOT_PROOF_OF_ABSENCE" in status
def test_time_class():
    first,last=d("2024-09-06T00:20:00Z"),d("2024-09-10T00:15:00Z")
    assert classify("20240905000000",first,last)=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify("20240907005714",first,last)=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    assert classify("20240911000000",first,last)=="POST_WEEK_INDEX_ONLY"
