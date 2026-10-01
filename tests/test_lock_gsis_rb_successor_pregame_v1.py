import gzip
import json
from pathlib import Path
import pandas as pd
from scripts.research.lock_gsis_rb_successor_pregame_v1 import build_lock

def _snapshot(path:Path,capture="2026-10-01T12:00:00Z"):
    players_a=["A","X","1","2","3","4","5","6","7","8","9"]
    players_b=["B","X","1","2","3","4","5","6","7","8","9"]
    def row(names,plays):
        return {"cells":[
          {"tag":"td","text":", ".join(names)},
          {"tag":"td","text":str(plays)},
        ]}
    obj={
      "snapshot_id":"synthetic","season":2026,"phase":"REG",
      "records":[{
        "report":"Lineup Detail","mode":"Offense",
        "capture_timestamp_utc":capture,
        "filters":[{"id":"select2","value":"DEN"}],
        "tables":[{"rows":[
          {"cells":[{"tag":"th","text":"Lineup"},{"tag":"th","text":"Plays"}]},
          row(players_a,30),row(players_b,10),
        ]}],
      }],
    }
    raw=json.dumps(obj).encode()
    with gzip.GzipFile(filename="",mode="wb",fileobj=path.open("wb"),mtime=0) as f:
        f.write(raw)
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _vacancy():
    return pd.DataFrame([
      {"target_season":2026,"target_week":5,"team":"DEN","successor_player_clean_key":"a",
       "vacated_rush_share":0.20,"successor_weight":0.4,"transfer_rush_share":0.08,
       "unavailable_players":"u"},
      {"target_season":2026,"target_week":5,"team":"DEN","successor_player_clean_key":"b",
       "vacated_rush_share":0.20,"successor_weight":0.6,"transfer_rush_share":0.12,
       "unavailable_players":"u"},
    ])

def _pool(extra=False):
    rows=[
      {"target_season":2026,"target_week":5,"team":"DEN","successor_player_clean_key":"a"},
      {"target_season":2026,"target_week":5,"team":"DEN","successor_player_clean_key":"b"},
    ]
    if extra:
        rows.append({"target_season":2026,"target_week":5,"team":"DEN","successor_player_clean_key":"c"})
    return pd.DataFrame(rows)

def test_future_lock_conserves_both_arms(tmp_path):
    p=tmp_path/"s.json.gz"; sha=_snapshot(p)
    events=pd.DataFrame([{"target_season":2026,"target_week":5,"team":"DEN","event_id":"g","kickoff_utc":"2026-10-02T00:00:00Z"}])
    out,audit=build_lock(snapshot_path=p,expected_snapshot_sha256=sha,vacancy=_vacancy(),events=events,successor_pool=_pool())
    assert audit["events_locked"]==1
    assert abs(out.snap_transfer_rush_share.sum()-0.20)<1e-12
    assert abs(out.gsis_transfer_rush_share.sum()-0.20)<1e-12
    got=dict(zip(out.successor_player_clean_key,out.gsis_successor_weight))
    assert abs(got["a"]-.75)<1e-12
    assert abs(got["b"]-.25)<1e-12

def test_postkickoff_snapshot_fails_closed_without_private_rows(tmp_path):
    p=tmp_path/"s.json.gz"; sha=_snapshot(p,capture="2026-10-03T12:00:00Z")
    events=pd.DataFrame([{"target_season":2026,"target_week":5,"team":"DEN","event_id":"g","kickoff_utc":"2026-10-02T00:00:00Z"}])
    out,audit=build_lock(snapshot_path=p,expected_snapshot_sha256=sha,vacancy=_vacancy(),events=events,successor_pool=_pool())
    assert out.empty
    assert audit["disposition"]=="SOURCE_TIMING_INVALID"
    assert audit["events_source_timing_invalid"]==1

def test_snapshot_hash_is_fail_closed(tmp_path):
    p=tmp_path/"s.json.gz"; _snapshot(p)
    events=pd.DataFrame([{"target_season":2026,"target_week":5,"team":"DEN","event_id":"g","kickoff_utc":"2026-10-02T00:00:00Z"}])
    try:
        build_lock(snapshot_path=p,expected_snapshot_sha256="0"*64,vacancy=_vacancy(),events=events,successor_pool=_pool())
    except RuntimeError as e:
        assert "SHA mismatch" in str(e)
    else:
        raise AssertionError("expected hash mismatch failure")

def test_active_successor_without_snap_weight_can_receive_gsis_mass(tmp_path):
    p=tmp_path/"s.json.gz"
    # Explicit synthetic snapshot: A has 30 plays, C has 20, B has 10.
    obj={
      "snapshot_id":"synthetic","season":2026,"phase":"REG",
      "records":[{
        "report":"Lineup Detail","mode":"Offense",
        "capture_timestamp_utc":"2026-10-01T12:00:00Z",
        "filters":[{"id":"select2","value":"DEN"}],
        "tables":[{"rows":[
          {"cells":[{"tag":"th","text":"Lineup"},{"tag":"th","text":"Plays"}]},
          {"cells":[{"tag":"td","text":"A, X, 1, 2, 3, 4, 5, 6, 7, 8, 9"},{"tag":"td","text":"30"}]},
          {"cells":[{"tag":"td","text":"B, X, 1, 2, 3, 4, 5, 6, 7, 8, 9"},{"tag":"td","text":"10"}]},
          {"cells":[{"tag":"td","text":"C, X, 1, 2, 3, 4, 5, 6, 7, 8, 9"},{"tag":"td","text":"20"}]},
        ]}],
      }],
    }
    raw=json.dumps(obj).encode()
    with gzip.GzipFile(filename="",mode="wb",fileobj=p.open("wb"),mtime=0) as z:
        z.write(raw)
    import hashlib
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    events=pd.DataFrame([{"target_season":2026,"target_week":5,"team":"DEN","event_id":"g","kickoff_utc":"2026-10-02T00:00:00Z"}])
    out,audit=build_lock(snapshot_path=p,expected_snapshot_sha256=sha,vacancy=_vacancy(),events=events,successor_pool=_pool(extra=True))
    c_row=out.loc[out.successor_player_clean_key.eq("c")].iloc[0]
    assert c_row.snap_successor_weight==0.0
    assert c_row.snap_transfer_rush_share==0.0
    assert c_row.gsis_successor_weight>0.0
    assert c_row.gsis_transfer_rush_share>0.0
    assert audit["successor_pool_mode"]=="EXPLICIT_ACTIVE_POOL"
