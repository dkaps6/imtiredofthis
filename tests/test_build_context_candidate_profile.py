import subprocess, sys
from pathlib import Path
import pandas as pd

SCRIPT=Path(__file__).parents[1]/"scripts/research/build_context_candidate_profile.py"

def run(tmp_path, df, extra=None):
    inp=tmp_path/"in.csv"; out=tmp_path/"out.csv"; df.to_csv(inp,index=False)
    cmd=[sys.executable,str(SCRIPT),"--input",str(inp),"--features","x","--family","role","--grain","player_game","--key-cols","season,week,player_id","--intended-component","usage","--mechanism-note","detect stale usage","--out",str(out)]
    if extra: cmd.extend(extra)
    p=subprocess.run(cmd,capture_output=True,text=True)
    return p,out

def test_profiles_coverage_and_identity(tmp_path):
    df=pd.DataFrame({"season":[2025,2025,2025],"week":[1,2,3],"player_id":["a"]*3,"x":[1.0,None,2.0],"eligible":[1,1,0],"stable":[1,1,1],"unknown":[0,1,0],"support":[5,6,7]})
    p,out=run(tmp_path,df,["--eligible-col","eligible","--stable-id-col","stable","--unknown-col","unknown","--prior-support-col","support"])
    assert p.returncode==0,p.stderr
    r=pd.read_csv(out).iloc[0]
    assert r.eligible_rows==2
    assert r.pregame_coverage==0.5
    assert r.stable_id_coverage==1.0
    assert r.unknown_rate==0.5
    assert r.prior_support_median==5.0

def test_duplicate_keys_are_reported(tmp_path):
    df=pd.DataFrame({"season":[2025,2025],"week":[1,1],"player_id":["a","a"],"x":[1,2]})
    p,out=run(tmp_path,df)
    assert p.returncode==0
    assert pd.read_csv(out).iloc[0].duplicate_key_count==2

def test_missing_feature_fails_closed(tmp_path):
    df=pd.DataFrame({"season":[2025],"week":[1],"player_id":["a"]})
    p,_=run(tmp_path,df)
    assert p.returncode!=0
    assert "missing required columns" in (p.stderr+p.stdout)

def test_duplicate_stability_rows_fail_closed(tmp_path):
    df=pd.DataFrame({"season":[2025],"week":[1],"player_id":["a"],"x":[1]})
    stab=tmp_path/"stab.csv"
    pd.DataFrame({"feature_name":["x","x"],"stability_stat":["s","s"],"stability_value":[.5,.6]}).to_csv(stab,index=False)
    p,_=run(tmp_path,df,["--stability",str(stab)])
    assert p.returncode!=0
    assert "duplicate feature_name" in (p.stderr+p.stdout)
