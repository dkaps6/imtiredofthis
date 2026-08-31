"""M95D implementation-fix shim.

This does not alter the pre-specified M95D model families or advancement gates.
It fixes two implementation issues found before a successful scientific run:
1) the AUC gain reporter resolved Series.corr as the pandas method instead of
   the stored `corr` result column;
2) nflverse PFR weekly defense uses `def_missed_tackles` and
   `def_tackles_combined` column names.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import scripts.backtest.evaluate_rb_rushing_environment_scheme as m


def gain_table_fixed(r: pd.DataFrame) -> pd.DataFrame:
    rows=[];base="role_plus_m95c_environment";cand="full_environment_matchup"
    for (sp,t),g in r.groupby(["split","target"]):
        a=g.loc[g["family"].eq(base)];b=g.loc[g["family"].eq(cand)]
        if a.empty or b.empty:continue
        if t.endswith("_auc"):
            gain=float(b.iloc[0]["corr"]-a.iloc[0]["corr"]);metric="auc_gain"
        else:
            gain=float(a.iloc[0]["mae"]-b.iloc[0]["mae"]);metric="mae_gain"
        rows.append({"split":sp,"target":t,"baseline":base,"candidate":cand,"metric":metric,"gain":gain})
    return pd.DataFrame(rows)


def read_pfr_def_fixed(root: Path):
    fs=[]
    for p in sorted(root.glob("advstats_week_def_*.csv")):
        try:fs.append(m.lower(pd.read_csv(p,low_memory=False)))
        except Exception:pass
    if not fs:
        return pd.DataFrame(),pd.DataFrame([{"source":"pfr_week_def","status":"missing","rows":0,"join_rate":np.nan}])
    x=pd.concat(fs,ignore_index=True,sort=False)
    sc=m.alias(x,["season"]);wc=m.alias(x,["week"]);tc=m.alias(x,["team","tm"])
    mt=m.alias(x,["def_missed_tackles","missed_tackles","miss_tkl","mtkl","tackles_missed"])
    tk=m.alias(x,["def_tackles_combined","tackles_combined","comb","tackles","total_tackles"])
    if not all([sc,wc,tc,mt]):
        return pd.DataFrame(),pd.DataFrame([{"source":"pfr_week_def","status":"schema_no_missed_tackle","rows":len(x),"join_rate":np.nan}])
    x["_miss"]=m.num(x[mt],0);x["_tackles"]=m.num(x[tk],0) if tk else np.nan
    x["team"]=x[tc].map(m.team);x["season"]=m.num(x[sc]);x["week"]=m.num(x[wc])
    # PFR weekly defense is player-level. Summing tackles and missed tackles to
    # team-week yields a consistent tackling-vulnerability proxy.
    g=x.groupby(["season","week","team"],as_index=False).agg(
        missed_tackles_pg=("_miss","sum"),tackles_pg=("_tackles","sum"))
    g["missed_tackle_rate"]=g.missed_tackles_pg/(g.tackles_pg+g.missed_tackles_pg).replace(0,np.nan)
    return g,pd.DataFrame([{"source":"pfr_week_def","status":"ok","rows":len(x),"join_rate":np.nan}])


m.gain_table=gain_table_fixed
m.read_pfr_def=read_pfr_def_fixed

if __name__=="__main__":
    raise SystemExit(m.main())
