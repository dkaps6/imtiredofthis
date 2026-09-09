#!/usr/bin/env python3
"""Pool R23 confirmation folds and apply the frozen promotion gates."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def metric(a,p):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'p':pd.to_numeric(p,errors='coerce')}).dropna()
    z=z[np.isfinite(z.a)&np.isfinite(z.p)]
    e=z.p-z.a; ae=e.abs()
    return {'n':int(len(z)),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e.to_numpy()**2))),'bias':float(e.mean()),'p90':float(ae.quantile(.90)),'miss30':float((ae>=30).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dirs',nargs='+',required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    frames=[]
    for d in a.dirs:
        x=pd.read_csv(Path(d)/'r23_predictions.csv',low_memory=False); frames.append(x)
    x=pd.concat(frames,ignore_index=True)
    if int(x.sportsbook_inputs_used.fillna(0).sum()) or int(x.future_outcomes_used.fillna(0).sum()): raise RuntimeError('integrity violation')
    rows=[]
    for season in sorted(x.season.unique()):
        g=x[x.season.eq(season)]
        for v in ('baseline','candidate'):
            for m,ac in (('targets','actual_targets'),('receptions','actual_receptions'),('rec_yards','actual_rec_yards')):
                rows.append({'scope':str(season),'role':'ALL','variant':v,'market':m,**metric(g[ac],g[f'{v}_{m}'])})
    for role,g in x.groupby('role'):
        for v in ('baseline','candidate'):
            rows.append({'scope':'POOLED','role':role,'variant':v,'market':'rec_yards',**metric(g.actual_rec_yards,g[f'{v}_rec_yards'])})
    for v in ('baseline','candidate'):
        for m,ac in (('targets','actual_targets'),('receptions','actual_receptions'),('rec_yards','actual_rec_yards')):
            rows.append({'scope':'POOLED','role':'ALL','variant':v,'market':m,**metric(x[ac],x[f'{v}_{m}'])})
    s=pd.DataFrame(rows)
    def r(scope,variant,market,role='ALL'): return s[(s.scope.eq(str(scope)))&(s.variant.eq(variant))&(s.market.eq(market))&(s.role.eq(role))].iloc[0]
    bt,ct=r('POOLED','baseline','targets'),r('POOLED','candidate','targets')
    br,cr=r('POOLED','baseline','receptions'),r('POOLED','candidate','receptions')
    by,cy=r('POOLED','baseline','rec_yards'),r('POOLED','candidate','rec_yards')
    season_changes={}; improves=0; max_worsen=0.0
    for season in sorted(x.season.unique()):
        b,c=r(season,'baseline','rec_yards'),r(season,'candidate','rec_yards')
        pct=(float(c.mae)-float(b.mae))/float(b.mae) if float(b.mae)>0 else 0.0
        season_changes[str(int(season))]=pct
        improves += int(float(c.mae)<float(b.mae)); max_worsen=max(max_worsen,pct)
    role_gate=True; role_changes={}
    for role in ('RB1','RB2+'):
        b,c=r('POOLED','baseline','rec_yards',role),r('POOLED','candidate','rec_yards',role)
        pct=(float(c.mae)-float(b.mae))/float(b.mae) if float(b.mae)>0 else 0.0; role_changes[role]=pct; role_gate &= pct<=0.01+1e-12
    gates={
      'targets_mae_improves':float(ct.mae)<float(bt.mae),
      'targets_rmse_nonworse':float(ct.rmse)<=float(bt.rmse)+1e-12,
      'receptions_mae_improves':float(cr.mae)<float(br.mae),
      'receptions_rmse_nonworse':float(cr.rmse)<=float(br.rmse)+1e-12,
      'rec_yards_mae_improves_1pct':float(cy.mae)<=float(by.mae)*0.99,
      'rec_yards_rmse_nonworse':float(cy.rmse)<=float(by.rmse)+1e-12,
      'directional_replication':improves>=2 and max_worsen<=0.01+1e-12,
      'p90_protected':float(cy.p90)<=float(by.p90)*1.02+1e-12,
      'miss30_protected':float(cy.miss30)<=float(by.miss30)+0.01+1e-12,
      'yards_bias_protected':abs(float(cy.bias))<=abs(float(by.bias))+1.0+1e-12,
      'receptions_bias_protected':abs(float(cr.bias))<=abs(float(br.bias))+0.10+1e-12,
      'role_robustness':bool(role_gate),
      'sportsbook_zero':True,'future_2026_zero':True
    }
    passed=all(gates.values())
    out={'candidate':'RB_R23_RECEIVING_ENTITLEMENT_MEAN_V1','disposition':'PASS_FROZEN_SCIENTIFIC_GATES_INTEGRATION_AUTHORIZED' if passed else 'MIXED_OR_FAIL_NO_PROMOTION','pass':passed,'gates':gates,'season_rec_yards_mae_pct_changes':season_changes,'role_rec_yards_mae_pct_changes':role_changes,'pooled':{'targets':{'baseline_mae':float(bt.mae),'candidate_mae':float(ct.mae),'baseline_rmse':float(bt.rmse),'candidate_rmse':float(ct.rmse)},'receptions':{'baseline_mae':float(br.mae),'candidate_mae':float(cr.mae),'baseline_rmse':float(br.rmse),'candidate_rmse':float(cr.rmse),'baseline_bias':float(br.bias),'candidate_bias':float(cr.bias)},'rec_yards':{'baseline_mae':float(by.mae),'candidate_mae':float(cy.mae),'baseline_rmse':float(by.rmse),'candidate_rmse':float(cy.rmse),'baseline_bias':float(by.bias),'candidate_bias':float(cy.bias),'baseline_p90':float(by.p90),'candidate_p90':float(cy.p90),'baseline_miss30':float(by.miss30),'candidate_miss30':float(cy.miss30)}}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); s.to_csv(a.out.with_suffix('.csv'),index=False)
    print(json.dumps(out,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
