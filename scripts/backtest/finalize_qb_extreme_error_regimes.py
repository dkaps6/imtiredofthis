#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def num(v): return pd.to_numeric(v,errors='coerce')
def read(p):
    x=pd.read_csv(p); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def effect(a,b):
    a=num(a).dropna(); b=num(b).dropna()
    if len(a)<5 or len(b)<5:return np.nan
    pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(len(a)+len(b)-2,1))
    return float((a.mean()-b.mean())/pooled) if pooled>0 else np.nan

def univ_auc(x,y):
    z=pd.DataFrame({'x':num(x),'y':num(y)}).dropna()
    if len(z)<30 or z.y.nunique()<2 or z.x.nunique()<2:return (np.nan,'none')
    a=roc_auc_score(z.y,z.x)
    return (float(a),'higher') if a>=.5 else (float(1-a),'lower')

def metric_row(name,y,p):
    z=pd.DataFrame({'y':num(y),'p':num(p)}).dropna()
    if z.y.nunique()<2:return {'model':name,'n':len(z),'roc_auc':np.nan,'average_precision':np.nan,'brier':np.nan}
    return {'model':name,'n':len(z),'roc_auc':roc_auc_score(z.y,z.p),'average_precision':average_precision_score(z.y,z.p),'brier':brier_score_loss(z.y,z.p)}

def fit_models(train,test,features,target):
    Xtr=train[features]; ytr=num(train[target]).astype(int); Xte=test[features]
    log=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.35,class_weight='balanced',max_iter=3000,random_state=62))
    log.fit(Xtr,ytr); pl=log.predict_proba(Xte)[:,1]
    imp=SimpleImputer(strategy='median'); A=imp.fit_transform(Xtr); B=imp.transform(Xte)
    weights=np.where(ytr.eq(1),max((ytr.eq(0).sum()/max(ytr.eq(1).sum(),1)),1),1.0)
    h=HistGradientBoostingClassifier(learning_rate=.035,max_iter=180,max_leaf_nodes=12,min_samples_leaf=18,l2_regularization=4.0,random_state=62)
    h.fit(A,ytr,sample_weight=weights); ph=h.predict_proba(B)[:,1]
    return {'logistic':pl,'hist_gb':ph}

def risk_buckets(test,pred,label):
    z=test.copy(); z['score']=pred
    try:z['bucket']=pd.qcut(z.score,5,labels=['Q1_low','Q2','Q3','Q4','Q5_high'],duplicates='drop')
    except:z['bucket']='all'
    rows=[]
    for b,g in z.groupby('bucket',observed=True):
        rows.append({'target':label,'bucket':str(b),'n':len(g),'mean_score':g.score.mean(),'mae':g.abs_pass_error.mean(),'good_lt50_rate':g.good_lt50.mean(),'poor_75plus_rate':g.poor_75plus.mean(),'catastrophic_rate':g.catastrophic_100plus.mean(),'attempt_miss_10plus_rate':g.attempt_miss_10plus.mean(),'actual_40plus_attempt_rate':g.actual_40plus_attempts.mean()})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--season-file',action='append',type=Path,required=True); ap.add_argument('--manifest-file',action='append',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    games=pd.concat([read(p) for p in a.season_file],ignore_index=True); games['season']=num(games.season).astype(int)
    manifests=[set(read(p).feature.astype(str)) for p in a.manifest_file]; features=sorted(set.intersection(*manifests)); features=[f for f in features if f in games and num(games[f]).notna().mean()>=.50 and num(games[f]).nunique(dropna=True)>=2]
    screens=[]
    for f in features:
        row={'feature':f,'coverage':num(games[f]).notna().mean(),'nunique':num(games[f]).nunique(dropna=True),'corr_abs_error':num(games[f]).corr(num(games.abs_pass_error))}
        for target in ('catastrophic_100plus','good_lt50','attempt_miss_10plus','actual_40plus_attempts'):
            auc,direction=univ_auc(games[f],games[target]); row[f'{target}_auc']=auc; row[f'{target}_direction']=direction
        row['cat_effect_d']=effect(games[games.catastrophic_100plus.eq(1)][f],games[games.catastrophic_100plus.eq(0)][f]); row['good_effect_d']=effect(games[games.good_lt50.eq(1)][f],games[games.good_lt50.eq(0)][f])
        for s in (2024,2025):
            q=games[games.season.eq(s)]; row[f'cat_mean_diff_{s}']=num(q[q.catastrophic_100plus.eq(1)][f]).mean()-num(q[q.catastrophic_100plus.eq(0)][f]).mean(); row[f'good_mean_diff_{s}']=num(q[q.good_lt50.eq(1)][f]).mean()-num(q[q.good_lt50.eq(0)][f]).mean()
        row['cat_direction_replicates']=bool(np.sign(row['cat_mean_diff_2024'])==np.sign(row['cat_mean_diff_2025']) and np.sign(row['cat_mean_diff_2024'])!=0) if np.isfinite(row['cat_mean_diff_2024']) and np.isfinite(row['cat_mean_diff_2025']) else False
        row['good_direction_replicates']=bool(np.sign(row['good_mean_diff_2024'])==np.sign(row['good_mean_diff_2025']) and np.sign(row['good_mean_diff_2024'])!=0) if np.isfinite(row['good_mean_diff_2024']) and np.isfinite(row['good_mean_diff_2025']) else False
        screens.append(row)
    screen=pd.DataFrame(screens).sort_values(['catastrophic_100plus_auc','cat_effect_d'],ascending=[False,False])
    bands=[]; order=[('lt25',0,25),('25_49',25,50),('50_74',50,75),('75_99',75,100),('100plus',100,np.inf)]; total=games.abs_pass_error.sum()
    for name,lo,hi in order:
        g=games[(games.abs_pass_error>=lo)&(games.abs_pass_error<hi)]; bands.append({'error_band':name,'n':len(g),'share':len(g)/len(games),'mae':g.abs_pass_error.mean(),'rmse':np.sqrt(np.mean(g.pass_error**2)) if len(g) else np.nan,'bias':g.pass_error.mean(),'correlation':g.raw_projection.corr(g.actual),'error_share':g.abs_pass_error.sum()/total if total else np.nan})
    banddf=pd.DataFrame(bands); cf=[]; cats=games[games.abs_pass_error>=100]; other=games[games.abs_pass_error<100]
    for replacement in (100,75,60,50,40,30,20): cf.append({'assumed_mean_mae_for_current_100plus_games':replacement,'resulting_overall_mae':(other.abs_pass_error.sum()+len(cats)*replacement)/len(games)})
    cf=pd.DataFrame(cf)
    train=games[games.season.eq(2024)].copy(); test=games[games.season.eq(2025)].copy(); model_rows=[]; score_tables=[]; bucket_tables=[]
    for target in ('catastrophic_100plus','good_lt50'):
        preds=fit_models(train,test,features,target)
        for name,p in preds.items(): model_rows.append({'target':target,**metric_row(name,test[target],p)})
        candidate=pd.DataFrame([r for r in model_rows if r['target']==target]).sort_values(['roc_auc','average_precision'],ascending=False).iloc[0]; best=candidate['model']; p=preds[best]
        sc=test[['season','week','team','player_clean_key','actual','raw_projection','abs_pass_error','good_lt50','poor_75plus','catastrophic_100plus','attempt_miss_10plus','actual_40plus_attempts']].copy(); sc['target']=target; sc['model']=best; sc['score']=p; score_tables.append(sc); bucket_tables.append(risk_buckets(test,p,target).assign(model=best))
    metrics=pd.DataFrame(model_rows); scores=pd.concat(score_tables,ignore_index=True); buckets=pd.concat(bucket_tables,ignore_index=True)
    catm=metrics[metrics.target.eq('catastrophic_100plus')].sort_values('roc_auc',ascending=False).iloc[0]; catb=buckets[(buckets.target.eq('catastrophic_100plus'))&(buckets.model.eq(catm.model))]; base_cat=test.catastrophic_100plus.mean(); hi=catb.sort_values('mean_score').iloc[-1]; cat_action=bool(catm.roc_auc>=.60 and hi.catastrophic_rate>=base_cat*1.5)
    goodm=metrics[metrics.target.eq('good_lt50')].sort_values('roc_auc',ascending=False).iloc[0]; goodb=buckets[(buckets.target.eq('good_lt50'))&(buckets.model.eq(goodm.model))]; base_good=test.good_lt50.mean(); base_mae=test.abs_pass_error.mean(); ghi=goodb.sort_values('mean_score').iloc[-1]; good_action=bool(goodm.roc_auc>=.60 and ghi.good_lt50_rate>=base_good+.10 and ghi.mae<=base_mae*.85)
    replicated=screen[(screen.cat_direction_replicates)|(screen.good_direction_replicates)].copy(); strong=replicated[(replicated.catastrophic_100plus_auc>=.58)|(replicated.good_lt50_auc>=.58)].copy()
    verdict=pd.DataFrame([{'catastrophic_risk_classifier_actionable':cat_action,'good_game_confidence_classifier_actionable':good_action,'best_cat_model':catm.model,'cat_2025_auc':catm.roc_auc,'cat_2025_top_quintile_rate':hi.catastrophic_rate,'cat_2025_base_rate':base_cat,'best_good_model':goodm.model,'good_2025_auc':goodm.roc_auc,'good_2025_top_quintile_hit_rate':ghi.good_lt50_rate,'good_2025_top_quintile_mae':ghi.mae,'good_2025_base_hit_rate':base_good,'good_2025_base_mae':base_mae,'replicated_strong_univariate_features':len(strong),'m62_actionable_regime_signal':bool(cat_action or good_action or len(strong)>=2)}])
    a.out_dir.mkdir(parents=True,exist_ok=True); games.to_csv(a.out_dir/'m62_combined_enriched_games.csv',index=False); banddf.to_csv(a.out_dir/'m62_combined_error_bands.csv',index=False); cf.to_csv(a.out_dir/'m62_tail_mae_counterfactuals.csv',index=False); screen.to_csv(a.out_dir/'m62_feature_screen.csv',index=False); strong.to_csv(a.out_dir/'m62_replicated_strong_features.csv',index=False); metrics.to_csv(a.out_dir/'m62_2024_to_2025_classifier_metrics.csv',index=False); scores.to_csv(a.out_dir/'m62_2025_oos_risk_scores.csv',index=False); buckets.to_csv(a.out_dir/'m62_2025_risk_quintiles.csv',index=False); verdict.to_csv(a.out_dir/'m62_precommitted_interpretation.csv',index=False)
    print('=== M62 COMBINED ERROR BANDS ==='); print(banddf.to_string(index=False)); print('\n=== M62 TOP FEATURE SCREEN ==='); print(screen.head(25).to_string(index=False)); print('\n=== M62 2024->2025 CLASSIFIERS ==='); print(metrics.to_string(index=False)); print('\n=== M62 2025 RISK QUINTILES ==='); print(buckets.to_string(index=False)); print('\n=== M62 INTERPRETATION ==='); print(verdict.to_string(index=False))
if __name__=='__main__': main()
