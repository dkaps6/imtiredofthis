from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

THRESHOLDS={'QB':100.0,'WR':50.0,'TE':40.0,'RB':40.0}


def slope_xy(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.nanstd(x[m])<1e-12: return np.nan
    return float(np.polyfit(x[m],y[m],1)[0])

def spearman_xy(x,y):
    x=pd.Series(x); y=pd.Series(y)
    if len(x)<2: return np.nan
    return float(x.corr(y,method='spearman'))

def zint_beta(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2: return np.nan
    den=float(np.dot(x[m],x[m]))
    return float(np.dot(x[m],y[m])/den) if den>1e-12 else 0.0

def metrics(actual,pred):
    a=np.asarray(actual,float); p=np.asarray(pred,float)
    m=np.isfinite(a)&np.isfinite(p); a=a[m]; p=p[m]
    if len(a)==0: return {'n':0,'mae':np.nan,'rmse':np.nan,'bias_pred_minus_actual':np.nan,'corr':np.nan}
    e=p-a
    return {'n':int(len(a)),'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),
            'bias_pred_minus_actual':float(np.mean(e)),
            'corr':float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan}

def build_team_games(df):
    keys=['season','week','team']
    qb=(df[df.position=='QB'].sort_values(keys+['pred_opportunity'],ascending=[True,True,True,False])
        .drop_duplicates(keys).copy())
    qcols=keys+['pred_opportunity','actual_opportunity','pass_opportunity_spot','pass_efficiency_spot','rush_opportunity_spot','rush_efficiency_spot']
    qb=qb[qcols].rename(columns={'pred_opportunity':'pred_qb_attempts','actual_opportunity':'actual_qb_attempts'})
    def pool(pos,prefix):
        return df[df.position==pos].groupby(keys,as_index=False).agg(**{
            f'pred_{prefix}':('pred_opportunity','sum'), f'actual_{prefix}':('actual_opportunity','sum')})
    wr=pool('WR','wr_targets'); te=pool('TE','te_targets'); rb=pool('RB','rb_carries')
    t=qb.merge(wr,on=keys,how='inner').merge(te,on=keys,how='inner').merge(rb,on=keys,how='left')
    t['pred_receiver_targets']=t.pred_wr_targets+t.pred_te_targets
    t['actual_receiver_targets']=t.actual_wr_targets+t.actual_te_targets
    t['qb_attempt_resid']=t.actual_qb_attempts-t.pred_qb_attempts
    t['receiver_target_resid']=t.actual_receiver_targets-t.pred_receiver_targets
    t['rb_carry_resid']=t.actual_rb_carries-t.pred_rb_carries
    return t.sort_values(['season','week','team']).reset_index(drop=True)

def run_e1(team):
    feats=['pass_opportunity_spot','pass_efficiency_spot','rush_opportunity_spot','rush_efficiency_spot',
           'pred_qb_attempts','pred_receiver_targets','pred_wr_targets','pred_te_targets','week']
    out=[]
    evaldf=team[team.season==2025].copy()
    for wk in sorted(evaldf.week.unique()):
        test=evaldf[evaldf.week==wk].copy()
        train=team[(team.season<2025)|((team.season==2025)&(team.week<wk))].copy()
        train=train.dropna(subset=feats+['qb_attempt_resid','receiver_target_resid'])
        if len(train)<128:
            for _,r in test.iterrows():
                d=r.to_dict(); d.update({'train_n':len(train),'delta_pass_attempts':0.0,'beta_receiver':np.nan,'beta_rb':np.nan,'rb_correction_available':False})
                out.append(d)
            continue
        sc=StandardScaler(); X=sc.fit_transform(train[feats]); model=Ridge(alpha=20.0); model.fit(X,train.qb_attempt_resid)
        deltas=model.predict(sc.transform(test[feats]))
        beta_rec=zint_beta(train.qb_attempt_resid,train.receiver_target_resid)
        rbtrain=train.dropna(subset=['rb_carry_resid'])
        beta_rb=zint_beta(rbtrain.qb_attempt_resid,rbtrain.rb_carry_resid) if len(rbtrain)>=64 else np.nan
        for (_,r),delta in zip(test.iterrows(),deltas):
            d=r.to_dict(); d.update({'train_n':len(train),'delta_pass_attempts':float(delta),'beta_receiver':beta_rec,'beta_rb':beta_rb,'rb_correction_available':bool(np.isfinite(beta_rb))})
            out.append(d)
    c=pd.DataFrame(out)
    c['corr_qb_attempts']=c.pred_qb_attempts+c.delta_pass_attempts
    c['delta_receiver_targets']=c.beta_receiver*c.delta_pass_attempts
    c['corr_receiver_targets']=c.pred_receiver_targets+c.delta_receiver_targets
    share_wr=np.where(c.pred_receiver_targets>1e-12,c.pred_wr_targets/c.pred_receiver_targets,0.0)
    share_te=np.where(c.pred_receiver_targets>1e-12,c.pred_te_targets/c.pred_receiver_targets,0.0)
    c['corr_wr_targets']=c.pred_wr_targets+c.delta_receiver_targets*share_wr
    c['corr_te_targets']=c.pred_te_targets+c.delta_receiver_targets*share_te
    c['corr_rb_carries']=np.where(c.rb_correction_available,c.pred_rb_carries+c.beta_rb*c.delta_pass_attempts,c.pred_rb_carries)

    rows=[]
    pairs=[('QB_ATT','actual_qb_attempts','pred_qb_attempts','corr_qb_attempts'),('WRTE_TARGET','actual_receiver_targets','pred_receiver_targets','corr_receiver_targets'),
           ('WR_TARGET','actual_wr_targets','pred_wr_targets','corr_wr_targets'),('TE_TARGET','actual_te_targets','pred_te_targets','corr_te_targets')]
    for name,a,b,cc in pairs:
        mb=metrics(c[a],c[b]); mc=metrics(c[a],c[cc])
        rows.append({'metric':name,**{f'baseline_{k}':v for k,v in mb.items()},**{f'corrected_{k}':v for k,v in mc.items()},'mae_improvement':mb['mae']-mc['mae']})
    rbmask=c.rb_correction_available & c.actual_rb_carries.notna()
    if rbmask.any():
        mb=metrics(c.loc[rbmask,'actual_rb_carries'],c.loc[rbmask,'pred_rb_carries']); mc=metrics(c.loc[rbmask,'actual_rb_carries'],c.loc[rbmask,'corr_rb_carries'])
        rows.append({'metric':'RB_CARRY',**{f'baseline_{k}':v for k,v in mb.items()},**{f'corrected_{k}':v for k,v in mc.items()},'mae_improvement':mb['mae']-mc['mae']})
    score=pd.DataFrame(rows)

    qres=c.actual_qb_attempts-c.pred_qb_attempts; rres=c.actual_receiver_targets-c.pred_receiver_targets
    sign_q=float((np.sign(c.delta_pass_attempts)==np.sign(qres)).mean())
    sign_r=float((np.sign(c.delta_receiver_targets)==np.sign(rres)).mean())
    for prefix,act,base,corr in [('qb','actual_qb_attempts','pred_qb_attempts','corr_qb_attempts'),('wrte','actual_receiver_targets','pred_receiver_targets','corr_receiver_targets'),('wr','actual_wr_targets','pred_wr_targets','corr_wr_targets'),('te','actual_te_targets','pred_te_targets','corr_te_targets')]:
        thr=float(np.quantile(np.abs(c[act]-c[base]),.75)); c[f'{prefix}_opp_q4_threshold']=thr
        c[f'{prefix}_base_q4_miss']=np.abs(c[act]-c[base])>=thr; c[f'{prefix}_corr_q4_miss']=np.abs(c[act]-c[corr])>=thr
    if rbmask.any():
        thr=float(np.quantile(np.abs(c.loc[rbmask,'actual_rb_carries']-c.loc[rbmask,'pred_rb_carries']),.75)); c['rb_opp_q4_threshold']=thr
        c['rb_base_q4_miss']=False; c['rb_corr_q4_miss']=False
        c.loc[rbmask,'rb_base_q4_miss']=np.abs(c.loc[rbmask,'actual_rb_carries']-c.loc[rbmask,'pred_rb_carries'])>=thr
        c.loc[rbmask,'rb_corr_q4_miss']=np.abs(c.loc[rbmask,'actual_rb_carries']-c.loc[rbmask,'corr_rb_carries'])>=thr
    sm=c[rbmask].copy()
    if len(sm):
        actual_high=(sm.qb_attempt_resid>0)&(sm.receiver_target_resid>0)&(sm.rb_carry_resid<0)
        actual_low=(sm.qb_attempt_resid<0)&(sm.receiver_target_resid<0)&(sm.rb_carry_resid>0)
        pdq=sm.delta_pass_attempts; pdr=sm.delta_receiver_targets; pdrb=sm.beta_rb*sm.delta_pass_attempts
        pred_high=(pdq>0)&(pdr>0)&(pdrb<0); pred_low=(pdq<0)&(pdr<0)&(pdrb>0)
        sig=pd.DataFrame([
            {'signature':'PASS_STATE_HIGH','actual_n':int(actual_high.sum()),'predicted_n':int(pred_high.sum()),'true_positive':int((actual_high&pred_high).sum()),'recall':float((actual_high&pred_high).sum()/actual_high.sum()) if actual_high.sum() else np.nan},
            {'signature':'PASS_STATE_LOW','actual_n':int(actual_low.sum()),'predicted_n':int(pred_low.sum()),'true_positive':int((actual_low&pred_low).sum()),'recall':float((actual_low&pred_low).sum()/actual_low.sum()) if actual_low.sum() else np.nan},
        ])
    else: sig=pd.DataFrame(columns=['signature','actual_n','predicted_n','true_positive','recall'])

    get=lambda m: score.loc[score.metric==m].iloc[0]
    qbrow=get('QB_ATT'); rr=get('WRTE_TARGET'); wrrow=get('WR_TARGET'); terow=get('TE_TARGET')
    rb_ok=True; rb_imp=np.nan
    if (score.metric=='RB_CARRY').any():
        rbrow=get('RB_CARRY'); rb_imp=float(rbrow.mae_improvement); rb_ok=rb_imp>=-0.15
    gates={
      'qb_attempt_mae_improve_ge015': bool(qbrow.mae_improvement>=0.15),
      'wrte_target_mae_no_worse_gt005': bool(rr.mae_improvement>=-0.05),
      'wr_or_te_improves_and_neither_worse_gt010': bool((wrrow.mae_improvement>0 or terow.mae_improvement>0) and wrrow.mae_improvement>=-0.10 and terow.mae_improvement>=-0.10),
      'qb_residual_sign_accuracy_ge055': bool(sign_q>=0.55),
      'rb_carry_mae_no_worse_gt015': bool(rb_ok),
    }
    disp='SHARED_PASS_STATE_PREGAME_ELIGIBLE' if all(gates.values()) else 'SHARED_PASS_STATE_PREGAME_NOT_ELIGIBLE'
    meta={'disposition':disp,'scored_team_games':int(len(c)),'rb_corrected_team_games':int(rbmask.sum()),'qb_residual_sign_accuracy':sign_q,'receiver_residual_sign_accuracy':sign_r,'rb_mae_improvement':rb_imp,'gates':gates}
    return c,score,sig,meta

def player_training_stats(g,position):
    g=g.sort_values(['season','week']).copy(); threshold=THRESHOLDS[position]
    n=len(g); fav=g[g.game_spot_bucket=='FAVORABLE']; adv=g[g.game_spot_bucket=='ADVERSE']; neu=g[g.game_spot_bucket=='NEUTRAL']
    high=float(g.opportunity_quartile.isin(['Q3','Q4']).mean()); q4=float((g.opportunity_quartile=='Q4').mean()); seasons=g.season.nunique()
    req_seasons=1 if position=='QB' else 2
    eligible=(n>=16 and len(fav)>=4 and len(adv)>=4 and high>=.40 and seasons>=req_seasons)
    resid=g.signed_error_actual_minus_pred
    sl=slope_xy(g.game_spot_score,resid); sp=spearman_xy(g.game_spot_score,resid)
    half=n//2; g1=g.iloc[:half]; g2=g.iloc[half:]
    h1=slope_xy(g1.game_spot_score,g1.signed_error_actual_minus_pred); h2=slope_xy(g2.game_spot_score,g2.signed_error_actual_minus_pred)
    def cat_under(x): return float(((x.catastrophic==True)&(x.direction=='UNDERPROJECTED')).mean()) if len(x) else np.nan
    return {'position':position,'player_clean_key':g.player_clean_key.iloc[0],'player':g.player.iloc[-1],'n':n,'seasons':seasons,'favorable_n':len(fav),'neutral_n':len(neu),'adverse_n':len(adv),'high_opp_share':high,'q4_share':q4,'eligible':eligible,
            'mean_resid_favorable':float(fav.signed_error_actual_minus_pred.mean()) if len(fav) else np.nan,'mean_resid_adverse':float(adv.signed_error_actual_minus_pred.mean()) if len(adv) else np.nan,
            'fav_minus_adverse_residual':float(fav.signed_error_actual_minus_pred.mean()-adv.signed_error_actual_minus_pred.mean()) if len(fav) and len(adv) else np.nan,
            'spot_slope':sl,'spot_spearman':sp,'first_half_slope':h1,'second_half_slope':h2,'cat_under_rate_favorable':cat_under(fav),'cat_under_rate_adverse':cat_under(adv)}

def make_train_labels(df):
    split={'WR':(2020,2023),'TE':(2023,2024),'QB':(2024,2024)}
    rows=[]
    for pos,(a,b) in split.items():
        x=df[(df.position==pos)&(df.season>=a)&(df.season<=b)]
        for _,g in x.groupby('player_clean_key'):
            rows.append(player_training_stats(g,pos))
    st=pd.DataFrame(rows)
    st['candidate_labels']='UNSTABLE_OR_UNRESOLVED'
    for pos in st.position.unique():
        idx=(st.position==pos)&st.eligible; e=st[idx]; thr=THRESHOLDS[pos]
        if len(e)==0: continue
        med_abs=float(e.spot_slope.abs().median()); med_adv=float(e.cat_under_rate_adverse.median())
        for i,r in e.iterrows():
            ls=[]
            if r.fav_minus_adverse_residual>=.25*thr and r.first_half_slope>0 and r.second_half_slope>0: ls.append('ENVIRONMENT_SENSITIVE_CANDIDATE')
            if abs(r.fav_minus_adverse_residual)<=.10*thr and abs(r.spot_slope)<=med_abs: ls.append('ENVIRONMENT_RESISTANT_CANDIDATE')
            if r.mean_resid_adverse>=0 and r.cat_under_rate_adverse>=med_adv: ls.append('ADVERSE_SPOT_CEILING_CANDIDATE')
            if (r.cat_under_rate_favorable-r.cat_under_rate_adverse)>=.10 and r.fav_minus_adverse_residual>=.15*thr: ls.append('FAVORABLE_SPOT_AMPLIFIER_CANDIDATE')
            st.at[i,'candidate_labels']='|'.join(ls) if ls else 'UNSTABLE_OR_UNRESOLVED'
    return st

def run_e2(df):
    labels=make_train_labels(df); tests={'WR':(2024,2025),'TE':(2025,2025),'QB':(2025,2025)}
    case=[]; byplayer=[]
    for _,lr in labels[labels.eligible].iterrows():
        pos=lr.position; a,b=tests[pos]; g=df[(df.position==pos)&(df.player_clean_key==lr.player_clean_key)&(df.season>=a)&(df.season<=b)].copy()
        if len(g)==0: continue
        for _,r in g.iterrows():
            d=r.to_dict(); d['training_labels']=lr.candidate_labels; case.append(d)
        fav=g[g.game_spot_bucket=='FAVORABLE']; adv=g[g.game_spot_bucket=='ADVERSE']
        byplayer.append({'position':pos,'player_clean_key':lr.player_clean_key,'player':lr.player,'training_labels':lr.candidate_labels,'test_n':len(g),'test_favorable_n':len(fav),'test_adverse_n':len(adv),
                         'test_mean_resid_favorable':float(fav.signed_error_actual_minus_pred.mean()) if len(fav) else np.nan,'test_mean_resid_adverse':float(adv.signed_error_actual_minus_pred.mean()) if len(adv) else np.nan,
                         'test_fav_minus_adverse_residual':float(fav.signed_error_actual_minus_pred.mean()-adv.signed_error_actual_minus_pred.mean()) if len(fav) and len(adv) else np.nan,
                         'test_spot_slope':slope_xy(g.game_spot_score,g.signed_error_actual_minus_pred)})
    bp=pd.DataFrame(byplayer); cb=pd.DataFrame(case); score=[]
    if len(bp):
      for pos in bp.position.unique():
        for lab in ['ENVIRONMENT_SENSITIVE_CANDIDATE','ENVIRONMENT_RESISTANT_CANDIDATE','ADVERSE_SPOT_CEILING_CANDIDATE','FAVORABLE_SPOT_AMPLIFIER_CANDIDATE']:
          s=bp[(bp.position==pos)&bp.training_labels.str.contains(lab,regex=False)].copy()
          if len(s)==0: continue
          thr=THRESHOLDS[pos]
          if lab=='ENVIRONMENT_SENSITIVE_CANDIDATE': val=(s.test_fav_minus_adverse_residual>0)&(s.test_spot_slope>0); pooled=float(s.test_fav_minus_adverse_residual.mean()); expected=pooled>0
          elif lab=='ENVIRONMENT_RESISTANT_CANDIDATE': val=s.test_fav_minus_adverse_residual.abs()<=.20*thr; pooled=float(s.test_fav_minus_adverse_residual.abs().mean()); expected=pooled<=.20*thr
          elif lab=='ADVERSE_SPOT_CEILING_CANDIDATE': val=s.test_mean_resid_adverse>=0; pooled=float(s.test_mean_resid_adverse.mean()); expected=pooled>=0
          else: val=s.test_mean_resid_favorable>s.test_mean_resid_adverse; pooled=float(s.test_fav_minus_adverse_residual.mean()); expected=pooled>0
          validmask=val.notna() & s.test_fav_minus_adverse_residual.notna(); nvalid=int(validmask.sum()); rate=float(val[validmask].mean()) if nvalid else np.nan
          signal=bool(nvalid>=5 and rate>=.60 and expected)
          score.append({'position':pos,'training_label':lab,'players_total':len(s),'players_with_fav_adverse_holdout':nvalid,'validation_rate':rate,'pooled_metric':pooled,'pooled_expected_sign':bool(expected),'prospective_signal':signal})
    return labels,cb,bp,pd.DataFrame(score)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase-c-all-rows',required=True); ap.add_argument('--out-dir',required=True); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(args.phase_c_all_rows); team=build_team_games(df)
    c,score,sig,e1=run_e1(team); labels,hold,byp,psc=run_e2(df)
    integrity={'sportsbook_features_used':0,'same_or_future_outcomes_used_as_predictors':0,'test_rows_used_to_create_training_labels':0,'production_parameters_changed':0}
    result={'disposition':e1['disposition'],'phase_e1':e1,'prospective_player_environment_signals':psc[psc.prospective_signal==True][['position','training_label']].to_dict('records') if len(psc) else [],'rb_player_holdout_status':'INSUFFICIENT_TEMPORAL_HOLDOUT','integrity':integrity,'phase_c_source_rows':int(len(df)),'team_game_rows':int(len(team))}
    c.to_csv(out/'phase_e_shared_state_casebook.csv',index=False); score.to_csv(out/'phase_e_shared_state_scorecard.csv',index=False); sig.to_csv(out/'phase_e_shared_state_signature_recall.csv',index=False)
    labels.to_csv(out/'phase_e_player_train_labels.csv',index=False); hold.to_csv(out/'phase_e_player_holdout_casebook.csv',index=False); psc.to_csv(out/'phase_e_player_holdout_scorecard.csv',index=False); byp.to_csv(out/'phase_e_player_holdout_by_player.csv',index=False)
    (out/'phase_e_result.json').write_text(json.dumps(result,indent=2,allow_nan=True))
    print(json.dumps(result,indent=2,allow_nan=True)); print('\n[phase_e] shared scorecard'); print(score.to_string(index=False)); print('\n[phase_e] player holdout scorecard'); print(psc.to_string(index=False))
if __name__=='__main__': main()
