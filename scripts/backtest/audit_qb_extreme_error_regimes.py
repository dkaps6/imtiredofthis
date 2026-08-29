#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

SAFE_MARKET_FEATURES = [
    'market_spread','market_abs_spread','market_total','market_team_implied','market_opp_implied','market_is_underdog','market_moneyline',
    'market_win_probability','pregame_win_probability','expected_trailing_probability','competitive_game','total_x_trailing','spread_x_total',
    'team_neutral_pace','team_plays_est','team_dropback_rate','team_proe','team_success_rate_off','team_pressure_rate_generated','team_def_pass_epa','team_success_rate_def',
    'opp_neutral_pace','opp_plays_est','opp_dropback_rate','opp_proe','opp_success_rate_off','opp_pressure_rate_generated','opp_def_pass_epa','opp_success_rate_def',
    'opponent_force_pass','qb_recent_ypa','qb_recent_completion_pct','qb_recent_td_rate','qb_recent_int_rate','qb_recent_epa_per_att','qb_recent_pass_att',
    'team_pressure_rate_allowed','team_explosive_play_rate_allowed','team_coverage_man_rate','team_coverage_zone_rate','opp_pressure_rate_allowed','opp_explosive_play_rate_allowed',
    'opp_coverage_man_rate','opp_coverage_zone_rate','controlled_environment']
RAW_MODEL_FEATURES=['attempts_current','attempts_gamescript','pred_attempts','attempts_raw','att_raw_delta','ypa_current','ypa_contextual','pred_ypa','ypa_raw','ypa_raw_delta']
ATTEMPT_HISTORY_FEATURES=['qb_att_last1','qb_att_last3','qb_att_mean8','qb_att_std8','qb_att_iqr8','qb_att_min8','qb_att_max8','qb_att_games8','qb_att_40plus_rate8','qb_att_trend8','raw_minus_recent','abs_raw_minus_recent']
SITUATION_FEATURES=['is_home','division_game','standalone_window_proxy','team_rest_days','opp_rest_days','rest_advantage_days','short_rest','post_bye','same_season_rematch','rematch_after_loss','rematch_after_14plus_loss','prior_matchup_margin','prior_postseason_rematch','coach_change','coach_tenure_games','qb_former_opponent_recent','qb_new_team_this_season','team_games_played','team_win_pct','team_point_diff_pg','team_last3_margin','team_streak','opp_games_played','opp_win_pct','opp_point_diff_pg','opp_last3_margin','opp_streak','prior_season_win_pct','surprise_success','late_season','week15_plus','late_bubble_proxy','late_contender_proxy','desperation_proxy','spoiler_proxy','cinderella_proxy','late_division_game','division_rematch_proxy']

def num(v): return pd.to_numeric(v, errors='coerce')
def read(path):
    x=pd.read_csv(path); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def to_pd(o): return o.copy() if isinstance(o,pd.DataFrame) else o.to_pandas() if hasattr(o,'to_pandas') else pd.DataFrame(o)
def load_sched(seasons):
    import nflreadpy as nfl
    z=[]
    for s in sorted(set(map(int,seasons))):
        q=to_pd(nfl.load_schedules(s)); q.columns=[str(c).strip().lower() for c in q.columns]; z.append(q)
    return pd.concat(z,ignore_index=True,sort=False)

def perspectives(raw):
    rows=[]
    for _,r in raw.iterrows():
        sv=num(pd.Series([r.get('season')])).iloc[0]; wv=num(pd.Series([r.get('week')])).iloc[0]
        if not np.isfinite(sv) or not np.isfinite(wv): continue
        season,week=int(sv),int(wv); home,away=canon_team(r.get('home_team','')),canon_team(r.get('away_team',''))
        if not home or not away: continue
        hs=num(pd.Series([r.get('home_score')])).iloc[0]; aws=num(pd.Series([r.get('away_score')])).iloc[0]
        gt=str(r.get('game_type',r.get('season_type','')) or '').upper()
        for team,opp,is_home in [(home,away,1),(away,home,0)]:
            rows.append({'season':season,'week':week,'team':team,'opponent':opp,'game_id':str(r.get('game_id','') or ''),'game_type':gt,
                'weekday':str(r.get('weekday','') or ''),'gametime':str(r.get('gametime','') or ''),'division_game':float(bool(r.get('div_game',False))) if pd.notna(r.get('div_game')) else np.nan,
                'is_home':float(is_home),'points_for':hs if is_home else aws,'points_against':aws if is_home else hs,
                'team_rest_days':r.get('home_rest' if is_home else 'away_rest',np.nan),'opp_rest_days':r.get('away_rest' if is_home else 'home_rest',np.nan),
                'coach':str(r.get('home_coach' if is_home else 'away_coach','') or '')})
    return pd.DataFrame(rows)

def reg_before(pg,team,season,week):
    z=pg[(pg.team.eq(team))&num(pg.season).eq(season)&(num(pg.week)<week)].copy()
    if 'game_type' in z: z=z[z.game_type.astype(str).str.upper().isin(['REG','REGULAR','RS',''])]
    z['points_for']=num(z.points_for); z['points_against']=num(z.points_against)
    return z[z.points_for.notna()&z.points_against.notna()].sort_values('week')

def record(pg,team,season,week,prefix):
    g=reg_before(pg,team,season,week)
    if g.empty: return {f'{prefix}_games_played':0.,f'{prefix}_win_pct':np.nan,f'{prefix}_point_diff_pg':np.nan,f'{prefix}_last3_margin':np.nan,f'{prefix}_streak':0.}
    m=num(g.points_for)-num(g.points_against); streak=0
    for v in m.iloc[::-1]:
        s=1 if v>0 else -1 if v<0 else 0
        if streak==0: streak=s
        elif s==0 or np.sign(streak)!=s: break
        else: streak+=s
    return {f'{prefix}_games_played':float(len(g)),f'{prefix}_win_pct':float((m.gt(0).sum()+.5*m.eq(0).sum())/len(m)),f'{prefix}_point_diff_pg':float(m.mean()),f'{prefix}_last3_margin':float(m.tail(3).mean()),f'{prefix}_streak':float(streak)}

def prior_wp(pg,team,season):
    g=pg[(pg.team.eq(team))&num(pg.season).eq(season)].copy()
    if 'game_type' in g:g=g[g.game_type.astype(str).str.upper().isin(['REG','REGULAR','RS',''])]
    m=(num(g.points_for)-num(g.points_against)).dropna(); return np.nan if m.empty else float((m.gt(0).sum()+.5*m.eq(0).sum())/len(m))
def hour(s):
    s=str(s or '').strip().lower()
    try:
        h=int(s.split(':',1)[0]); h=h+12 if 'pm' in s and h<12 else 0 if 'am' in s and h==12 else h; return h
    except:return None

def situation(pg,row,logs):
    season,week=int(row.season),int(row.week); team,opp,key=canon_team(row.team),canon_team(row.opponent),str(row.player_clean_key)
    cur=pg[(num(pg.season).eq(season))&num(pg.week).eq(week)&pg.team.eq(team)]
    cur=cur.iloc[0] if len(cur) else pd.Series(dtype=object)
    d=record(pg,team,season,week,'team'); d.update(record(pg,opp,season,week,'opp'))
    pwp=prior_wp(pg,team,season-1); d['prior_season_win_pct']=pwp; d['surprise_success']=d['team_win_pct']-pwp if np.isfinite(d['team_win_pct']) and np.isfinite(pwp) else np.nan
    rest=num(pd.Series([cur.get('team_rest_days',np.nan)])).iloc[0]; orest=num(pd.Series([cur.get('opp_rest_days',np.nan)])).iloc[0]
    wd=str(cur.get('weekday','') or '').lower(); hh=hour(cur.get('gametime','')); standalone=wd.startswith(('mon','thu','sat')) or (wd.startswith('sun') and hh is not None and hh>=20)
    prev=reg_before(pg,team,season,week); pm=prev[prev.opponent.eq(opp)]
    prior_margin=float(num(pd.Series([pm.iloc[-1].points_for])).iloc[0]-num(pd.Series([pm.iloc[-1].points_against])).iloc[0]) if len(pm) else np.nan
    post=pg[(num(pg.season).eq(season-1))&pg.team.eq(team)&pg.opponent.eq(opp)].copy()
    if 'game_type' in post: post=post[~post.game_type.astype(str).str.upper().isin(['REG','REGULAR','RS',''])]
    curcoach=str(cur.get('coach','') or '').strip(); prevcoach=str(prev.iloc[-1].coach if len(prev) else '').strip(); change=bool(curcoach and prevcoach and curcoach!=prevcoach)
    tenure=0
    if curcoach:
        for c in prev.coach.astype(str).iloc[::-1]:
            if c.strip()!=curcoach: break
            tenure+=1
    pl=logs.copy(); pl['season']=num(pl.season); pl['week']=num(pl.week)
    pp=pl[((pl.season<season)|((pl.season==season)&(pl.week<week)))&pl.player_clean_key.astype(str).eq(key)]
    former=float(pp.team.astype(str).map(canon_team).eq(opp).any()) if len(pp) else 0.; py=pp[pp.season.eq(season-1)]; new=float(len(py)>0 and not py.team.astype(str).map(canon_team).eq(team).all())
    tw,ow=d['team_win_pct'],d['opp_win_pct']; late=week>=12; div=bool(cur.get('division_game',False))
    d.update({'is_home':float(cur.get('is_home',np.nan)),'division_game':float(cur.get('division_game',np.nan)),'standalone_window_proxy':float(standalone),'team_rest_days':rest,'opp_rest_days':orest,
      'rest_advantage_days':rest-orest if np.isfinite(rest) and np.isfinite(orest) else np.nan,'short_rest':float(np.isfinite(rest) and rest<=6),'post_bye':float(np.isfinite(rest) and rest>=10),
      'same_season_rematch':float(len(pm)>0),'rematch_after_loss':float(np.isfinite(prior_margin) and prior_margin<0),'rematch_after_14plus_loss':float(np.isfinite(prior_margin) and prior_margin<=-14),'prior_matchup_margin':prior_margin,'prior_postseason_rematch':float(len(post)>0),
      'coach_change':float(change),'coach_tenure_games':float(tenure),'qb_former_opponent_recent':former,'qb_new_team_this_season':new,'late_season':float(late),'week15_plus':float(week>=15),
      'late_bubble_proxy':float(late and np.isfinite(tw) and .35<=tw<=.70),'late_contender_proxy':float(late and np.isfinite(tw) and tw>=.55),'desperation_proxy':float(late and np.isfinite(tw) and .30<=tw<=.60 and d['team_streak']<=-2),
      'spoiler_proxy':float(week>=13 and np.isfinite(tw) and np.isfinite(ow) and tw<=.35 and ow>=.55),'cinderella_proxy':float(d['team_games_played']>=6 and np.isfinite(tw) and np.isfinite(d['surprise_success']) and tw>=.60 and d['surprise_success']>=.15),
      'late_division_game':float(late and div),'division_rematch_proxy':float(div and len(pm)>0)})
    return d

def add_att_hist(x,logs):
    p=logs.copy(); p['season']=num(p.season); p['week']=num(p.week); ac='pass_att' if 'pass_att' in p else 'attempts'
    out=[]
    for _,r in x.iterrows():
        g=p[((p.season<int(r.season))|((p.season==int(r.season))&(p.week<int(r.week))))&p.player_clean_key.astype(str).eq(str(r.player_clean_key))].sort_values(['season','week'])
        aa=num(g[ac]).dropna().tail(8).to_numpy(float); d={}
        if len(aa): d={'qb_att_last1':aa[-1],'qb_att_last3':aa[-3:].mean(),'qb_att_mean8':aa.mean(),'qb_att_std8':aa.std(),'qb_att_iqr8':np.percentile(aa,75)-np.percentile(aa,25) if len(aa)>=2 else 0.,'qb_att_min8':aa.min(),'qb_att_max8':aa.max(),'qb_att_games8':len(aa),'qb_att_40plus_rate8':np.mean(aa>=40),'qb_att_trend8':np.polyfit(np.arange(len(aa)),aa,1)[0] if len(aa)>=2 else 0.}
        else: d={c:np.nan for c in ATTEMPT_HISTORY_FEATURES if c not in ('raw_minus_recent','abs_raw_minus_recent')}
        out.append(d)
    h=pd.DataFrame(out,index=x.index); x=pd.concat([x,h],axis=1); x['raw_minus_recent']=num(x.attempts_raw)-num(x.qb_att_mean8); x['abs_raw_minus_recent']=x.raw_minus_recent.abs(); return x

def band(v): return 'lt25' if v<25 else '25_49' if v<50 else '50_74' if v<75 else '75_99' if v<100 else '100plus'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--season',type=int,required=True); ap.add_argument('--game-attribution',type=Path,required=True); ap.add_argument('--market-trace',type=Path,required=True); ap.add_argument('--player-logs',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
    g,m,logs=read(a.game_attribution),read(a.market_trace),read(a.player_logs); g['season']=a.season
    if 'opponent' not in g:
        g=g.merge(m[['week','team','player_clean_key','opponent']].drop_duplicates(),on=['week','team','player_clean_key'],how='left',validate='one_to_one')
    safe=[c for c in SAFE_MARKET_FEATURES if c in m]; mm=m[['week','team','player_clean_key']+safe].drop_duplicates(['week','team','player_clean_key']); x=g.merge(mm,on=['week','team','player_clean_key'],how='left',validate='one_to_one',suffixes=('','_market'))
    pg=perspectives(load_sched([a.season-1,a.season])); x=pd.concat([x.reset_index(drop=True),pd.DataFrame([situation(pg,r,logs) for _,r in x.iterrows()])],axis=1); x=add_att_hist(x,logs)
    x['actual']=num(x.actual); x['raw_projection']=num(x.mc_proj_attempts_raw_only); x['pass_error']=x.raw_projection-x.actual; x['abs_pass_error']=x.pass_error.abs(); x['error_band']=x.abs_pass_error.map(band)
    x['good_lt50']=x.abs_pass_error.lt(50).astype(int); x['poor_75plus']=x.abs_pass_error.ge(75).astype(int); x['catastrophic_100plus']=x.abs_pass_error.ge(100).astype(int); x['cat_under_100plus']=x.pass_error.le(-100).astype(int); x['cat_over_100plus']=x.pass_error.ge(100).astype(int)
    apass=num(x.get('actual_pass_att',np.nan)); ay=num(x.get('actual_ypa',np.nan)); x['actual_40plus_attempts']=apass.ge(40).astype(int); x['attempt_abs_error']=(num(x.attempts_raw)-apass).abs(); x['ypa_abs_error']=(num(x.ypa_contextual)-ay).abs(); x['attempt_miss_8plus']=x.attempt_abs_error.ge(8).astype(int); x['attempt_miss_10plus']=x.attempt_abs_error.ge(10).astype(int); x['ypa_miss_1_5plus']=x.ypa_abs_error.ge(1.5).astype(int); x['ypa_miss_2plus']=x.ypa_abs_error.ge(2).astype(int)
    features=[]
    for c in RAW_MODEL_FEATURES+safe+ATTEMPT_HISTORY_FEATURES+SITUATION_FEATURES:
        if c in x and c not in features: features.append(c); x[c]=num(x[c])
    br=[]
    for b,z in x.groupby('error_band'):
        br.append({'season':a.season,'error_band':b,'n':len(z),'share':len(z)/len(x),'mae':z.abs_pass_error.mean(),'rmse':np.sqrt(np.mean(z.pass_error**2)),'bias':z.pass_error.mean(),'correlation':z.raw_projection.corr(z.actual),'error_share':z.abs_pass_error.sum()/x.abs_pass_error.sum(),'mean_actual_attempts':num(z.get('actual_pass_att',np.nan)).mean(),'mean_pred_attempts':num(z.attempts_raw).mean(),'mean_actual_ypa':num(z.get('actual_ypa',np.nan)).mean(),'mean_pred_ypa':num(z.ypa_contextual).mean()})
    sr=[]
    for f in SITUATION_FEATURES:
        if f not in x or num(x[f]).dropna().nunique()>2: continue
        for flag in (0,1):
            z=x[num(x[f]).eq(flag)]
            if len(z): sr.append({'season':a.season,'feature':f,'flag':flag,'n':len(z),'mae':z.abs_pass_error.mean(),'good_lt50_rate':z.good_lt50.mean(),'poor_75plus_rate':z.poor_75plus.mean(),'catastrophic_rate':z.catastrophic_100plus.mean(),'mean_actual_attempts':num(z.get('actual_pass_att',np.nan)).mean(),'mean_pred_attempts':num(z.attempts_raw).mean()})
    q=x.copy(); q['component_pattern']=np.select([(q.attempt_miss_8plus.eq(1)&q.ypa_miss_1_5plus.eq(1)),q.attempt_miss_8plus.eq(1),q.ypa_miss_1_5plus.eq(1)],['attempt_and_ypa','attempt_only','ypa_only'],default='neither_threshold')
    comp=q.groupby(['error_band','component_pattern']).agg(n=('actual','size'),mae=('abs_pass_error','mean'),mean_attempt_abs_error=('attempt_abs_error','mean'),mean_ypa_abs_error=('ypa_abs_error','mean')).reset_index()
    a.out_dir.mkdir(parents=True,exist_ok=True); x.to_csv(a.out_dir/f'm62_enriched_games_{a.season}.csv',index=False); pd.DataFrame({'feature':features}).to_csv(a.out_dir/f'm62_safe_feature_manifest_{a.season}.csv',index=False); pd.DataFrame(br).to_csv(a.out_dir/f'm62_error_bands_{a.season}.csv',index=False); pd.DataFrame(sr).to_csv(a.out_dir/f'm62_situational_binary_slices_{a.season}.csv',index=False); comp.to_csv(a.out_dir/f'm62_component_patterns_{a.season}.csv',index=False)
    print(pd.DataFrame(br).to_string(index=False)); print(f'[m62] season={a.season} games={len(x)} safe_features={len(features)}')
if __name__=='__main__': main()
