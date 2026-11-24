import pandas as pd, numpy as np, pickle
SCR='/tmp/claude-0/-home-user-imtiredofthis/1701f8f3-afc9-57ea-bbc6-cce4dc6b5fbd/scratchpad'
TNF='74d39e19225caf211ad2c8eb4110179f'
RNG=np.random.default_rng(7); N=200_000
MK={'player_reception_yds_alternate':'rec_yards','player_rush_yds_alternate':'rush_yards',
    'player_pass_yds_alternate':'pass_yards','player_rush_reception_yds_alternate':'rush_rec_yards'}
STAT={'rec_yards':'receiving_yards','rush_yards':'rushing_yards',
      'pass_yards':'passing_yards','rush_rec_yards':'rush_rec_yards'}
FAM={'rec_yards':('WR','TE'),'rush_yards':('RB',),'rush_rec_yards':('RB',),'pass_yards':('QB',)}
nmk=lambda p:''.join(c for c in str(p).lower() if c.isalpha())
def ip(x): x=float(x); return (-x)/((-x)+100) if x<0 else 100/(x+100)

# ---------- 1. ENVIRONMENT ----------
g=pd.read_csv('data/market_track_record/game_odds/2026_wk10_part3.csv')
tot=g[g.market=='totals'].groupby('event_id').point.first()
LEAGUE_TOTAL=float(tot.mean()); GAME_TOTAL=float(tot.loc[TNF])
sp=g[(g.event_id==TNF)&(g.market=='spreads')].groupby('name').point.first()
print(f"ENVIRONMENT  game total {GAME_TOTAL}  week-10 mean total {LEAGUE_TOTAL:.1f}"
      f"  ratio {GAME_TOTAL/LEAGUE_TOTAL:.3f}   spread NYG {sp.get('New York Giants')}")

# ---------- 2. MATCHUP ----------
st=pd.read_csv(f'{SCR}/w2026_stats.csv',low_memory=False)
st['rush_rec_yards']=st.rushing_yards.fillna(0)+st.receiving_yards.fillna(0)
st['nm']=st.player_display_name.astype(str).str.lower().str.replace(r'[^a-z]','',regex=True)
pos=st.groupby('nm').position.first(); st['fam']=st.nm.map(pos)
def def_factor(opp, fam):
    col={'WR':'receiving_yards','TE':'receiving_yards','RB':'rushing_yards','QB':'passing_yards'}[fam[0]]
    sub=st[st.fam.isin(fam)]
    pg=sub.groupby(['opponent_team','week'])[col].sum().groupby('opponent_team').mean()
    if opp not in pg.index or pg.mean()==0: return 1.0, np.nan, np.nan
    return float(pg.loc[opp]/pg.mean()), float(pg.loc[opp]), float(pg.mean())

# ---------- 3. SOURCES ----------
with open(f'{SCR}/tnf/draws.pkl','rb') as f: D=pickle.load(f)
alt=pd.read_csv('data/market_track_record/alt_props/2026_wk10_thu6.csv')
a=alt[(alt.event_id==TNF)&(alt.side.astype(str).str.upper()=='OVER')].copy()
a['market']=a.market.map(MK); a=a[a.market.notna()]
board=pd.read_csv('data/market_track_record/boards/2026_wk10.csv')

def hold(pl,mk):
    """Two-sided hold from the MAIN line, used to de-vig the one-sided ladder."""
    r=board[(board.player==pl)&(board.market==mk)]
    if r.empty or 'vegas_over_odds' not in r.columns: return 1.0
    r=r.iloc[0]
    try: s=ip(r.vegas_over_odds)+ip(r.vegas_under_odds)
    except Exception: return 1.0
    return s if s>0.9 else 1.0

def market_cdf(pl,mk):
    """De-vigged market survival function from the WHOLE ladder, monotone."""
    s=a[(a.player==pl)&(a.market==mk)].sort_values('line')
    if s.empty: return None
    h=hold(pl,mk)
    x=s.groupby('line').price_american.max()
    p=(x.map(ip)/h).clip(0.001,0.999).values
    L=x.index.values.astype(float)
    p=np.minimum.accumulate(p)                      # enforce monotonicity
    return L,p,h

def own_adjusted(pl,mk,opp):
    v=st[(st.nm==nmk(pl))&(st.season==2026)][STAT[mk]].dropna().astype(float).values
    if len(v)<4: return None,None,None
    df,_,_=def_factor(opp,FAM[mk])
    env=GAME_TOTAL/LEAGUE_TOTAL
    # Own history is opponent- and environment-NAIVE, so it is the one source
    # that must be adjusted. Model and market already price both.
    scale=env*df
    return v*scale, scale, df

rows=[]
W_MKT,W_MODEL,W_OWN=0.45,0.25,0.30
for (pl,mk),v in D.items():
    mc=market_cdf(pl,mk)
    if mc is None: continue
    L,pmkt,h=mc
    opp=v['opp']
    ov,scale,df=own_adjusted(pl,mk,opp)
    if ov is None: continue
    draws=v['draws']
    for thr,pm in zip(L,pmkt):
        p_model=float((draws>thr).mean())
        p_own=float((ov>thr).mean())
        n=len(ov)
        blend=W_MKT*pm+W_MODEL*p_model+W_OWN*p_own
        px=float(a[(a.player==pl)&(a.market==mk)&(a.line==thr)].price_american.max())
        d=(1+px/100) if px>0 else (1+100/-px)
        rows.append(dict(player=pl,market=mk,thr=thr,price=px,
            p_mkt_devig=pm,p_model=p_model,p_own=p_own,own_n=n,
            env=GAME_TOTAL/LEAGUE_TOTAL,matchup=df,scale=scale,
            p_blend=blend,dec=d,breakeven=1/d,ev=blend*d-1,
            spread=max(p_mkt for p_mkt in [pm])-min([p_model,p_own]),
            disagree=max(pm,p_model,p_own)-min(pm,p_model,p_own)))
B=pd.DataFrame(rows)
B.to_csv(f'{SCR}/tnf/blended.csv',index=False)
print(f"blended rungs: {len(B)}  players: {B.player.nunique()}")
print()
print("=== MATCHUP + ENVIRONMENT applied to own history ===")
m=B.drop_duplicates(['player','market'])[['player','market','matchup','env','scale','own_n']]
print(m.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
