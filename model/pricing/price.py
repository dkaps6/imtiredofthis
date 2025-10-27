import pandas as pd, numpy as np
from math import erf, sqrt

def american_to_prob(odds):
    try: o=float(odds)
    except: return np.nan
    return 100.0/(o+100.0) if o>0 else -o/(-o+100.0)

def american_to_decimal(odds):
    o=float(odds); return 1+(o/100 if o>0 else 100/(-o))

def p_over(line, mu, sd):
    if pd.isna(mu) or pd.isna(sd) or sd<=0: return np.nan
    z=(line-mu)/sd
    return 1 - 0.5*(1+erf(z/np.sqrt(2)))

def price_all(m):
    dev = m.assign(price_prob=m['price_american'].apply(american_to_prob))
    pv = dev.pivot_table(index=['event_id','player','market','line'], columns='side', values='price_prob', aggfunc='first')

    def fair_probs(r):
        key=(r['event_id'], r['player'], r['market'], r['line'])
        pO = pv.loc[key]['OVER'] if key in pv.index and 'OVER' in pv.columns else np.nan
        pU = pv.loc[key]['UNDER'] if key in pv.index and 'UNDER' in pv.columns else np.nan
        if pd.notna(pO) and pd.notna(pU) and (pO+pU)>0:
            return pO/(pO+pU), pU/(pO+pU)
        if str(r['side']).upper()=='OVER':
            pO=american_to_prob(r['price_american']); return pO, 1-pO
        else:
            pU=american_to_prob(r['price_american']); return 1-pU, pU

    rows=[]
    for _, r in m.iterrows():
        try: L=float(r['line'])
        except: continue
        mk=str(r['market']); side=str(r['side']).upper()
        mu=sd=np.nan
        if mk=='player_reception_yds':
            mu,sd=r['rec_yards_mu'], r['rec_yards_sd']
        elif mk=='player_receptions':
            mu,sd=r['receptions_mu'], r['receptions_sd']
        elif mk=='player_rush_yds':
            mu,sd=r['rush_yards_mu'], r['rush_yards_sd']
        elif mk=='player_pass_yds':
            mu,sd=r['pass_yards_mu'], r['pass_yards_sd']
        elif mk=='player_rush_reception_yds':
            mu=(r.get('rush_yards_mu') or 0)+(r.get('rec_yards_mu') or 0)
            sd=sqrt(r['rush_yards_sd']**2 + r['rec_yards_sd']**2)
        else:
            continue

        p_model_over=p_over(L, mu, sd)
        p_model = 1-p_model_over if side=='UNDER' else p_model_over
        pO,pU=fair_probs(r)
        p_mkt = pU if side=='UNDER' else pO
        if pd.isna(p_mkt): p_mkt = p_model
        p_blend = 0.65*p_model + 0.35*p_mkt
        edge = (p_blend - p_mkt)*100

        rows.append({
            'player': r.get('player_pf', r['player']),
            'position': r.get('position',''),
            'team': r['team'], 'opponent': r['opponent'],
            'market': mk, 'side': side, 'line': r['line'], 'price_american': r['price_american'],
            'model_mu': mu, 'model_sd': sd,
            'p_model': p_model, 'p_market_fair': p_mkt, 'p_blend': p_blend,
            'edge_pct_pts': edge
        })

    edges = pd.DataFrame(rows)
    edges['tier'] = pd.cut(edges['edge_pct_pts'], bins=[-np.inf,1,4,6,np.inf],
                           labels=['RED (<1%)','AMBER (1–4%)','GREEN (4–6%)','ELITE (≥6%)'])

    def kelly(p, odds_american, cap=0.05):
        d=american_to_decimal(odds_american); b=d-1; q=1-p
        frac=(p*b - q)/b
        return float(max(0.0, min(cap, frac)))

    edges['kelly_frac_cap5'] = edges.apply(lambda r: kelly(r['p_blend'], r['price_american'], 0.05), axis=1)
    return edges
