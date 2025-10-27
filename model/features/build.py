import pandas as pd, numpy as np

def _nfill(series, default):
    return pd.to_numeric(series, errors='coerce').fillna(default)

def build_features(m, team_frame, cov):
    off_env = team_frame.set_index('team').add_prefix('off_')
    def_env = team_frame.set_index('team').add_prefix('def_')
    m = m.join(off_env, on='team', how='left')
    m = m.join(def_env, on='opponent', how='left')

    cand_pass = [c for c in ['off_neutral_db_rate','off_pass_rate','off_pass_rate_overall','off_db_rate'] if c in m.columns]
    m['off_pass_rate_final'] = _nfill(m[cand_pass[0]] if cand_pass else pd.Series(np.nan, index=m.index), 0.56)
    cand_plays = [c for c in ['off_plays_est','off_plays_per_game','off_plays'] if c in m.columns]
    m['off_plays_est_final'] = _nfill(m[cand_plays[0]] if cand_plays else pd.Series(np.nan, index=m.index), 62.0)
    m['off_rush_rate_final'] = 1.0 - m['off_pass_rate_final']

    def getfirst(cols, default=0.0):
        for c in cols:
            if c in m.columns: return _nfill(m[c], default)
        return pd.Series(default, index=m.index)

    m['def_press_z'] = getfirst(['def_def_pressure_rate_z','def_press_rate_z','def_pressure_z'])
    m['def_pass_epa_z'] = getfirst(['def_def_pass_epa_z','def_pass_epa_z','def_epa_pass_z'])
    m['def_sack_z'] = getfirst(['def_def_sack_rate_z','def_sack_rate_z'])
    m['def_light_box_z'] = getfirst(['def_light_box_rate_z','def_light_box_z'])
    m['def_heavy_box_z'] = getfirst(['def_heavy_box_rate_z','def_heavy_box_z'])

    for c, d in [('route_rate',0.70), ('tgt_share',0.18), ('receptions_per_target',0.65),
                 ('yprr',np.nan), ('ypt',np.nan), ('rush_share',0.25), ('ypc',4.2), ('ypa',7.1),
                 ('rz_tgt_share',np.nan), ('rz_rush_share',np.nan), ('rz_share',np.nan)]:
        if c not in m.columns: m[c] = np.nan
        m[c] = _nfill(m[c], d)

    m['ypt'] = m['ypt'].fillna(m['yprr']*1.1).fillna(7.5)
    m['rz_share'] = m['rz_share'].fillna(pd.concat([m['rz_tgt_share'], m['rz_rush_share']], axis=1).max(axis=1)).fillna(0.22)

    role_pen = {}
    for _, r in cov.iterrows():
        off = str(r.get('offense_team','')).upper()
        wr1, wr2, slot = str(r.get('wr1','')), str(r.get('wr2','')), str(r.get('slot_wr',''))
        wr1_pen = float(pd.to_numeric(r.get('wr1_penalty_pct',0), errors='coerce') or 0.0)
        wr2_pen = float(pd.to_numeric(r.get('wr2_penalty_pct',0), errors='coerce') or 0.0)
        slot_boost = float(pd.to_numeric(r.get('slot_boost_pct',0), errors='coerce') or 0.0)
        shadow_flag = str(r.get('shadow_flag','')).strip().lower() == 'yes'
        if wr1: role_pen[(off, wr1)] = ('wr1', wr1_pen, shadow_flag)
        if wr2: role_pen[(off, wr2)] = ('wr2', wr2_pen, shadow_flag)
        if slot: role_pen[(off, slot)] = ('slot', slot_boost, False)

    def apply_cov(row):
        k = (str(row.get('team','')).upper(), str(row.get('player_pf','')))
        if k in role_pen and str(row.get('position','')).upper()=='WR':
            role, pct, shadow = role_pen[k]
            fac = 1.0 + pct/100.0
            if role in ('wr1','wr2'):
                row['tgt_share'] = max(0.0, row['tgt_share'] * fac)
                if shadow: row['ypt'] = max(0.1, row['ypt'] * 0.94)
            else:
                row['tgt_share'] = row['tgt_share'] * fac
                row['ypt'] = row['ypt'] * 1.04
        return row
    m = m.apply(apply_cov, axis=1)

    m['dropbacks'] = m['off_plays_est_final']*m['off_pass_rate_final']*(1 - 0.15*m['def_sack_z'].clip(lower=0))
    m['rush_team'] = m['off_plays_est_final']*m['off_rush_rate_final']
    m['ypc_adj'] = m['ypc']*(1 + 0.07*(m['def_light_box_z']>0.6) - 0.06*(m['def_heavy_box_z']>0.6))
    m['ypa_adj'] = m['ypa']*(1 - 0.35*m['def_press_z'])*(1 - 0.25*m['def_pass_epa_z'])

    m['targets_mu'] = (m['route_rate']*m['tgt_share']*m['dropbacks']).clip(lower=0)
    m['carries_mu'] = (m['rush_share']*m['rush_team']).clip(lower=0)
    m['rec_yards_mu'] = m['targets_mu']*m['ypt']
    m['receptions_mu'] = m['targets_mu']*m['receptions_per_target']
    m['rush_yards_mu'] = m['carries_mu']*m['ypc_adj']
    m['pass_yards_mu'] = (m['off_plays_est_final']*m['off_pass_rate_final']*0.93)*m['ypa_adj']

    m['rec_yards_sd'] = 26.0*(1 + 0.10*(m['def_press_z'].abs()>1.0))
    m['receptions_sd'] = 1.9
    m['rush_yards_sd'] = 23.0*(1 + 0.08*(m['def_heavy_box_z'].abs()>1.0))
    m['pass_yards_sd'] = 48.0*(1 + 0.15*(m['def_press_z'].abs()>1.0))

    return m
