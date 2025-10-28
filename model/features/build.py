import pandas as pd, numpy as np

def _nfill(series, default):
    return pd.to_numeric(series, errors='coerce').fillna(default)

def build_features(m: pd.DataFrame, team_frame: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    # Join offense env (by team) and defense env (by opponent)
    off_env = team_frame.set_index('team').add_prefix('off_')
    def_env = team_frame.set_index('team').add_prefix('def_')
    m = m.join(off_env, on='team', how='left')
    m = m.join(def_env, on='opponent', how='left')

    # PROE & plays (neutral baseline)
    cand_pass = [c for c in ['off_proe_neutral','off_neutral_db_rate','off_pass_rate','off_pass_rate_overall','off_db_rate'] if c in m.columns]
    base_pass = _nfill(m[cand_pass[0]] if cand_pass else pd.Series(np.nan, index=m.index), 0.56)
    m['off_pass_rate_final'] = np.clip(base_pass, 0.40, 0.70)

    cand_plays = [c for c in ['off_plays_est','off_plays_per_game','off_plays','off_expected_plays'] if c in m.columns]
    m['off_plays_est_final'] = _nfill(cand_plays and m[cand_plays[0]] or pd.Series(np.nan, index=m.index), 62.0)

    m['off_rush_rate_final'] = 1.0 - m['off_pass_rate_final']

    # Defensive z’s
    def getfirst(cols, default=0.0):
        for c in cols:
            if c in m.columns: return _nfill(m[c], default)
        return pd.Series(default, index=m.index)

    m['def_press_z']     = getfirst(['def_def_pressure_rate_z','def_press_rate_z','def_pressure_z'])
    m['def_pass_epa_z']  = getfirst(['def_def_pass_epa_z','def_pass_epa_z','def_epa_pass_z'])
    m['def_sack_z']      = getfirst(['def_def_sack_rate_z','def_sack_rate_z'])
    m['def_light_box_z'] = getfirst(['def_light_box_rate_z','def_light_box_z'])
    m['def_heavy_box_z'] = getfirst(['def_heavy_box_rate_z','def_heavy_box_z'])

    # Player priors / fills
    for c, d in [('route_rate',0.70), ('tgt_share',0.18), ('receptions_per_target',0.65),
                 ('yprr',np.nan), ('ypt',np.nan), ('rush_share',0.25), ('ypc',4.2), ('ypa',7.1),
                 ('rz_tgt_share',np.nan), ('rz_rush_share',np.nan), ('rz_share',np.nan)]:
        if c not in m.columns: m[c] = np.nan
        m[c] = _nfill(m[c], d)

    m['ypt'] = m['ypt'].fillna(m['yprr']*1.1).fillna(7.5)
    m['rz_share'] = m['rz_share'].fillna(pd.concat([m['rz_tgt_share'], m['rz_rush_share']], axis=1).max(axis=1)).fillna(0.22)

    # -----------------------
    # Coverage & WR exposure
    # -----------------------
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

    # Optional alignment exposure file
    wr_exp = getattr(m, "_wr_exp", pd.DataFrame())
    if isinstance(wr_exp, pd.DataFrame) and not wr_exp.empty:
        wr_exp['team'] = wr_exp.get('team','').astype(str).str.upper().str.strip()
        wr_exp_map = { (r['team'], str(r.get('player_pf',''))): r for _, r in wr_exp.iterrows() }
    else:
        wr_exp_map = {}

    def apply_cov(row):
        k = (str(row.get('team','')).upper(), str(row.get('player_pf','')))
        # coverage shadow/slot
        if k in role_pen and str(row.get('position','')).upper() == 'WR':
            role, pct, shadow = role_pen[k]
            fac = 1.0 + pct/100.0
            if role in ('wr1','wr2'):
                row['tgt_share'] = max(0.0, row['tgt_share'] * fac)
                if shadow: row['ypt'] = max(0.1, row['ypt'] * 0.94)
            else:
                row['tgt_share'] *= fac
                row['ypt'] *= 1.04

        # alignment exposure (slot/wide)
        exp = wr_exp_map.get(k)
        if exp is not None:
            slot_pct = float(pd.to_numeric(exp.get('slot_pct', np.nan), errors='coerce') or np.nan)
            if not np.isnan(slot_pct):
                # mild slot boost in heavy-zone concepts
                row['ypt'] *= (1.0 + 0.04*(slot_pct >= 0.5))
        return row

    m = m.apply(apply_cov, axis=1)

    # -----------------------
    # Weather (wind/precip/surface)
    # -----------------------
    wx = getattr(m, "_wx", pd.DataFrame())
    if isinstance(wx, pd.DataFrame) and not wx.empty:
        cols = [c for c in ['event_id','wind_mph','temp_f','precip','surface','dome'] if c in wx.columns]
        if 'event_id' in cols and 'event_id' in m.columns:
            m = m.merge(wx[cols], on='event_id', how='left')
        else:
            # fall back to team join (we expect duplicates, but last-merge wins)
            pass

        # penalties
        wind = pd.to_numeric(m.get('wind_mph', np.nan), errors='coerce')
        precip = m.get('precip','').astype(str).str.lower()
        dome = m.get('dome','').astype(str).str.lower().isin(['1','true','yes','y'])

        # Wind >= 15mph: pass/rec efficiency down a touch
        wind_pen = (~dome) & (wind >= 15)
        m.loc[wind_pen, 'ypt'] *= 0.94
        m.loc[wind_pen, 'ypa'] = _nfill(m['ypa'], 7.1) * 0.96

        # Rain/Snow: lower YAC → rec down, rush rate up a touch
        precip_pen = (~dome) & precip.isin(['rain','snow'])
        m.loc[precip_pen, 'ypt'] *= 0.97
        m.loc[precip_pen, 'off_rush_rate_final'] = np.clip(m['off_rush_rate_final'] + 0.02, 0.25, 0.80)
        m['off_pass_rate_final'] = 1.0 - m['off_rush_rate_final']

    # -----------------------
    # QB mobility (scramble, designed)
    # -----------------------
    qb_mob = getattr(m, "_qb_mob", pd.DataFrame())
    if isinstance(qb_mob, pd.DataFrame) and not qb_mob.empty:
        qb_mob['team'] = qb_mob.get('team','').astype(str).str.upper().str.strip()
        m = m.merge(qb_mob[['player','team','scramble_rate','designed_rush_share']].rename(columns={'player':'player_pf'}),
                    on=['player_pf','team'], how='left')
        srate = _nfill(m['scramble_rate'], 0.06)  # default ~6%
        m['off_pass_rate_final'] = np.clip(m['off_pass_rate_final'] * (1 - 0.35*m['def_press_z']), 0.35, 0.75)
        # Dropbacks reduced by scramble
        # (the scramble share diverts some dropbacks away from attempts)
        # We keep it simple here because the pricing layer uses μ/σ on yards outcomes
        # If you later add QB rush yards market, reuse designed_rush_share there.
    else:
        srate = pd.Series(0.06, index=m.index)

    # Core volume/efficiency after all mods
    m['dropbacks']   = m['off_plays_est_final']*m['off_pass_rate_final']*(1 - 0.15*m['def_sack_z'].clip(lower=0))
    m['dropbacks']  *= (1 - 0.30*srate.clip(0,0.25))  # scramble elasticity
    m['rush_team']   = m['off_plays_est_final']*(1 - m['off_pass_rate_final'])

    m['ypc_adj']     = m['ypc']*(1 + 0.07*(m['def_light_box_z']>0.6) - 0.06*(m['def_heavy_box_z']>0.6))
    m['ypa_adj']     = m['ypa']*(1 - 0.35*m['def_press_z'])*(1 - 0.25*m['def_pass_epa_z'])

    m['targets_mu']     = (m['route_rate']*m['tgt_share']*m['dropbacks']).clip(lower=0)
    m['carries_mu']     = (m['rush_share']*m['rush_team']).clip(lower=0)
    m['rec_yards_mu']   = m['targets_mu']*m['ypt']
    m['receptions_mu']  = m['targets_mu']*m['receptions_per_target']
    m['rush_yards_mu']  = m['carries_mu']*m['ypc_adj']
    m['pass_yards_mu']  = (m['off_plays_est_final']*m['off_pass_rate_final']*0.93)*m['ypa_adj']

    # SDs with volatility widening
    m['rec_yards_sd'] = 26.0*(1 + 0.10*(m['def_press_z'].abs()>1.0))
    m['receptions_sd'] = 1.9
    m['rush_yards_sd'] = 23.0*(1 + 0.08*(m['def_heavy_box_z'].abs()>1.0))
    m['pass_yards_sd'] = 48.0*(1 + 0.15*(m['def_press_z'].abs()>1.0))

    return m
