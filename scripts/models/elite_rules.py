import numpy as np
import pandas as pd

def pressure_qb_adjust(mu_base, z_opp_pressure=0.0, z_opp_epa_pass=0.0):
    # Browning-type fix: push down passing vs elite pressure + pass EPA defenses
    return mu_base * (1 - 0.35*z_opp_pressure) * (1 - 0.25*z_opp_epa_pass)

def sack_to_attempts(att_base, sack_rate_above_avg=0.0):
    # Lower team pass attempts with poor protection
    return att_base * (1 - 0.15*sack_rate_above_avg)

def funnel_multiplier(pass_side=True, def_rush_epa_z=0.0, def_pass_epa_z=0.0):
    # If run funnel (rush easier; pass tougher) shift volume from pass to run; reverse for pass funnels
    if not pass_side:
        # rushing side
        if def_rush_epa_z <= -0.6 and def_pass_epa_z >= 0.6:  # opp def bad vs run, good vs pass
            return 1.04
        return 1.00
    else:
        # passing side
        if def_pass_epa_z <= -0.6 and def_rush_epa_z >= 0.6:  # opp def bad vs pass, good vs run
            return 1.04
        return 1.00

def injury_redistribution(alpha_share, wr2_share, slot_te_share, rb_share, alpha_limited=False):
    # Redistribute alpha share when limited/out: 60/30/10 to WR2/slot+TE/RB
    if not alpha_limited: return alpha_share, wr2_share, slot_te_share, rb_share
    give = alpha_share * 0.5  # cap alpha to ~50% of projection
    return alpha_share - give, wr2_share + give*0.60, slot_te_share + give*0.30, rb_share + give*0.10

def coverage_penalty(yards_per_target, target_share, tough_shadow=False, heavy_man=False, heavy_zone=False):
    ypt = yards_per_target
    ts  = target_share
    if tough_shadow or heavy_man:
        ypt *= 0.94
        ts  *= 0.92
    if heavy_zone:
        ypt *= 1.04
        ts  *= 1.06
    return ypt, ts

def airy_cap(ypr, team_ay_per_att, cap_pct=0.80, ay_threshold_z=-0.8):
    # If team air yards per att is very low, cap WR YPR to 80% of career median proxy
    if team_ay_per_att is not None and team_ay_per_att <= ay_threshold_z:
        return ypr * cap_pct
    return ypr

def boxcount_ypp_mod(ypc, light_box_share=None, heavy_box_share=None):
    if light_box_share is not None and light_box_share >= 0.60:
        ypc *= 1.07
    if heavy_box_share is not None and heavy_box_share >= 0.60:
        ypc *= 0.94
    return ypc

def script_escalators(rb_atts, qb_scrambles, win_prob=0.5):
    if win_prob >= 0.55:
        rb_atts += 3
        qb_scrambles = max(0, qb_scrambles - 3)
    return rb_atts, qb_scrambles

def pace_smoothing(plays_base, z_our_pace, z_opp_pace):
    return plays_base * (1 + 0.5*(z_our_pace + z_opp_pace))

def volatility_widen(sigma, pressure_mismatch=False, qb_inconsistent=False):
    widen = 0.0
    if pressure_mismatch: widen += 0.10
    if qb_inconsistent:   widen += 0.10
    return sigma * (1 + widen)
