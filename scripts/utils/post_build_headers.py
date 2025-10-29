#!/usr/bin/env python3
from scripts.utils.io_utils import ensure_header

TARGETS = {
    'data/cb_coverage_team.csv':        ['team','man_rate','zone_rate'],
    'data/cb_coverage_player.csv':      ['player','team','slot_pct','wide_pct','man_rate','zone_rate','primary_cb','shadow_flag'],
    'data/opponent_map_from_props.csv': ['player','team','opponent','week','season','game_timestamp'],
    'data/player_form_consensus.csv':   ['player','team','week','opponent'],
    'data/weather_week.csv':            ['team','opponent','week','stadium','roof','forecast_summary','temp_f','wind_mph','precip_prob','forecast_datetime_utc'],
    'data/injuries_weekly.csv':         ['team','player','week','status','practice_status','designation','report_time_utc'],
    'data/qb_designed_runs.csv':        ['player','team','week','designed_runs','scrambles','red_zone_qb_runs'],
    'data/wr_cb_exposure.csv':          ['player','player_pf','team','opponent','week','season','game_timestamp','slot_pct','wide_pct','man_rate','zone_rate','exp_vs_man','exp_vs_zone','primary_cb','shadow_flag'],
    'data/play_volume_splits.csv':      ['team','situation','plays_per_game','seconds_per_play','no_huddle_rate'],
    'data/volatility_widening.csv':     ['team','week','stdev_margin','stdev_total','injury_load_index'],
    'data/run_pass_funnel.csv':         ['team','opp','week','rush_oe','pass_oe','rush_evs','pass_evs'],
    'data/coverage_penalties.csv':      ['team','week','def_penalties','def_penalty_yards','dpi_calls','hold_calls'],
    'data/script_escalators.csv':       ['team','lead_script_rate','trail_script_rate','neutral_script_rate'],
}

if __name__ == '__main__':
    for path, cols in TARGETS.items():
        ensure_header(path, cols)
    print('[post_build_headers] header sweep complete')
