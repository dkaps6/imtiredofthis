#!/usr/bin/env python3
import subprocess, sys, os

BUILD_SCRIPTS = [
    "scripts/build/build_cb_coverage_team.py",
    "scripts/build/build_cb_coverage_player.py",
    "scripts/build/build_weather_week.py",
    "scripts/build/build_injuries_weekly.py",
    "scripts/build/build_qb_run_metrics.py",
    "scripts/build/build_wr_cb_exposure.py",
    "scripts/build/build_play_volume_splits.py",
    "scripts/build/build_volatility_widening.py",
    "scripts/build/build_run_pass_funnel.py",
    "scripts/build/build_coverage_penalties.py",
    "scripts/build/build_script_escalators.py",
    "scripts/build/build_opponent_map_from_props.py",
]

def run(cmd):
    print(f"\n$ {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[WARN] step failed: {cmd}", file=sys.stderr)

def main():
    os.makedirs("data", exist_ok=True)
    for path in BUILD_SCRIPTS:
        run(f"python {path}")

    # Fix opponents in player_form_consensus in place
    if os.path.exists("data/opponent_map_from_props.csv") and os.path.exists("data/player_form_consensus.csv"):
        run("python scripts/util/merge_opponent_into_player_form.py")

    # OPTIONAL: if you’re using the /model pipeline we added
    if os.path.exists("model/features/build.py"):
        run('python -c "from model.features.build import build_matchup_frame as bm; df=bm(); df.to_csv(\'outputs/matchup_features.csv\', index=False); print(df.shape, \'-> outputs/matchup_features.csv\')"')

if __name__ == "__main__":
    main()
