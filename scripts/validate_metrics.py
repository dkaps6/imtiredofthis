# scripts/validate_metrics.py
import sys
import pandas as pd

REQUIRED_TEAM_COLS = [
    "team","season","def_pass_epa","def_rush_epa","def_sack_rate",
    "pace","proe","light_box_rate","heavy_box_rate","ay_per_att"
]
REQUIRED_PLAYER_COLS = [
    "player","team","season","role","position",
    "tgt_share","route_rate","rush_share","yprr","ypt","ypc","ypa","rz_share"
]
SEASON = 2025
TEAM_MIN_COVERAGE = 1.00  # 100%
PLAYER_MIN_COVERAGE = 1.00

def coverage(df: pd.DataFrame, cols):
    if df.empty: return 0.0
    sub = df[cols]
    return float(sub.notna().all(axis=1).mean())  # fraction of rows with all fields non-null

def main():
    tf = pd.read_csv("data/team_form.csv")
    pf = pd.read_csv("data/player_form.csv")

    # enforce season
    if "season" in tf.columns: tf = tf[tf["season"] == SEASON]
    if "season" in pf.columns: pf = pf[pf["season"] == SEASON]

    # missing columns?
    missing_t = [c for c in REQUIRED_TEAM_COLS if c not in tf.columns]
    missing_p = [c for c in REQUIRED_PLAYER_COLS if c not in pf.columns]
    errs = []
    if missing_t:
        errs.append(f"team_form.csv missing columns: {missing_t}")
    if missing_p:
        errs.append(f"player_form.csv missing columns: {missing_p}")

    if errs:
        print("[validate_metrics] ❌ Column errors:\n  - " + "\n  - ".join(errs), file=sys.stderr)
        sys.exit(1)

    team_cov = coverage(tf, REQUIRED_TEAM_COLS)
    plyr_cov = coverage(pf, REQUIRED_PLAYER_COLS)

    if team_cov < TEAM_MIN_COVERAGE or plyr_cov < PLAYER_MIN_COVERAGE:
        print(f"[validate_metrics] ❌ Coverage too low. team={team_cov:.3f}, player={plyr_cov:.3f}", file=sys.stderr)
        # optional: print examples of bad rows
        bad_t = tf[~tf[REQUIRED_TEAM_COLS].notna().all(axis=1)].head(10)
        bad_p = pf[~pf[REQUIRED_PLAYER_COLS].notna().all(axis=1)].head(10)
        print("[validate_metrics] team bad rows (sample):\n", bad_t, file=sys.stderr)
        print("[validate_metrics] player bad rows (sample):\n", bad_p, file=sys.stderr)
        sys.exit(1)

    print(f"[validate_metrics] ✅ OK (team={team_cov:.3f}, player={plyr_cov:.3f})")

if __name__ == "__main__":
    main()
