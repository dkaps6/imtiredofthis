#!/usr/bin/env python3
# build_qb_run_metrics.py
#
# Outputs two CSVs (weekly, 2025 season by default):
#   A) qb_scramble_rates.csv  -> player,week,scramble_rate,scrambles,dropbacks
#   B) qb_designed_runs.csv   -> player,week,designed_run_rate,designed_runs,snaps
#
# Data source (free):
# - nflverse/nflfastR play-by-play (contains qb_scramble, qb_dropback, qb_kneel, play_type, rush_attempt, passer_player_name, rusher_player_name, week)
#   Docs: https://nflfastr.com/reference/fast_scraper.html  (variable: qb_scramble)
#         https://nflreadr.nflverse.com/ (data access & dictionaries)
#         https://brieger.esalq.usp.br/CRAN/web/packages/nflreadr/vignettes/dictionary_pbp.html  (qb_dropback, qb_kneel)
#
# Implementation notes:
# - We fetch the public season CSV from nflverse releases (pbp_2025.csv.gz). If that URL fails, try a small set of fallbacks.
# - QB identification for rush plays uses a pragmatic rule:
#     * Build a set of names that appear as a passer in 2025 (season-to-date).
#     * A rusher is treated as a QB if their name appears in that passer set.
# - "Snaps" proxy = dropbacks + all QB rush attempts (scrambles + designed). This aligns with QB involvement on offensive plays.
# - Designed runs exclude scrambles and kneels: rush_attempt==1 & qb_scramble!=1 & qb_kneel!=1 & rusher is QB.
#
import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025


def compute_qb_sets(df: pd.DataFrame) -> set:
    # Names that appear as passers in 2025
    qb_names = set(df["passer_player_name"].dropna().unique().tolist())
    # Also some plays record qb_name in a different field; rusher could be QB with no pass that season (rare). Keep it simple.
    return qb_names


def choose_qb_name(row, qb_names) -> str:
    # For dropbacks, passer is the QB when present; sacks sometimes lack passer, then rusher name may be NaN—leave blank
    if (row.get("qb_dropback", 0) == 1) and pd.notna(row.get("passer_player_name")):
        return row["passer_player_name"]
    # For scrambles, the rusher is the QB
    if (row.get("qb_scramble", 0) == 1) and pd.notna(row.get("rusher_player_name")):
        return row["rusher_player_name"]
    # For rush plays that are QB designed runs (not scrambles), attribute to rusher if he is a known QB
    if (row.get("rush_attempt", 0) == 1) and pd.notna(row.get("rusher_player_name")):
        name = row["rusher_player_name"]
        if name in qb_names:
            return name
    # Else unknown
    return np.nan


def main():
    pbp = get_pbp_2025()
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        raise RuntimeError("PBP fetch failed")

    # Ensure numeric flags are numeric (0/1)
    for col in ["qb_scramble", "qb_dropback", "qb_kneel", "rush_attempt", "sack"]:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0).astype(int)

    # Build QB name set and attribute QB per play
    qb_names = compute_qb_sets(pbp)
    pbp["qb_name"] = pbp.apply(lambda r: choose_qb_name(r, qb_names), axis=1)

    # ---------- A) Scramble rates ----------
    scrambles = (
        pbp[(pbp["qb_scramble"] == 1) & pd.notna(pbp["qb_name"])]
        .groupby(["qb_name", "week"])
        .size()
        .rename("scrambles")
    )
    dropbacks = (
        pbp[(pbp["qb_dropback"] == 1) & pd.notna(pbp["qb_name"])]
        .groupby(["qb_name", "week"])
        .size()
        .rename("dropbacks")
    )
    a = pd.concat([scrambles, dropbacks], axis=1).fillna(0)
    a["scramble_rate"] = (a["scrambles"] / a["dropbacks"]).replace(
        [np.inf, -np.inf], np.nan
    )
    a = a.reset_index().rename(columns={"qb_name": "player"})
    # Reorder columns and round
    a["scramble_rate"] = a["scramble_rate"].round(4)
    a = a[["player", "week", "scramble_rate", "scrambles", "dropbacks"]].sort_values(
        ["player", "week"]
    )

    # ---------- B) Designed QB runs ----------
    # QB rush attempts overall
    qb_rush = pbp[(pbp["rush_attempt"] == 1) & pd.notna(pbp["qb_name"])]
    # Designed = QB rushes that are NOT scrambles and NOT kneels
    designed = qb_rush[(qb_rush["qb_scramble"] == 0) & (qb_rush["qb_kneel"] == 0)]
    designed_runs = designed.groupby(["qb_name", "week"]).size().rename("designed_runs")
    # Snaps proxy = dropbacks + all QB rush attempts (scramble + designed)
    qb_rush_counts = qb_rush.groupby(["qb_name", "week"]).size().rename("qb_rushes")
    snaps = pd.concat([dropbacks, qb_rush_counts], axis=1).fillna(0)
    snaps["snaps"] = snaps["dropbacks"] + snaps["qb_rushes"]
    snaps = snaps["snaps"]
    b = pd.concat([designed_runs, snaps], axis=1).fillna(0)
    b["designed_run_rate"] = (b["designed_runs"] / b["snaps"]).replace(
        [np.inf, -np.inf], np.nan
    )
    b = b.reset_index().rename(columns={"qb_name": "player"})
    b["designed_run_rate"] = b["designed_run_rate"].round(4)
    b = b[
        ["player", "week", "designed_run_rate", "designed_runs", "snaps"]
    ].sort_values(["player", "week"])

    # Write outputs
    a.to_csv("qb_scramble_rates.csv", index=False)
    b.to_csv("qb_designed_runs.csv", index=False)
    print(
        f"Wrote qb_scramble_rates.csv ({len(a)} rows) and qb_designed_runs.csv ({len(b)} rows)."
    )


if __name__ == "__main__":
    main()
