from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_dynamic_min_rows(
    schedule_csv: str = "data/opponent_map_from_props.csv",
    *,
    full_season_cap: int = 80000,
    per_week_rows: int = 2500,
    floor: int = 15000,
) -> int:
    """
    Compute a reasonable minimum PBP row threshold based on the current slate week.

    Logic:
      - read max 'week' from opponent_map_from_props.csv (built early in workflow)
      - threshold = clamp(max(floor, week * per_week_rows), upper=full_season_cap)

    Defaults yield ~20k rows @ week 8 and cap at 80k for a full season.
    """
    try:
        p = Path(schedule_csv)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[pbp_threshold] WARN: {schedule_csv} missing/empty. Using floor={floor}.")
            return floor

        om = pd.read_csv(p)
        if "week" not in om.columns or om["week"].dropna().empty:
            print(f"[pbp_threshold] WARN: no 'week' column in {schedule_csv}. Using floor={floor}.")
            return floor

        wk = pd.to_numeric(om["week"], errors="coerce").max()
        if pd.isna(wk):
            print(f"[pbp_threshold] WARN: cannot parse week from {schedule_csv}. Using floor={floor}.")
            return floor

        wk = int(max(1, min(int(wk), 18)))  # clamp to [1, 18]
        est = max(floor, wk * per_week_rows)
        dyn = min(est, full_season_cap)
        print(
            f"[pbp_threshold] dynamic min_rows = {dyn} (week={wk}, per_week_rows={per_week_rows}, cap={full_season_cap})"
        )
        return dyn
    except Exception as e:
        print(f"[pbp_threshold] WARN: fallback due to error: {e}. Using floor={floor}.")
        return floor
