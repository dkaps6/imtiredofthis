from __future__ import annotations

import pandas as pd
from pathlib import Path


def get_dynamic_min_rows(
    schedule_csv: str = "data/opponent_map_from_props.csv",
    *,
    floor_hard: int = 20000,
    soft_target_per_week: int = 2500,
    cap_full_season: int = 80000,
) -> int:
    """
    Compute a soft target for how many PBP rows we *expect* to see right now.

    We CANNOT just take max(week) from opponent_map_from_props.csv, because that
    file can include future weeks (1-18). So we'll approximate "current" week by
    looking at the 75th percentile of weeks in that file.

    We then do:
      soft_target = clamp( max(floor_hard, wk_guess * soft_target_per_week), cap_full_season )

    - floor_hard = 20k: if below ~20k rows, we consider the dataset unusable.
    - cap_full_season = 80k: don't demand more than ~80k rows.
    """
    try:
        p = Path(schedule_csv)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[pbp_threshold] WARN: {schedule_csv} missing/empty. using floor_hard={floor_hard}")
            return floor_hard

        om = pd.read_csv(p)

        # Estimate "played" week as ~75th percentile to avoid future weeks from schedule
        if "week" in om.columns and not om["week"].dropna().empty:
            week_series = pd.to_numeric(om["week"], errors="coerce").dropna()
            if len(week_series) > 0:
                wk_guess = int(week_series.quantile(0.75))
            else:
                wk_guess = 1
        else:
            wk_guess = 1

        # clamp guess to [1,18]
        if wk_guess < 1:
            wk_guess = 1
        if wk_guess > 18:
            wk_guess = 18

        soft_target = wk_guess * soft_target_per_week
        soft_target = max(floor_hard, soft_target)
        soft_target = min(cap_full_season, soft_target)

        print(f"[pbp_threshold] inferred wk≈{wk_guess}, soft_target rows={soft_target}")
        return int(soft_target)

    except Exception as e:
        print(f"[pbp_threshold] WARN: fallback due to error {e}. using floor_hard={floor_hard}")
        return floor_hard


def enforce_min_rows(pbp_df: pd.DataFrame, requested_min_rows: int) -> None:
    """
    We enforce two tiers:

    1. Hard floor (20k rows): if we're below that, data is trash → raise RuntimeError.
    2. Soft target (requested_min_rows): if we're below that but still above floor,
       just warn. This keeps mid-season slates alive but will still scream if something
       is clearly wrong.

    This keeps CI from blowing up in Week 8 just because we aren't at 80k rows yet,
    but still fails loud if we got garbage (like only ~2k preseason rows).
    """
    actual = len(pbp_df)
    floor = 20000

    if actual < floor:
        raise RuntimeError(
            f"PBP rows way too small ({actual} < {floor}). Aborting: data is not trustworthy."
        )

    if actual < requested_min_rows:
        print(f"[pbp_threshold] WARN: PBP rows {actual} < soft target {requested_min_rows}, continuing anyway.")
    else:
        print(f"[pbp_threshold] OK: PBP rows {actual} ≥ soft target {requested_min_rows}.")
