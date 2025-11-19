import logging
from typing import Dict

import pandas as pd

from scripts.models.shared_types import TeamScriptFeatures

logger = logging.getLogger(__name__)


def _get_opt(row: pd.Series, col: str):
    """
    Helper: safely get a column from a pandas row, returning None if missing.

    This lets us keep TeamScriptFeatures robust to small schema changes
    without breaking the whole pipeline.
    """
    if col not in row.index:
        return None
    val = row[col]
    # Normalize NaN to None for easier downstream handling
    if pd.isna(val):
        return None
    return val


def load_team_script_features(
    path: str = "data/team_form.csv",
) -> Dict[str, TeamScriptFeatures]:
    """
    Load team-level script features from the team_form.csv file produced by
    scripts/make_team_form.py.

    Returns:
        Dict keyed by team_abbr (e.g. "DAL", "LV", "PHI"), with TeamScriptFeatures
        instances for each row.

    Behavior:
        - Logs the path and row count.
        - Handles missing columns gracefully by filling the corresponding
          dataclass fields with None.
        - Computes pressure_rate_diff as pressure_rate - pressure_rate_allowed
          when both are available.
    """
    logger.info("Loading team script features from %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("team_form CSV not found at %s", path)
        return {}
    except Exception as exc:
        logger.exception("Failed to read team_form CSV at %s: %s", path, exc)
        return {}

    if df.empty:
        logger.warning("team_form CSV at %s is empty; no team features loaded", path)
        return {}

    # Normalize expected core columns
    required_cols = ["team", "team_abbr", "season"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        logger.error(
            "team_form CSV at %s is missing required columns: %s",
            path,
            ", ".join(missing_required),
        )
        return {}

    # Log the high-level shape and some of the relevant columns for debugging
    logger.info(
        "team_form CSV loaded from %s with shape %s and columns: %s",
        path,
        df.shape,
        ", ".join(df.columns),
    )

    features_by_team: Dict[str, TeamScriptFeatures] = {}

    for _, row in df.iterrows():
        team = str(row["team"])
        team_abbr = str(row["team_abbr"])
        season_val = row["season"]
        try:
            season = int(season_val)
        except Exception:
            # If season is not cleanly castable, log and skip this row
            logger.warning(
                "Skipping row for team_abbr=%s due to non-integer season value: %r",
                team_abbr,
                season_val,
            )
            continue

        # Success rate & explosives
        success_rate_off = _get_opt(row, "success_rate_off")
        success_rate_def = _get_opt(row, "success_rate_def")
        success_rate_diff = _get_opt(row, "success_rate_diff")
        explosive_play_rate_allowed = _get_opt(row, "explosive_play_rate_allowed")

        # Pressure metrics
        pressure_rate = _get_opt(row, "pressure_rate")
        pressure_rate_allowed = _get_opt(row, "pressure_rate_allowed")

        if pressure_rate is not None and pressure_rate_allowed is not None:
            pressure_rate_diff = float(pressure_rate) - float(pressure_rate_allowed)
        else:
            pressure_rate_diff = None

        # Pace / plays / pass tendency
        neutral_pace = _get_opt(row, "neutral_pace")
        neutralpacelast5 = _get_opt(row, "neutralpacelast5")
        secplay_last_5 = _get_opt(row, "secplay_last_5")
        plays_per_game = _get_opt(row, "plays_per_game")
        plays_est = _get_opt(row, "plays_est")
        pass_rate_over_expected = _get_opt(row, "pass_rate_over_expected")
        proe = _get_opt(row, "proe")

        # Coverage & box metrics
        coverage_man_rate = _get_opt(row, "coverage_man_rate")
        coverage_zone_rate = _get_opt(row, "coverage_zone_rate")
        middle_open_rate = _get_opt(row, "middle_open_rate")
        light_box_rate = _get_opt(row, "light_box_rate")
        heavy_box_rate = _get_opt(row, "heavy_box_rate")

        # YPT allowed by position / alignment
        ypt_allowed_wr = _get_opt(row, "ypt_allowed_wr")
        ypt_allowed_te = _get_opt(row, "ypt_allowed_te")
        ypt_allowed_rb = _get_opt(row, "ypt_allowed_rb")
        ypt_allowed_outside = _get_opt(row, "ypt_allowed_outside")
        ypt_allowed_slot = _get_opt(row, "ypt_allowed_slot")

        # Trenches / OL-DL rush metrics
        yards_before_contact_per_rb_rush_x = _get_opt(
            row, "yards_before_contact_per_rb_rush_x"
        )
        rush_stuff_rate_x = _get_opt(row, "rush_stuff_rate_x")
        yards_before_contact_per_rb_rush_y = _get_opt(
            row, "yards_before_contact_per_rb_rush_y"
        )
        rush_stuff_rate_y = _get_opt(row, "rush_stuff_rate_y")

        features = TeamScriptFeatures(
            team=team,
            team_abbr=team_abbr,
            season=season,
            success_rate_off=success_rate_off,
            success_rate_def=success_rate_def,
            success_rate_diff=success_rate_diff,
            explosive_play_rate_allowed=explosive_play_rate_allowed,
            pressure_rate=pressure_rate,
            pressure_rate_allowed=pressure_rate_allowed,
            pressure_rate_diff=pressure_rate_diff,
            neutral_pace=neutral_pace,
            neutralpacelast5=neutralpacelast5,
            secplay_last_5=secplay_last_5,
            plays_per_game=plays_per_game,
            plays_est=plays_est,
            pass_rate_over_expected=pass_rate_over_expected,
            proe=proe,
            coverage_man_rate=coverage_man_rate,
            coverage_zone_rate=coverage_zone_rate,
            middle_open_rate=middle_open_rate,
            light_box_rate=light_box_rate,
            heavy_box_rate=heavy_box_rate,
            ypt_allowed_wr=ypt_allowed_wr,
            ypt_allowed_te=ypt_allowed_te,
            ypt_allowed_rb=ypt_allowed_rb,
            ypt_allowed_outside=ypt_allowed_outside,
            ypt_allowed_slot=ypt_allowed_slot,
            yards_before_contact_per_rb_rush_x=yards_before_contact_per_rb_rush_x,
            rush_stuff_rate_x=rush_stuff_rate_x,
            yards_before_contact_per_rb_rush_y=yards_before_contact_per_rb_rush_y,
            rush_stuff_rate_y=rush_stuff_rate_y,
        )

        # If there somehow are duplicate team_abbr rows, last-one-wins but we log it.
        if team_abbr in features_by_team:
            logger.warning(
                "Duplicate team_abbr %s encountered in team_form; overwriting previous TeamScriptFeatures",
                team_abbr,
            )

        features_by_team[team_abbr] = features

    logger.info(
        "Loaded %d TeamScriptFeatures entries from %s",
        len(features_by_team),
        path,
    )

    return features_by_team
