# scripts/models/shared_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Leg:
    """
    Represents a single betting leg (player × market × line)
    with all contextual features attached for modeling.
    """
    player_id: str
    player: str
    team: str
    market: str
    line: float
    features: Dict[str, Any]


@dataclass
class LegResult:
    """
    The output of any single-leg model evaluation.
    p_model: model-implied probability of Over (or Yes)
    p_market: market-implied (de-vigged) probability
    mu, sigma: distribution parameters used for the model
    notes: optional string describing any adjustments made
    """
    p_model: float
    p_market: Optional[float] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_model": self.p_model,
            "p_market": self.p_market,
            "mu": self.mu,
            "sigma": self.sigma,
            "notes": self.notes,
        }


@dataclass
class MarketSummary:
    """
    High-level market summary for a given player/market combo
    used when aggregating or blending model outputs.
    """
    player: str
    market: str
    line: float
    model_prob: float
    market_prob: float
    fair_odds: float
    edge: float
    kelly: float
    tier: str


@dataclass
class TeamScriptFeatures:
    """
    Team-level script and matchup features derived from team_form.csv.

    This is intentionally aligned with the columns produced by scripts/make_team_form.py,
    plus a derived pressure_rate_diff for convenience.

    All fields are floats where possible; if a column is missing, the loader will
    fill the field with None and log a warning.
    """

    team: str
    team_abbr: str
    season: int

    # Success rate & explosive plays
    success_rate_off: Optional[float]
    success_rate_def: Optional[float]
    success_rate_diff: Optional[float]
    explosive_play_rate_allowed: Optional[float]

    # Pressure metrics
    pressure_rate: Optional[float]
    pressure_rate_allowed: Optional[float]
    pressure_rate_diff: Optional[float]  # derived: pressure_rate - pressure_rate_allowed

    # Pace / plays / pass tendency
    neutral_pace: Optional[float]
    neutralpacelast5: Optional[float]
    secplay_last_5: Optional[float]
    plays_per_game: Optional[float]
    plays_est: Optional[float]
    pass_rate_over_expected: Optional[float]
    proe: Optional[float]

    # Coverage & boxes
    coverage_man_rate: Optional[float]
    coverage_zone_rate: Optional[float]
    middle_open_rate: Optional[float]
    light_box_rate: Optional[float]
    heavy_box_rate: Optional[float]

    # Yards-per-target allowed by position / alignment
    ypt_allowed_wr: Optional[float]
    ypt_allowed_te: Optional[float]
    ypt_allowed_rb: Optional[float]
    ypt_allowed_outside: Optional[float]
    ypt_allowed_slot: Optional[float]

    # Trenches / OL-DL rush metrics
    yards_before_contact_per_rb_rush_x: Optional[float]
    rush_stuff_rate_x: Optional[float]
    yards_before_contact_per_rb_rush_y: Optional[float]
    rush_stuff_rate_y: Optional[float]
