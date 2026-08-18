"""Canonical data contracts for the NFL projection stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TeamContext:
    team: str
    season: int
    success_rate_off: Optional[float] = None
    success_rate_def: Optional[float] = None
    pressure_rate_generated: Optional[float] = None
    pressure_rate_allowed: Optional[float] = None
    neutral_pace: Optional[float] = None
    neutral_pace_last5: Optional[float] = None
    sec_per_play_last5: Optional[float] = None
    plays_est: Optional[float] = None
    proe: Optional[float] = None
    explosive_play_rate_allowed: Optional[float] = None
    coverage_man_rate: Optional[float] = None
    coverage_zone_rate: Optional[float] = None
    middle_open_rate: Optional[float] = None
    light_box_rate: Optional[float] = None
    heavy_box_rate: Optional[float] = None
    def_pass_epa: Optional[float] = None
    def_rush_epa: Optional[float] = None


@dataclass(frozen=True)
class PlayerProjectionInput:
    player: str
    team: str
    opponent: str
    season: int
    week: int
    position: str
    market: str
    game_id: str = ""
    line: Optional[float] = None
    features: Dict[str, Any] = field(default_factory=dict)
    offense: Optional[TeamContext] = None
    defense: Optional[TeamContext] = None


@dataclass(frozen=True)
class ComponentPrediction:
    component: str
    mean: Optional[float] = None
    sd: Optional[float] = None
    probability: Optional[float] = None
    available: bool = True
    notes: str = ""


@dataclass(frozen=True)
class ProjectionDistribution:
    player: str
    market: str
    mean: float
    sd: float
    iterations: int
    game_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
