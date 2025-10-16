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
