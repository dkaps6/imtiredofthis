# scripts/models/types.py
"""
Shared dataclasses and types for model inputs/outputs used across predictors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from scripts.models.shared_types import TeamScriptFeatures

@dataclass
class LegResult:
    """
    Represents one simulated leg result for Monte Carlo / prop pricing.

    Attributes:
        p_model: Model-estimated probability (Over or Yes hit rate).
        mu: Mean (expected value) of the model distribution.
        sigma: Standard deviation of the model distribution.
        p_market: Market (vig-removed) probability if available.
        edge: Model - market differential (in probability points).
        kelly: Kelly fraction (recommended % of bankroll).
        notes: Optional text describing the adjustment / reasoning.
    """
    p_model: float
    mu: Optional[float] = None
    sigma: Optional[float] = None
    p_market: Optional[float] = None
    edge: Optional[float] = None
    kelly: Optional[float] = None
    notes: str = ""

    def as_dict(self):
        """Convert to dict for CSV/export convenience."""
        return {
            "p_model": self.p_model,
            "mu": self.mu,
            "sigma": self.sigma,
            "p_market": self.p_market,
            "edge": self.edge,
            "kelly": self.kelly,
            "notes": self.notes,
        }

    def __post_init__(self):
        # Ensure numeric types
        try:
            self.p_model = float(self.p_model)
        except Exception:
            self.p_model = None
        for attr in ("mu", "sigma", "p_market", "edge", "kelly"):
            val = getattr(self, attr)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except Exception:
                    setattr(self, attr, None)


@dataclass
class PlayerModelInput:
    """Represents the full context for a single player/market model evaluation."""

    player_id: str
    player_name: str
    team_abbr: str
    opponent_abbr: str
    market: str
    line: Optional[float] = None
    stat_type: Optional[str] = None
    bookmaker: Optional[str] = None
    event_id: Optional[str] = None
    model_features: Dict[str, Any] = field(default_factory=dict)

    # Team-level script features (offense & defense)
    offense_script: Optional[TeamScriptFeatures] = None
    defense_script: Optional[TeamScriptFeatures] = None
