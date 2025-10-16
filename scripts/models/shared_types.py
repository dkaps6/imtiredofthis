# scripts/models/shared_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Leg:
    player_id: str
    player: str
    team: str
    market: str
    line: float
    features: Dict[str, Any]

@dataclass
class LegResult:
    p_model: float
    p_market: Optional[float] = None
    mu: Optional[float] = None      # <-- add
    sigma: Optional[float] = None   # <-- add
    notes: str = ""
