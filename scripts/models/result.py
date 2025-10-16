# scripts/models/result.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class LegResult:
    # required in your predictors
    p_model: float
    # new/optional fields some models pass
    mu: Optional[float] = None
    sigma: Optional[float] = None
    # keep room for other fields some paths may set
    p_market: Optional[float] = None
    edge: Optional[float] = None
    kelly: Optional[float] = None
    notes: str = ""
