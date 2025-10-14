from math import erf, sqrt
from .shared_types import Leg, LegResult

try:
    from scripts.config import MONTE_CARLO_TRIALS
except Exception:
    MONTE_CARLO_TRIALS = 25000


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _to_float_or_none(v):
    try:
        return float(v)
    except Exception:
        return None


def run(leg: Leg, n: int = None) -> LegResult:
    """
    Monte Carlo seatbelt:
      - If leg.market is binary (Anytime TD), skip Normal calc and pass through a calibrated p.
      - If mu/sd are missing or non-numeric, return a safe fallback using p_model or p_market_fair.
      - Otherwise, identical behavior to your current implementation.
    """
    n = n or MONTE_CARLO_TRIALS

    # 1) Handle binary markets explicitly (Anytime TD and common aliases)
    if getattr(leg, "market", None) in {"player_anytime_td", "anytime_td", "player_td_anytime"}:
        p = (
            leg.features.get("p_model")
            if leg.features.get("p_model") is not None
            else leg.features.get("p_market_fair", 0.5)
        )
        return LegResult(p_model=float(p), mu=0.0, sigma=1.0, notes=f"MC:binary passthrough n={n}")

    # 2) Read mu/sd, but fall back safely if missing/invalid
    mu = _to_float_or_none(leg.features.get("mu", None))
    sd = _to_float_or_none(leg.features.get("sd", None))

    if mu is None or sd is None:
        # Safe fallback: if we don't have continuous parameters, don't crash.
        p = (
            leg.features.get("p_model")
            if leg.features.get("p_model") is not None
            else leg.features.get("p_market_fair", 0.5)
        )
        return LegResult(p_model=float(p), mu=0.0, sigma=1.0, notes=f"MC:fallback(no_mu_sd) n={n}")

    # 3) Normal calc (unchanged when mu/sd are present)
    sd = max(1e-6, float(sd))
    z = (leg.line - float(mu)) / sd
    p_over = 1.0 - normal_cdf(z)
    return LegResult(p_model=p_over, mu=float(mu), sigma=sd, notes=f"MC n={n}")
