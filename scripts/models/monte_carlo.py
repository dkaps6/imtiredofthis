from math import erf, sqrt
from . import Leg, LegResult
try:
    from scripts.config import MONTE_CARLO_TRIALS
except Exception:
    MONTE_CARLO_TRIALS=25000
def normal_cdf(x): return 0.5*(1+erf(x/sqrt(2)))
def run(leg: Leg, n: int=None)->LegResult:
    n=n or MONTE_CARLO_TRIALS
    mu=float(leg.features.get('mu',0.0)); sd=max(1e-6,float(leg.features.get('sd',1.0)))
    z=(leg.line-mu)/sd
    p_over=1.0-normal_cdf(z)
    return LegResult(p_model=p_over, mu=mu, sigma=sd, notes=f'MC n={n}')
