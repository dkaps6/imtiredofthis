from math import erf, sqrt
from .shared_types import Leg, LegResult
def run(leg: Leg)->LegResult:
    att=leg.features.get('adj_attempts'); eff=leg.features.get('eff_mu'); sd=leg.features.get('eff_sd')
    if att and eff and sd:
        mu=att*eff; sigma=(att**0.5)*sd; z=(leg.line-mu)/max(1e-6,sigma)
        p_over=1-0.5*(1+erf(z/sqrt(2)))
        return LegResult(p_model=p_over, mu=mu, sigma=sigma, notes='Markov volume-adjusted')
    return LegResult(p_model=0.5, mu=None, sigma=None, notes='Markov fallback')
