from math import erf, sqrt
from . import Leg, LegResult
def run(leg: Leg)->LegResult:
    mu=leg.features.get('bayes_mu', leg.features.get('mu',0.0))
    sd=leg.features.get('bayes_sd', leg.features.get('sd',1.0))
    z=(leg.line-mu)/max(1e-6,sd)
    p_over=1-0.5*(1+erf(z/sqrt(2)))
    return LegResult(p_model=p_over, mu=mu, sigma=sd, notes='Bayes pooled')
