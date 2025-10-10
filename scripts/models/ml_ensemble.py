from . import Leg, LegResult
def run(leg: Leg)->LegResult:
    p_ml=leg.features.get('p_ml')
    if p_ml is None: return LegResult(p_model=0.5, mu=None, sigma=None, notes='ML fallback 0.5')
    return LegResult(p_model=float(p_ml), mu=None, sigma=None, notes='ML ensemble')
