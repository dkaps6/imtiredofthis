from . import monte_carlo, bayes_hier, markov, ml_ensemble
def blend(leg, context):
    r_mc=monte_carlo.run(leg); r_b=bayes_hier.run(leg); r_mk=markov.run(leg); r_ml=ml_ensemble.run(leg)
    w_mc=context.get('w_mc',0.25); w_b=context.get('w_bayes',0.25); w_mk=context.get('w_markov',0.25); w_ml=context.get('w_ml',0.25)
    p_blend=w_mc*r_mc.p_model+w_b*r_b.p_model+w_mk*r_mk.p_model+w_ml*r_ml.p_model
    p_mkt=leg.features.get('p_market_fair',0.5)
    p_final=0.65*p_blend+0.35*(p_mkt if p_mkt is not None else 0.5)
    return {'p_mc':r_mc.p_model,'p_bayes':r_b.p_model,'p_markov':r_mk.p_model,'p_ml':r_ml.p_model,'p_blend':p_blend,'p_market':p_mkt,'p_final':p_final,'notes':'65/35 model↔market blend'}
