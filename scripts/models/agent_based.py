# scripts/models/agent_based.py
def matchup_adjust(features: dict) -> dict:
    # Apply CB shadow penalties, man/zone boosts, box-rate RB YPC mods, pressure→YPA
    adj = dict(features)
    if adj.get("cb_shadow_elite"): adj["eff_mu"] *= 0.94; adj["target_share"] *= 0.92
    if adj.get("heavy_zone"):      adj["slot_bonus"] = 1.04
    if adj.get("light_box_rate",0) >= 0.6: adj["rb_ypc"] *= 1.07
    if adj.get("heavy_box_rate",0) >= 0.6: adj["rb_ypc"] *= 0.94
    return adj

