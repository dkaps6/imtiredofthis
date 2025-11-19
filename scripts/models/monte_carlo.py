import logging
from dataclasses import dataclass
from math import erf, sqrt
from typing import Optional

from .shared_types import Leg, LegResult, TeamScriptFeatures

try:
    from scripts.models.types import PlayerModelInput  # type: ignore
except Exception:  # pragma: no cover - fallback for legacy structure
    PlayerModelInput = Leg  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ScriptModifiers:
    play_volume_factor: float = 1.0
    pass_volume_factor: float = 1.0
    rush_volume_factor: float = 1.0
    explosiveness_factor: float = 1.0
    pressure_turnover_factor: float = 1.0


def _extract_script_feature(player: PlayerModelInput, attr_name: str):
    if hasattr(player, attr_name):
        value = getattr(player, attr_name)
        if value is not None:
            return value
    features = getattr(player, "features", None)
    if isinstance(features, dict):
        return features.get(attr_name)
    return None


def _script_attr(script: Optional[TeamScriptFeatures], name: str, default=0.0):
    if script is None:
        return default
    try:
        value = getattr(script, name)
        if value is None:
            return default
        return value
    except AttributeError:
        pass
    # Fall back to dict-like get (Series/dicts)
    if hasattr(script, "get"):
        try:
            value = script.get(name, default)
            return default if value is None else value
        except Exception:
            return default
    return default


def _compute_script_modifiers(player: PlayerModelInput) -> ScriptModifiers:
    mods = ScriptModifiers()

    offense: Optional[TeamScriptFeatures] = _extract_script_feature(player, "offense_script")
    defense: Optional[TeamScriptFeatures] = _extract_script_feature(player, "defense_script")

    if offense is None or defense is None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "No script features for player=%s; returning neutral modifiers",
                getattr(player, "player", getattr(player, "player_name", "unknown")),
            )
        return mods

    success_diff = 0.0
    sr_diff = _script_attr(offense, "success_rate_diff", None)
    if sr_diff is not None:
        try:
            success_diff = float(sr_diff)
        except Exception:
            success_diff = 0.0
    else:
        off_sr = float(_script_attr(offense, "success_rate_off", 0.0) or 0.0)
        def_sr = float(_script_attr(defense, "success_rate_def", 0.0) or 0.0)
        success_diff = off_sr - def_sr

    success_diff = max(min(success_diff, 0.15), -0.15)
    mods.play_volume_factor *= 1.0 + (success_diff * (10.0 / 15.0))

    neutral_pace = float(_script_attr(offense, "neutral_pace", 0.0) or 0.0)
    neutral_pace_last5 = float(
        _script_attr(offense, "neutralpacelast5", _script_attr(offense, "neutral_pace_last5", 0.0))
        or 0.0
    )
    pace_z = float(_script_attr(offense, "pace_neutral_z", 0.0) or 0.0)

    if abs(pace_z) > 0.01:
        pace_boost = max(min(pace_z, 1.5), -1.5) * 0.06
    else:
        baseline_pace = 2.4
        diff = (neutral_pace_last5 or neutral_pace) - baseline_pace
        pace_boost = max(min(diff * 0.08, 0.10), -0.10)
    mods.play_volume_factor *= 1.0 + pace_boost

    def_pressure = float(_script_attr(defense, "pressure_rate", 0.0) or 0.0)
    off_pressure_allowed = float(
        _script_attr(offense, "pressure_rate_allowed", 0.0) or 0.0
    )
    pressure_diff = def_pressure - off_pressure_allowed
    pressure_diff = max(min(pressure_diff, 0.12), -0.12)
    mods.pressure_turnover_factor *= 1.0 + (pressure_diff * (20.0 / 12.0))

    proe_val = _script_attr(offense, "proe", _script_attr(offense, "pass_rate_over_expected", 0.0)) or 0.0
    proe = float(proe_val)
    proe_z = float(_script_attr(offense, "proe_z", 0.0) or 0.0)
    if abs(proe_z) > 0.01:
        proe_boost = max(min(proe_z, 1.5), -1.5) * 0.07
    else:
        proe_boost = max(min(proe * 0.5, 0.10), -0.10)

    def_pass_epa = float(_script_attr(defense, "def_pass_epa", 0.0) or 0.0)
    def_rush_epa = float(_script_attr(defense, "def_rush_epa", 0.0) or 0.0)
    epa_diff = def_pass_epa - def_rush_epa

    funnel_pass_boost = 0.0
    funnel_rush_boost = 0.0
    if epa_diff > 0.10:
        funnel_pass_boost = 0.06
    elif epa_diff < -0.10:
        funnel_rush_boost = 0.06

    pass_tilt = proe_boost + funnel_pass_boost - funnel_rush_boost
    pass_tilt = max(min(pass_tilt, 0.12), -0.12)

    mods.pass_volume_factor *= 1.0 + pass_tilt
    mods.rush_volume_factor *= 1.0 - (pass_tilt * 0.8)

    expl_allowed = float(_script_attr(defense, "explosive_play_rate_allowed", 0.0) or 0.0)
    airyards_att = float(
        _script_attr(offense, "airyardsatt", _script_attr(offense, "airyardsatt_sharp", 0.0))
        or 0.0
    )
    expl_boost = 0.0
    if expl_allowed > 0.12:
        expl_boost += 0.05
    if airyards_att > 8.5:
        expl_boost += 0.03
    mods.explosiveness_factor *= 1.0 + expl_boost

    return mods


PASS_YARDAGE_MARKETS = {
    "player_pass_yds",
    "player_passing_yards",
    "player_longest_completion",
}
PASS_VOLUME_MARKETS = {
    "player_pass_attempts",
    "player_pass_completions",
}
RECEIVING_YARDAGE_MARKETS = {
    "player_rec_yds",
    "player_receiving_yards",
    "player_longest_reception",
}
RECEPTION_MARKETS = {"player_receptions"}
RUSH_YARDAGE_MARKETS = {
    "player_rush_yds",
    "player_rushing_yards",
    "player_longest_rush",
}
RUSH_ATTEMPT_MARKETS = {"player_rush_attempts", "player_rushing_attempts"}
RUSH_REC_COMBO_MARKETS = {"player_rush_rec_yds"}
TURNOVER_MARKETS = {
    "player_pass_ints",
    "player_interceptions",
    "player_interceptions_thrown",
}
SACK_MARKETS = {"player_qb_sacks", "player_times_sacked", "player_pass_sacks"}


def _market_multiplier(market: str, mods: ScriptModifiers) -> float:
    market_key = (market or "").lower()

    play_volume = mods.play_volume_factor
    pass_volume = play_volume * mods.pass_volume_factor
    rush_volume = play_volume * mods.rush_volume_factor

    if market_key in PASS_YARDAGE_MARKETS or (
        "pass" in market_key and "yd" in market_key
    ):
        multiplier = pass_volume * mods.explosiveness_factor
    elif market_key in PASS_VOLUME_MARKETS or (
        "pass" in market_key and ("att" in market_key or "comp" in market_key)
    ):
        multiplier = pass_volume
    elif market_key in RECEIVING_YARDAGE_MARKETS or (
        ("rec" in market_key or "receive" in market_key) and "yd" in market_key
    ):
        multiplier = pass_volume * mods.explosiveness_factor
    elif market_key in RECEPTION_MARKETS or (
        "rec" in market_key and "tion" in market_key
    ):
        multiplier = pass_volume
    elif market_key in RUSH_YARDAGE_MARKETS or (
        "rush" in market_key and "yd" in market_key
    ):
        multiplier = rush_volume * mods.explosiveness_factor
    elif market_key in RUSH_ATTEMPT_MARKETS or (
        "rush" in market_key and "att" in market_key
    ):
        multiplier = rush_volume
    elif market_key in RUSH_REC_COMBO_MARKETS:
        avg_volume = (pass_volume + rush_volume) / 2.0
        multiplier = avg_volume * mods.explosiveness_factor
    elif market_key in TURNOVER_MARKETS or (
        "pass" in market_key and "int" in market_key
    ):
        multiplier = pass_volume * mods.pressure_turnover_factor
    elif market_key in SACK_MARKETS or "sack" in market_key:
        multiplier = play_volume * mods.pressure_turnover_factor
    else:
        multiplier = play_volume

    return max(multiplier, 0.0)

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

    mods = _compute_script_modifiers(leg)
    market_multiplier = _market_multiplier(getattr(leg, "market", ""), mods)
    if market_multiplier <= 0:
        market_multiplier = 1.0

    mu_val = float(mu) * market_multiplier
    sd_val = max(1e-6, float(sd) * max(market_multiplier, 0.1))

    # 3) Normal calc (unchanged when mu/sd are present)
    z = (leg.line - mu_val) / sd_val
    p_over = 1.0 - normal_cdf(z)
    return LegResult(p_model=p_over, mu=mu_val, sigma=sd_val, notes=f"MC n={n}")
