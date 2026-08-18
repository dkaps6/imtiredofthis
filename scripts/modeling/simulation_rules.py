"""Translate canonical football rules into simulation-ready player/team inputs.

The rules layer adjusts *assumptions* before Monte Carlo rather than multiplying
final projections after the fact.  This keeps finite team opportunity, shared
pace, and shared efficiency shocks intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from scripts.modeling.context_bridge import load_model_contexts
from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.modeling.rules_v2 import coverage_penalty, matchup_multipliers, project_game_script
from scripts.utils.canonical_names import canonicalize_player_name_safe


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _num(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _is_wr(position: str, role: str) -> bool:
    p = str(position or "").upper()
    r = str(role or "").upper()
    return p in {"WR", "LWR", "RWR", "SWR"} or "WR" in r


def _wr_role_labels(players: list[PlayerContext]) -> Dict[tuple[str, str], str]:
    """Map team/player to WR1, WR1_5, SLOT using alignment + pregame usage.

    This mirrors the original rulebook intent: SWR is SLOT; the two highest-use
    perimeter receivers are WR1 and WR1.5.  It never renames source positions.
    """
    out: Dict[tuple[str, str], str] = {}
    by_team: Dict[str, list[PlayerContext]] = {}
    for p in players:
        if _is_wr(p.position, p.role):
            by_team.setdefault(p.team, []).append(p)
    for team, group in by_team.items():
        slots = [p for p in group if str(p.position).upper() == "SWR" or "SLOT" in str(p.role).upper()]
        for p in slots:
            out[(team, _key(p.player))] = "SLOT"
        perim = [p for p in group if p not in slots]
        perim.sort(key=lambda p: _num(p.features.get("tgt_share"), 0.0), reverse=True)
        if perim:
            out[(team, _key(perim[0].player))] = "WR1"
        if len(perim) > 1:
            out[(team, _key(perim[1].player))] = "WR1_5"
    return out


def _injury_limited(ctx: PlayerContext) -> bool:
    status = str(ctx.features.get("injury_status") or "").upper()
    designation = str(ctx.features.get("injury_designation") or "").upper()
    text = f"{status} {designation}"
    return any(token in text for token in ("OUT", "DOUBTFUL", "IR", "PUP"))


def apply_rules_to_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of metrics with explicit rule-adjusted simulation columns.

    Required football identity is sourced from the canonical model context bridge.
    If a player has multiple sportsbook rows, the same football adjustment is
    applied to every line/book row for that player.
    """
    if metrics is None or metrics.empty:
        return metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame()

    teams, players = load_model_contexts()
    by_player = {(p.team, _key(p.player)): p for p in players}
    role_labels = _wr_role_labels(players)

    out = metrics.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out["_bridge_key"] = out.get("player_clean_key", out.get("player", "")).map(_key)

    # Defaults preserve current simulation behavior whenever a context/rule input
    # is unavailable.
    for col in (
        "rules_plays_est", "rules_pass_rate", "rules_tgt_share", "rules_rush_share",
        "rules_ypt", "rules_ypc", "rules_ypa", "rules_volatility_mult",
        "rules_pass_eff_mult", "rules_rush_eff_mult",
    ):
        out[col] = np.nan
    out["rules_applied"] = 0
    out["rules_role"] = ""

    for idx, row in out.iterrows():
        team = str(row.get("team", "") or "").upper().strip()
        ctx = by_player.get((team, str(row["_bridge_key"])))
        if ctx is None or ctx.offense is None or ctx.defense is None:
            continue

        script = project_game_script(ctx.offense, ctx.defense)
        mods = matchup_multipliers(ctx.offense, ctx.defense)
        role = role_labels.get((team, str(row["_bridge_key"])), "")

        base_tgt = _num(row.get("target_share", row.get("tgt_share", ctx.features.get("tgt_share"))))
        base_rush = _num(row.get("rush_share", ctx.features.get("rush_share")))
        base_ypt = _num(row.get("ypt", ctx.features.get("ypt")))
        base_ypc = _num(row.get("ypc", ctx.features.get("ypc")))
        base_ypa = _num(row.get("ypa", ctx.features.get("ypa")))

        tgt_mult = 1.0
        pos = str(ctx.position or "").upper()
        if role == "WR1":
            tgt_mult *= mods.wr1_target_mult
        elif role == "WR1_5":
            tgt_mult *= mods.wr1_5_target_mult
        elif role == "SLOT":
            tgt_mult *= mods.slot_target_mult
        elif pos == "TE":
            tgt_mult *= mods.te_target_mult
        elif pos in {"RB", "FB"}:
            tgt_mult *= mods.rb_rec_target_mult

        # Apply the legacy individual shadow/man/zone adjustment only when the
        # bridge says the relevant coverage information is actually available.
        if np.isfinite(base_ypt) and np.isfinite(base_tgt) and _is_wr(pos, ctx.role):
            matchup_available = int(_num(ctx.features.get("matchup_available"), 0.0)) == 1
            coverage_available = int(_num(ctx.features.get("team_coverage_available"), 0.0)) == 1
            tough_shadow = matchup_available and bool(str(ctx.features.get("primary_cb") or "").strip())
            man = _num(ctx.defense.coverage_man_rate, 0.0) >= 0.50 if coverage_available else False
            zone = _num(ctx.defense.coverage_zone_rate, 0.0) >= 0.60 if coverage_available else False
            adj_ypt, adj_share = coverage_penalty(
                base_ypt,
                base_tgt * tgt_mult,
                tough_shadow=tough_shadow,
                heavy_man=man and tough_shadow,
                heavy_zone=zone and not tough_shadow,
            )
            base_ypt = adj_ypt
            base_tgt = adj_share
        elif np.isfinite(base_tgt):
            base_tgt *= tgt_mult

        # Injury-limited players retain identity but lose part of their direct
        # opportunity. Team-level vacancy redistribution is deliberately deferred
        # until official injury rows exist and can be calibrated in walk-forward tests.
        if _injury_limited(ctx) and np.isfinite(base_tgt):
            base_tgt *= 0.50
        if _injury_limited(ctx) and np.isfinite(base_rush):
            base_rush *= 0.50

        out.at[idx, "rules_plays_est"] = script.projected_plays
        out.at[idx, "rules_pass_rate"] = script.projected_pass_attempts / script.projected_plays if script.projected_plays else np.nan
        out.at[idx, "rules_tgt_share"] = base_tgt
        out.at[idx, "rules_rush_share"] = base_rush
        out.at[idx, "rules_ypt"] = base_ypt * mods.pass_eff_mult if np.isfinite(base_ypt) else np.nan
        out.at[idx, "rules_ypc"] = base_ypc * mods.rb_rush_eff_mult if np.isfinite(base_ypc) else np.nan
        out.at[idx, "rules_ypa"] = base_ypa * mods.pass_eff_mult if np.isfinite(base_ypa) else np.nan
        out.at[idx, "rules_volatility_mult"] = mods.volatility_mult
        out.at[idx, "rules_pass_eff_mult"] = mods.pass_eff_mult
        out.at[idx, "rules_rush_eff_mult"] = mods.rb_rush_eff_mult
        out.at[idx, "rules_applied"] = 1
        out.at[idx, "rules_role"] = role

    out.drop(columns=["_bridge_key"], inplace=True)
    return out


def build_rule_diagnostics() -> pd.DataFrame:
    """Build market-independent diagnostics from canonical player contexts.

    This lets no-live-odds Full Slate runs prove that game-script/matchup rules
    are resolvable before live props are enabled.
    """
    teams, players = load_model_contexts()
    labels = _wr_role_labels(players)
    rows = []
    for p in players:
        if p.offense is None or p.defense is None:
            continue
        script = project_game_script(p.offense, p.defense)
        mods = matchup_multipliers(p.offense, p.defense)
        rows.append({
            "player": p.player,
            "team": p.team,
            "opponent": p.opponent,
            "season": p.season,
            "week": p.week,
            "position": p.position,
            "role": p.role,
            "rules_role": labels.get((p.team, _key(p.player)), ""),
            "projected_plays": script.projected_plays,
            "projected_pass_attempts": script.projected_pass_attempts,
            "projected_rush_attempts": script.projected_rush_attempts,
            "lead_prob": script.lead_prob,
            "trail_prob": script.trail_prob,
            "pass_eff_mult": mods.pass_eff_mult,
            "rush_eff_mult": mods.rb_rush_eff_mult,
            "wr1_target_mult": mods.wr1_target_mult,
            "wr1_5_target_mult": mods.wr1_5_target_mult,
            "slot_target_mult": mods.slot_target_mult,
            "te_target_mult": mods.te_target_mult,
            "rb_rec_target_mult": mods.rb_rec_target_mult,
            "volatility_mult": mods.volatility_mult,
        })
    return pd.DataFrame(rows)
