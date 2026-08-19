"""Translate canonical football rules into simulation-ready player/team inputs.

Rules adjust assumptions before Monte Carlo rather than multiplying final
projections. Migration 4A also allows empirical-Bayesian posterior baselines to
feed the rule layer before contextual matchup adjustments are applied.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from scripts.modeling.context_bridge import load_model_contexts
from scripts.modeling.contracts import PlayerContext
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


def _injury_target_overrides(
    players: list[PlayerContext],
    labels: Dict[tuple[str, str], str],
    share_overrides: Dict[tuple[str, str], float] | None = None,
) -> Dict[tuple[str, str], float]:
    """Apply the legacy alpha-vacancy rule while conserving redistributed share."""
    shares = share_overrides or {}
    overrides: Dict[tuple[str, str], float] = {}
    by_team: Dict[str, list[PlayerContext]] = {}
    for p in players:
        by_team.setdefault(p.team, []).append(p)

    def base_share(p: PlayerContext) -> float:
        key = (p.team, _key(p.player))
        return max(0.0, _num(shares.get(key, p.features.get("tgt_share")), 0.0))

    for team, group in by_team.items():
        alpha = next((p for p in group if labels.get((team, _key(p.player))) == "WR1"), None)
        if alpha is None or not _injury_limited(alpha):
            continue
        alpha_share = base_share(alpha)
        if alpha_share <= 0:
            continue
        give = alpha_share * 0.50
        overrides[(team, _key(alpha.player))] = alpha_share - give

        buckets = [
            (0.60, [p for p in group if labels.get((team, _key(p.player))) == "WR1_5"]),
            (0.30, [p for p in group if labels.get((team, _key(p.player))) == "SLOT" or str(p.position).upper() == "TE"]),
            (0.10, [p for p in group if str(p.position).upper() in {"RB", "FB"}]),
        ]
        for weight, recipients in buckets:
            if not recipients:
                continue
            current = [base_share(p) for p in recipients]
            total = sum(current)
            alloc = [(v / total if total > 0 else 1.0 / len(recipients)) for v in current]
            for p, frac, base in zip(recipients, alloc, current):
                overrides[(team, _key(p.player))] = base + give * weight * frac
    return overrides


def apply_rules_to_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame()

    _, players = load_model_contexts()
    by_player = {(p.team, _key(p.player)): p for p in players}
    role_labels = _wr_role_labels(players)

    out = metrics.copy()
    out.columns = [str(c).lower() for c in out.columns]
    source_key = out["player_clean_key"] if "player_clean_key" in out.columns else out["player"]
    out["_bridge_key"] = source_key.map(_key)

    # When Bayesian baselines are present, injury redistribution uses the same
    # posterior opportunity assumptions that will feed simulation.
    bayes_share_by_player: Dict[tuple[str, str], float] = {}
    if "bayes_tgt_share" in out.columns:
        unique = out.drop_duplicates(["team", "_bridge_key"])
        for _, r in unique.iterrows():
            v = _num(r.get("bayes_tgt_share"))
            if np.isfinite(v):
                bayes_share_by_player[(str(r.get("team", "")).upper().strip(), str(r["_bridge_key"]))] = v
    injury_overrides = _injury_target_overrides(players, role_labels, bayes_share_by_player)

    for col in (
        "rules_plays_est", "rules_pass_rate", "rules_tgt_share", "rules_rush_share",
        "rules_ypt", "rules_ypc", "rules_ypa", "rules_catch_rate", "rules_volatility_mult",
        "rules_pass_eff_mult", "rules_rush_eff_mult",
    ):
        out[col] = np.nan
    out["rules_applied"] = 0
    out["rules_role"] = ""
    out["rules_injury_redistribution"] = 0

    for idx, row in out.iterrows():
        team = str(row.get("team", "") or "").upper().strip()
        pkey = str(row["_bridge_key"])
        ctx = by_player.get((team, pkey))
        if ctx is None or ctx.offense is None or ctx.defense is None:
            continue

        script = project_game_script(ctx.offense, ctx.defense)
        mods = matchup_multipliers(ctx.offense, ctx.defense)
        role = role_labels.get((team, pkey), "")

        # Bayesian posterior is the preferred baseline. Raw/blended PlayerForm
        # remains an explicit fallback if a posterior metric is unavailable.
        base_tgt = _num(row.get("bayes_tgt_share", row.get("target_share", row.get("tgt_share", ctx.features.get("tgt_share")))))
        base_rush = _num(row.get("bayes_rush_share", row.get("rush_share", ctx.features.get("rush_share"))))
        base_ypt = _num(row.get("bayes_ypt", row.get("ypt", ctx.features.get("ypt"))))
        base_ypc = _num(row.get("bayes_ypc", row.get("ypc", ctx.features.get("ypc"))))
        base_ypa = _num(row.get("bayes_ypa", row.get("ypa", ctx.features.get("ypa"))))
        base_catch = _num(row.get("bayes_receptions_per_target", row.get("receptions_per_target", row.get("catch_rate", ctx.features.get("catch_rate")))))

        if (team, pkey) in injury_overrides:
            base_tgt = injury_overrides[(team, pkey)]
            out.at[idx, "rules_injury_redistribution"] = 1

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

        if np.isfinite(base_ypt) and np.isfinite(base_tgt) and _is_wr(pos, ctx.role):
            matchup_available = int(_num(ctx.features.get("matchup_available"), 0.0)) == 1
            coverage_available = int(_num(ctx.features.get("team_coverage_available"), 0.0)) == 1
            tough_shadow = matchup_available and bool(str(ctx.features.get("primary_cb") or "").strip())
            man = _num(ctx.defense.coverage_man_rate, 0.0) >= 0.50 if coverage_available else False
            zone = _num(ctx.defense.coverage_zone_rate, 0.0) >= 0.60 if coverage_available else False
            base_ypt, base_tgt = coverage_penalty(
                base_ypt,
                base_tgt * tgt_mult,
                tough_shadow=tough_shadow,
                heavy_man=man and tough_shadow,
                heavy_zone=zone and not tough_shadow,
            )
        elif np.isfinite(base_tgt):
            base_tgt *= tgt_mult

        if _injury_limited(ctx) and (team, pkey) not in injury_overrides:
            if np.isfinite(base_tgt):
                base_tgt *= 0.50
            if np.isfinite(base_rush):
                base_rush *= 0.50

        out.at[idx, "rules_plays_est"] = script.projected_plays
        out.at[idx, "rules_pass_rate"] = script.projected_pass_attempts / script.projected_plays if script.projected_plays else np.nan
        out.at[idx, "rules_tgt_share"] = base_tgt
        out.at[idx, "rules_rush_share"] = base_rush
        out.at[idx, "rules_ypt"] = base_ypt * mods.pass_eff_mult if np.isfinite(base_ypt) else np.nan
        out.at[idx, "rules_ypc"] = base_ypc * mods.rb_rush_eff_mult if np.isfinite(base_ypc) else np.nan
        out.at[idx, "rules_ypa"] = base_ypa * mods.pass_eff_mult if np.isfinite(base_ypa) else np.nan
        out.at[idx, "rules_catch_rate"] = base_catch
        out.at[idx, "rules_volatility_mult"] = mods.volatility_mult
        out.at[idx, "rules_pass_eff_mult"] = mods.pass_eff_mult
        out.at[idx, "rules_rush_eff_mult"] = mods.rb_rush_eff_mult
        out.at[idx, "rules_applied"] = 1
        out.at[idx, "rules_role"] = role

    out.drop(columns=["_bridge_key"], inplace=True)
    return out


def build_rule_diagnostics() -> pd.DataFrame:
    _, players = load_model_contexts()
    labels = _wr_role_labels(players)
    injury_overrides = _injury_target_overrides(players, labels)
    rows = []
    for p in players:
        if p.offense is None or p.defense is None:
            continue
        script = project_game_script(p.offense, p.defense)
        mods = matchup_multipliers(p.offense, p.defense)
        rows.append({
            "player": p.player, "team": p.team, "opponent": p.opponent,
            "season": p.season, "week": p.week, "position": p.position, "role": p.role,
            "rules_role": labels.get((p.team, _key(p.player)), ""),
            "injury_redistribution": int((p.team, _key(p.player)) in injury_overrides),
            "projected_plays": script.projected_plays,
            "projected_pass_attempts": script.projected_pass_attempts,
            "projected_rush_attempts": script.projected_rush_attempts,
            "lead_prob": script.lead_prob, "trail_prob": script.trail_prob,
            "pass_eff_mult": mods.pass_eff_mult, "rush_eff_mult": mods.rb_rush_eff_mult,
            "wr1_target_mult": mods.wr1_target_mult, "wr1_5_target_mult": mods.wr1_5_target_mult,
            "slot_target_mult": mods.slot_target_mult, "te_target_mult": mods.te_target_mult,
            "rb_rec_target_mult": mods.rb_rec_target_mult, "volatility_mult": mods.volatility_mult,
        })
    return pd.DataFrame(rows)
