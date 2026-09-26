#!/usr/bin/env python3
"""Diagnostic-only audit of availability -> opportunity rule ordering.

Uses an immutable no-odds Full Slate artifact restored into data/ and invokes
current production Bayesian/rule/target-entitlement code directly.

No target-game outcomes, sportsbook inputs, fitted parameters, candidate
projection variants, or production mutations are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import build_bayesian_baseline, apply_bayesian_to_metrics
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.simulation_v2 import _top_n_shares

DATA = Path("data")
OUT = Path("outputs/availability_opportunity_rule_order_audit_v1")
SKILL = {"RB", "FB", "WR", "TE"}


def _read(name: str) -> pd.DataFrame:
    p = DATA / name
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"required artifact missing: {p}")
    x = pd.read_csv(p, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df["team"].astype(str), df["player_clean_key"].astype(str)))


def _latest_prior(logs: pd.DataFrame, *, team: str, key: str, season: int, week: int):
    x = logs.copy()
    s = pd.to_numeric(x["season"], errors="coerce")
    w = pd.to_numeric(x["week"], errors="coerce")
    m = (
        x["team"].astype(str).eq(team)
        & x["player_clean_key"].astype(str).eq(key)
        & (s.lt(season) | (s.eq(season) & w.lt(week)))
    )
    q = x.loc[m].copy()
    if q.empty:
        return None
    q["_season"] = pd.to_numeric(q["season"], errors="coerce")
    q["_week"] = pd.to_numeric(q["week"], errors="coerce")
    return q.sort_values(["_season", "_week"], kind="mergesort").iloc[-1]


def _rush_team_audit(rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    player_rows = []
    unique = rules.sort_values(["team", "player_clean_key"], kind="mergesort").drop_duplicates(
        ["team", "player_clean_key"], keep="last"
    )
    for team, g in unique.groupby("team", sort=True):
        shares = pd.to_numeric(g["rules_rush_share"], errors="coerce").fillna(0.0).to_numpy(float)
        selected = _top_n_shares(shares, 5)
        clean = np.clip(np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.95)
        raw_sum = float(clean.sum())
        used = clean.copy()
        if raw_sum > 0.95:
            used *= 0.95 / raw_sum
        residual = max(0.0, 1.0 - float(used.sum()))
        rows.append({
            "team": str(team),
            "rush_top5_rules_sum": raw_sum,
            "rush_modeled_probability_sum": float(used.sum()),
            "rush_residual_probability": residual,
        })
        for (_, r), raw, prob in zip(g.iterrows(), clean, used):
            player_rows.append({
                "team": str(team),
                "player": r.get("player"),
                "player_clean_key": r.get("player_clean_key"),
                "position": r.get("position"),
                "rules_rush_share_after_top5": float(raw),
                "final_rush_allocation_probability": float(prob),
            })
    return pd.DataFrame(rows), pd.DataFrame(player_rows)


def main() -> int:
    availability = _read("current_player_availability.csv")
    active_roles = _read("roles_current_production_eligible_v1.csv")
    consensus = _read("player_form_consensus.csv")
    contexts = _read("model_context_bridge.csv")
    logs = _read("player_game_logs.csv")

    seasons = sorted(pd.to_numeric(consensus["season"], errors="coerce").dropna().astype(int).unique())
    weeks = sorted(pd.to_numeric(consensus["week"], errors="coerce").dropna().astype(int).unique())
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"expected one target season/week, got seasons={seasons} weeks={weeks}")
    season, week = seasons[0], weeks[0]

    eligible_teams = set(active_roles["team"].dropna().astype(str))
    unavailable = availability.loc[
        pd.to_numeric(availability["definitive_unavailable"], errors="coerce").fillna(0).eq(1)
        & availability["position_group"].astype(str).str.upper().isin(SKILL)
        & availability["team"].astype(str).isin(eligible_teams)
    ].copy()

    role_keys = _keys(active_roles)
    pf_keys = _keys(consensus)
    # model_context_bridge predates explicit key column; attach exact key from consensus.
    context_key = contexts.merge(
        consensus[["team", "player", "player_clean_key"]].drop_duplicates(),
        on=["team", "player"],
        how="left",
        validate="one_to_one",
    )
    context_keys = _keys(context_key.dropna(subset=["player_clean_key"]))

    # Production-exact player means -> rules.
    bayes = build_bayesian_baseline(consensus)
    metrics = apply_bayesian_to_metrics(consensus, baseline=bayes)

    # Target entitlement requires event_id. Use the already-materialized canonical
    # model-context game identity from the same immutable Full Slate artifact.
    identity = context_key[["team", "player_clean_key", "game_id"]].drop_duplicates().rename(
        columns={"game_id": "event_id"}
    )
    metrics = metrics.merge(identity, on=["team", "player_clean_key"], how="left", validate="many_to_one")
    if metrics["event_id"].isna().any():
        sample = metrics.loc[metrics["event_id"].isna(), ["team", "player", "player_clean_key"]].head(20)
        raise RuntimeError(f"missing game identity before rule audit: {sample.to_dict('records')}")

    rules = apply_rules_to_metrics(metrics)
    entitlement, target_trace = materialize_target_entitlement(rules)

    # One row/player for diagnostics.
    rule_players = rules.sort_values(["team", "player_clean_key"], kind="mergesort").drop_duplicates(
        ["team", "player_clean_key"], keep="last"
    )
    redistribution_rows = int(pd.to_numeric(rule_players["rules_injury_redistribution"], errors="coerce").fillna(0).sum())

    target_team = (
        target_trace.groupby(["event_id", "team"], as_index=False)
        .agg(
            target_raw_rules_sum=("raw_team_sum", "first"),
            target_post_m38_sum=("post_m38_team_sum", "first"),
            target_modeled_probability_sum=("modeled_player_sum", "first"),
            target_residual_probability=("residual_share", "first"),
        )
    )
    rush_team, rush_players = _rush_team_audit(rule_players)
    team_audit = target_team.merge(rush_team, on="team", how="left", validate="one_to_one")

    vacancy_rows = []
    for _, r in unavailable.sort_values(["team", "position_group", "player"], kind="mergesort").iterrows():
        team = str(r["team"])
        key = str(r["player_clean_key"])
        prior = _latest_prior(logs, team=team, key=key, season=season, week=week)
        team_row = team_audit.loc[team_audit["team"].eq(team)]
        if team_row.empty:
            raise RuntimeError(f"eligible vacancy team absent from team audit: {team}")
        t = team_row.iloc[0]
        vacancy_rows.append({
            "team": team,
            "player": r["player"],
            "player_clean_key": key,
            "position_group": str(r["position_group"]).upper(),
            "final_availability_state": r["final_availability_state"],
            "in_eligible_roles": int((team, key) in role_keys),
            "in_player_form": int((team, key) in pf_keys),
            "in_model_context": int((team, key) in context_keys),
            "strict_prior_season": np.nan if prior is None else prior.get("season"),
            "strict_prior_week": np.nan if prior is None else prior.get("week"),
            "strict_prior_tgt_share_game": np.nan if prior is None else prior.get("tgt_share_game"),
            "strict_prior_rush_share_game": np.nan if prior is None else prior.get("rush_share_game"),
            "strict_prior_targets": np.nan if prior is None else prior.get("targets"),
            "strict_prior_carries": np.nan if prior is None else prior.get("carries"),
            "current_target_rules_sum": float(t["target_raw_rules_sum"]),
            "current_target_modeled_probability_sum": float(t["target_modeled_probability_sum"]),
            "current_target_residual_probability": float(t["target_residual_probability"]),
            "current_rush_top5_rules_sum": float(t["rush_top5_rules_sum"]),
            "current_rush_modeled_probability_sum": float(t["rush_modeled_probability_sum"]),
            "current_rush_residual_probability": float(t["rush_residual_probability"]),
        })
    vacancy = pd.DataFrame(vacancy_rows)

    if vacancy.empty:
        disposition = "INSUFFICIENT_CURRENT_ARTIFACTS_CODE_PATH_ONLY"
    else:
        survived = int(vacancy[["in_eligible_roles", "in_player_form", "in_model_context"]].to_numpy().sum())
        definitive_wr = int(vacancy["position_group"].eq("WR").sum())
        definitive_rbfb = int(vacancy["position_group"].isin(["RB", "FB"]).sum())
        if survived != 0:
            disposition = "NO_RULE_ORDER_GAP"
        elif definitive_wr > 0 and definitive_rbfb > 0:
            disposition = "AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED"
        elif definitive_wr > 0:
            disposition = "WR_DEFINITIVE_VACANCY_RULE_UNREACHABLE_CONFIRMED"
        elif definitive_rbfb > 0:
            disposition = "RB_DEFINITIVE_VACANCY_NO_TRANSFER_CONFIRMED"
        else:
            disposition = "INSUFFICIENT_CURRENT_ARTIFACTS_CODE_PATH_ONLY"

    OUT.mkdir(parents=True, exist_ok=True)
    vacancy.to_csv(OUT / "vacancy_rows.csv", index=False)
    team_audit.to_csv(OUT / "team_opportunity_mass.csv", index=False)
    rule_players[[
        "team", "player", "player_clean_key", "position",
        "rules_tgt_share", "rules_rush_share", "rules_injury_redistribution",
    ]].to_csv(OUT / "rule_player_rows.csv", index=False)
    rush_players.to_csv(OUT / "rush_allocation_rows.csv", index=False)

    payload = {
        "disposition": disposition,
        "season": season,
        "week": week,
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_read": 0,
        "production_mutations": 0,
        "eligible_teams": len(eligible_teams),
        "definitive_unavailable_skill_players": int(len(vacancy)),
        "definitive_unavailable_wr": int(vacancy["position_group"].eq("WR").sum()) if not vacancy.empty else 0,
        "definitive_unavailable_te": int(vacancy["position_group"].eq("TE").sum()) if not vacancy.empty else 0,
        "definitive_unavailable_rb_fb": int(vacancy["position_group"].isin(["RB", "FB"]).sum()) if not vacancy.empty else 0,
        "unavailable_survived_eligible_roles": int(vacancy["in_eligible_roles"].sum()) if not vacancy.empty else 0,
        "unavailable_survived_player_form": int(vacancy["in_player_form"].sum()) if not vacancy.empty else 0,
        "unavailable_survived_model_context": int(vacancy["in_model_context"].sum()) if not vacancy.empty else 0,
        "production_reachable_rules_injury_redistribution_rows": redistribution_rows,
        "vacancy_team_target_residual_min": float(team_audit.loc[team_audit.team.isin(vacancy.team.unique()), "target_residual_probability"].min()) if not vacancy.empty else None,
        "vacancy_team_target_residual_max": float(team_audit.loc[team_audit.team.isin(vacancy.team.unique()), "target_residual_probability"].max()) if not vacancy.empty else None,
        "vacancy_team_rush_residual_min": float(team_audit.loc[team_audit.team.isin(vacancy.team.unique()), "rush_residual_probability"].min()) if not vacancy.empty else None,
        "vacancy_team_rush_residual_max": float(team_audit.loc[team_audit.team.isin(vacancy.team.unique()), "rush_residual_probability"].max()) if not vacancy.empty else None,
        "interpretation": (
            "Definitive-unavailable skill players are removed before PlayerContext. "
            "The legacy WR injury redistribution rule therefore cannot observe a definitive-unavailable WR. "
            "On the preserved Week-3 state, surviving rule-adjusted opportunity already exceeds the allocator cap "
            "on affected teams, so vacancies are absorbed through generic normalization of survivors rather than "
            "a vacancy-specific transfer. The frozen RB Vacancy V1 candidate is not included here."
        ),
    }
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    rendered = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    (OUT / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
