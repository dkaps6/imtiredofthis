#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_official_inactive_availability as m78

M78_SHA256 = "d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609"
EXPECTED_M78_ROWS = 1088
EXPECTED_M78_SEASONS = {2024: 544, 2025: 544}
BACKFIELD = {"RB", "FB", "HB"}
RECEIVING = {"WR", "TE", "RB", "FB", "HB"}
WR_TE = {"WR", "TE"}
RB_FB = {"RB", "FB", "HB"}
HISTORY_GAMES = 8


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def n(v):
    return pd.to_numeric(v, errors="coerce")


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def parse_inactives(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in frame.itertuples(index=False):
        for tok in str(r.inactive_tokens or "").split("|"):
            parts = tok.rsplit(":", 2)
            if len(parts) != 3:
                continue
            name, pos, flag = parts
            pos = str(pos).upper().strip()
            rows.append({
                "season": int(r.season),
                "week": int(r.week),
                "team": canon_team(r.team),
                "inactive_name_key": m78.norm_name(name),
                "position": pos,
                "inactive_flag": int(float(flag)) if str(flag).strip() else 1,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("parsed zero M78 inactive tokens")
    return out


def load_m78(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    digest = sha256_file(path)
    if digest != M78_SHA256:
        raise RuntimeError(f"M78 corrected snapshot SHA drift: {digest}")
    x = lower(pd.read_csv(path, low_memory=False))
    if len(x) != EXPECTED_M78_ROWS:
        raise RuntimeError(f"M78 row drift: {len(x)}")
    x["season"] = n(x["season"]).astype(int)
    x["week"] = n(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    counts = {int(k): int(v) for k, v in x["season"].value_counts().to_dict().items()}
    if counts != EXPECTED_M78_SEASONS:
        raise RuntimeError(f"M78 season-count drift: {counts}")
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("M78 duplicate team-week")
    return x, parse_inactives(x)


def roster_identity_table(seasons: list[int], snapshot_meta: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    inventory = []
    for season in seasons:
        raw = lower(m78.load_weekly_rosters(season, snapshot_meta))
        week_col = first_col(raw, ["week"])
        team_col = first_col(raw, ["team", "club_code", "recent_team"])
        id_col = first_col(raw, ["gsis_id", "player_id"])
        name_cols = [c for c in ["full_name", "player_name", "display_name", "football_name"] if c in raw.columns]
        inventory.append({
            "source": "nflverse_roster_weekly",
            "season": season,
            "rows": int(len(raw)),
            "week_field": week_col or "",
            "team_field": team_col or "",
            "player_id_field": id_col or "",
            "name_fields": "|".join(name_cols),
            "week_coverage": float(n(raw[week_col]).notna().mean()) if week_col else 0.0,
            "team_coverage": float(raw[team_col].notna().mean()) if team_col else 0.0,
            "player_id_coverage": float(raw[id_col].notna().mean()) if id_col else 0.0,
        })
        if not week_col or not team_col or not id_col or not name_cols:
            continue
        for r in raw.to_dict("records"):
            try:
                week = int(float(r.get(week_col)))
            except Exception:
                continue
            team = canon_team(r.get(team_col))
            pid = str(r.get(id_col, "") or "").strip()
            if not team or not pid or pid.lower() in {"nan", "none"}:
                continue
            for c in name_cols:
                key = m78.norm_name(r.get(c, ""))
                if key:
                    rows.append({
                        "season": season,
                        "week": week,
                        "team": team,
                        "inactive_name_key": key,
                        "player_id": pid,
                        "roster_name_field": c,
                    })
    ident = pd.DataFrame(rows).drop_duplicates()
    inv = pd.DataFrame(inventory)
    if ident.empty:
        raise RuntimeError("weekly roster identity table empty")
    return ident, inv


def resolve_inactives(inactive: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    key = ["season", "week", "team", "inactive_name_key"]
    grouped = roster.groupby(key, as_index=False).agg(
        player_ids=("player_id", lambda s: "|".join(sorted(set(map(str, s))))),
        player_id_n=("player_id", lambda s: len(set(map(str, s)))),
        roster_name_fields=("roster_name_field", lambda s: "|".join(sorted(set(map(str, s))))),
    )
    out = inactive.merge(grouped, on=key, how="left", validate="many_to_one")
    out["player_id_n"] = n(out["player_id_n"]).fillna(0).astype(int)
    out["resolution_status"] = np.where(
        out["player_id_n"].eq(1), "EXACT_NORMALIZED_UNIQUE",
        np.where(out["player_id_n"].gt(1), "AMBIGUOUS", "UNMAPPED"),
    )
    out["resolved_player_id"] = np.where(out["player_id_n"].eq(1), out["player_ids"].fillna(""), "")
    return out


def load_player_stats(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import nflreadpy as nfl

    parts = []
    inv = []
    chosen: dict[str, str] = {}
    for season in seasons:
        raw = nfl.load_player_stats(seasons=[season], summary_level="week")
        x = lower(raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw))
        if "season_type" in x.columns:
            reg = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x = reg
        season_col = first_col(x, ["season"])
        week_col = first_col(x, ["week"])
        id_col = first_col(x, ["player_id", "gsis_id"])
        team_col = first_col(x, ["recent_team", "team"])
        carry_col = first_col(x, ["carries", "rushing_attempts", "rush_attempts"])
        target_col = first_col(x, ["targets", "receiving_targets"])
        reception_col = first_col(x, ["receptions"])
        if season == seasons[0]:
            chosen = {
                "season": season_col or "", "week": week_col or "", "player_id": id_col or "",
                "team": team_col or "", "carries": carry_col or "", "targets": target_col or "",
                "receptions": reception_col or "",
            }
        else:
            current = {
                "season": season_col or "", "week": week_col or "", "player_id": id_col or "",
                "team": team_col or "", "carries": carry_col or "", "targets": target_col or "",
                "receptions": reception_col or "",
            }
            if current != chosen:
                raise RuntimeError(f"weekly player-stat schema drift season={season}: {current} vs {chosen}")

        inv.append({
            "source": "nflverse_weekly_player_stats",
            "season": season,
            "rows": int(len(x)),
            "season_field": season_col or "",
            "week_field": week_col or "",
            "player_id_field": id_col or "",
            "team_field": team_col or "",
            "carries_field": carry_col or "",
            "targets_field": target_col or "",
            "receptions_field": reception_col or "",
            "player_id_coverage": float(x[id_col].notna().mean()) if id_col else 0.0,
            "carries_coverage": float(n(x[carry_col]).notna().mean()) if carry_col else 0.0,
            "targets_coverage": float(n(x[target_col]).notna().mean()) if target_col else 0.0,
            "receptions_coverage": float(n(x[reception_col]).notna().mean()) if reception_col else 0.0,
        })
        if not all([season_col, week_col, id_col, team_col]):
            continue
        y = pd.DataFrame({
            "season": n(x[season_col]),
            "week": n(x[week_col]),
            "player_id": x[id_col].astype("string").fillna("").str.strip(),
            "team": x[team_col].map(canon_team),
            "carries": n(x[carry_col]) if carry_col else np.nan,
            "targets": n(x[target_col]) if target_col else np.nan,
            "receptions": n(x[reception_col]) if reception_col else np.nan,
        })
        y = y.loc[y["season"].notna() & y["week"].notna() & y["player_id"].ne("")].copy()
        y["season"] = y["season"].astype(int)
        y["week"] = y["week"].astype(int)
        parts.append(y)
    stats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if stats.empty:
        raise RuntimeError("weekly player stats empty")
    return stats, pd.DataFrame(inv), chosen


def prior_usage_audit(resolved: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    relevant = resolved.loc[resolved["position"].isin(RECEIVING | BACKFIELD)].copy()
    index = {
        str(pid): g.sort_values(["season", "week"]).reset_index(drop=True)
        for pid, g in stats.groupby("player_id", sort=False)
    }
    rows = []
    for r in relevant.itertuples(index=False):
        pid = str(r.resolved_player_id or "")
        hist = index.get(pid, pd.DataFrame())
        if not hist.empty:
            hist = hist.loc[
                (hist["season"] < int(r.season))
                | ((hist["season"] == int(r.season)) & (hist["week"] < int(r.week)))
            ].sort_values(["season", "week"]).tail(HISTORY_GAMES)
        carr = n(hist["carries"]).dropna() if not hist.empty and "carries" in hist else pd.Series(dtype=float)
        targ = n(hist["targets"]).dropna() if not hist.empty and "targets" in hist else pd.Series(dtype=float)
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "inactive_name_key": str(r.inactive_name_key),
            "position": str(r.position),
            "resolution_status": str(r.resolution_status),
            "resolved_player_id": pid,
            "prior_usage_games": int(len(hist)),
            "prior_carry_rows": int(len(carr)),
            "prior_target_rows": int(len(targ)),
            "prior_carries_per_game": float(carr.mean()) if len(carr) else np.nan,
            "prior_targets_per_game": float(targ.mean()) if len(targ) else np.nan,
            "strict_prior_max_season": int(hist["season"].max()) if not hist.empty else np.nan,
            "strict_prior_max_week": int(hist.loc[hist["season"].idxmax(), "week"]) if not hist.empty else np.nan,
        })
    return pd.DataFrame(rows)


def teamweek_feasibility(m78_teamweeks: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    u = usage.copy()
    u["run_value"] = np.where(u["position"].isin(BACKFIELD), n(u["prior_carries_per_game"]).fillna(0.0), 0.0)
    u["rec_value"] = np.where(u["position"].isin(RECEIVING), n(u["prior_targets_per_game"]).fillna(0.0), 0.0)
    u["wrte_value"] = np.where(u["position"].isin(WR_TE), n(u["prior_targets_per_game"]).fillna(0.0), 0.0)
    u["rbfb_rec_value"] = np.where(u["position"].isin(RB_FB), n(u["prior_targets_per_game"]).fillna(0.0), 0.0)
    u["mapped"] = u["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE").astype(int)
    agg = u.groupby(["season", "week", "team"], as_index=False).agg(
        relevant_inactive_events=("inactive_name_key", "count"),
        mapped_relevant_events=("mapped", "sum"),
        backfield_run_capacity_lost_raw=("run_value", "sum"),
        receiving_capacity_lost_raw=("rec_value", "sum"),
        wr_te_receiving_capacity_lost_raw=("wrte_value", "sum"),
        rb_fb_receiving_capacity_lost_raw=("rbfb_rec_value", "sum"),
    )
    base = m78_teamweeks[["season", "week", "team"]].copy()
    out = base.merge(agg, on=["season", "week", "team"], how="left", validate="one_to_one")
    fill = [
        "relevant_inactive_events", "mapped_relevant_events", "backfield_run_capacity_lost_raw",
        "receiving_capacity_lost_raw", "wr_te_receiving_capacity_lost_raw", "rb_fb_receiving_capacity_lost_raw",
    ]
    out[fill] = out[fill].fillna(0.0)
    return out


def rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m78", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    teamweeks, inactive = load_m78(a.m78)
    snapshots: list[dict] = []
    roster, roster_inv = roster_identity_table([2024, 2025], snapshots)
    identity = resolve_inactives(inactive, roster)
    stats, stats_inv, chosen = load_player_stats([2023, 2024, 2025])
    usage = prior_usage_audit(identity, stats)
    teamweek = teamweek_feasibility(teamweeks, usage)

    relevant = identity.loc[identity["position"].isin(RECEIVING | BACKFIELD)].copy()
    back_id = relevant.loc[relevant["position"].isin(BACKFIELD)].copy()
    rec_id = relevant.loc[relevant["position"].isin(RECEIVING)].copy()
    back_usage = usage.loc[usage["position"].isin(BACKFIELD) & usage["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE")].copy()
    rec_usage = usage.loc[usage["position"].isin(RECEIVING) & usage["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE")].copy()

    carry_schema_ok = bool(chosen.get("carries"))
    target_schema_ok = bool(chosen.get("targets"))
    carry_cov_min = float(stats_inv["carries_coverage"].min()) if len(stats_inv) else 0.0
    target_cov_min = float(stats_inv["targets_coverage"].min()) if len(stats_inv) else 0.0
    back_id_rate = rate(back_id["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE"))
    rec_id_rate = rate(rec_id["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE"))
    back_prior1 = rate(back_usage["prior_carry_rows"].ge(1))
    back_prior3 = rate(back_usage["prior_carry_rows"].ge(3))
    rec_prior1 = rate(rec_usage["prior_target_rows"].ge(1))
    rec_prior3 = rate(rec_usage["prior_target_rows"].ge(3))

    # Mechanical Week-1 support: a mapped Week-1 event can only use a season < target season.
    wk1 = usage.loc[usage["week"].eq(1) & usage["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE")].copy()
    wk1_with_history = wk1.loc[wk1["prior_usage_games"].ge(1)].copy()
    wk1_prior_only = bool(
        len(wk1_with_history) > 0
        and (n(wk1_with_history["strict_prior_max_season"]) < n(wk1_with_history["season"])).all()
    )

    no_target_outcomes = True
    no_sportsbook = True
    materially_distinct = True
    current_deployable_contract = True

    back_gates = {
        "m78_exact_contract": True,
        "backfield_identity_resolution_ge_0_95": bool(back_id_rate >= 0.95),
        "exact_carries_field_2023_2025": carry_schema_ok,
        "carries_coverage_ge_0_99_every_season": bool(carry_cov_min >= 0.99),
        "mapped_backfield_prior1_ge_0_90": bool(back_prior1 >= 0.90),
        "mapped_backfield_prior3_ge_0_75": bool(back_prior3 >= 0.75),
        "week1_prior_season_supported": wk1_prior_only,
        "no_target_or_sportsbook_required": bool(no_target_outcomes and no_sportsbook),
        "materially_distinct_from_m77_m79": materially_distinct,
    }
    rec_gates = {
        "m78_exact_contract": True,
        "receiving_identity_resolution_ge_0_95": bool(rec_id_rate >= 0.95),
        "exact_targets_field_2023_2025": target_schema_ok,
        "targets_coverage_ge_0_99_every_season": bool(target_cov_min >= 0.99),
        "mapped_receiving_prior1_ge_0_90": bool(rec_prior1 >= 0.90),
        "mapped_receiving_prior3_ge_0_75": bool(rec_prior3 >= 0.75),
        "week1_prior_season_supported": wk1_prior_only,
        "no_target_or_sportsbook_required": bool(no_target_outcomes and no_sportsbook),
        "materially_distinct_from_m77_m79": materially_distinct,
    }
    back_ok = all(back_gates.values())
    rec_ok = all(rec_gates.values())
    if back_ok and rec_ok:
        disposition = "DIRECTIONAL_PERSONNEL_CONSEQUENCE_SOURCE_ELIGIBLE"
    elif back_ok:
        disposition = "BACKFIELD_ONLY_SOURCE_ELIGIBLE_DIRECTIONAL_FAMILY_BLOCKED"
    elif rec_ok:
        disposition = "RECEIVING_ONLY_SOURCE_ELIGIBLE_DIRECTIONAL_FAMILY_BLOCKED"
    elif not carry_schema_ok or not target_schema_ok:
        disposition = "SOURCE_INELIGIBLE_SCHEMA"
    else:
        disposition = "SOURCE_INELIGIBLE_IDENTITY_OR_USAGE_COVERAGE"

    source_inventory = pd.concat([roster_inv, stats_inv], ignore_index=True, sort=False)
    snapshot_df = pd.DataFrame(snapshots)
    if not snapshot_df.empty:
        snapshot_df.to_csv(a.out_dir / "directional_personnel_source_snapshots.csv", index=False)
    source_inventory.to_csv(a.out_dir / "directional_personnel_source_inventory.csv", index=False)
    identity.to_csv(a.out_dir / "directional_personnel_identity_audit.csv", index=False)
    usage.to_csv(a.out_dir / "directional_personnel_prior_usage_audit.csv", index=False)
    teamweek.to_csv(a.out_dir / "directional_personnel_teamweek_feasibility.csv", index=False)

    result = {
        "migration": "QB_PASS_RATE_DIRECTIONAL_PERSONNEL_SOURCE_AUDIT_V1",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "m78_csv_sha256": M78_SHA256,
        "m78_rows": int(len(teamweeks)),
        "inactive_tokens_total": int(len(inactive)),
        "relevant_skill_inactive_tokens": int(len(relevant)),
        "backfield_inactive_tokens": int(len(back_id)),
        "receiving_skill_inactive_tokens": int(len(rec_id)),
        "backfield_identity_resolution_rate": back_id_rate,
        "receiving_identity_resolution_rate": rec_id_rate,
        "chosen_player_stat_fields": chosen,
        "minimum_carries_field_coverage_2023_2025": carry_cov_min,
        "minimum_targets_field_coverage_2023_2025": target_cov_min,
        "mapped_backfield_prior1_rate": back_prior1,
        "mapped_backfield_prior3_rate": back_prior3,
        "mapped_receiving_prior1_rate": rec_prior1,
        "mapped_receiving_prior3_rate": rec_prior3,
        "week1_mapped_events_with_prior_history": int(len(wk1_with_history)),
        "week1_prior_season_only_verified": wk1_prior_only,
        "target_outcomes_read": False,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "current_season_conceptual_contract_available": current_deployable_contract,
        "backfield_run_capacity_gates": back_gates,
        "receiving_capacity_gates": rec_gates,
        "backfield_run_capacity_source_eligible": back_ok,
        "receiving_capacity_source_eligible": rec_ok,
        "disposition": disposition,
    }
    (a.out_dir / "directional_personnel_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
