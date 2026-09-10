#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_official_inactive_availability as m78

EXPECTED_ROWS = 884
EXPECTED_SEASONS = {2024: 444, 2025: 440}
HISTORY_GAMES = 8
REQ_PBP = {
    "week", "posteam", "qb_dropback", "rush_attempt", "qb_scramble", "qb_kneel",
    "passer_player_id", "rusher_player_id",
}


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s):
    return pd.to_numeric(s, errors="coerce")


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def load_target_keys(root: Path) -> pd.DataFrame:
    path = one(root, "m89_corrected_qb_common_trace.csv")
    keys = ["season", "week", "team", "player_clean_key"]
    x = pd.read_csv(path, usecols=keys, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(m78.norm_name)
    x = x.loc[x["season"].isin([2024, 2025]) & x["week"].between(1, 18)].copy()
    counts = {int(k): int(v) for k, v in x["season"].value_counts().to_dict().items()}
    if len(x) != EXPECTED_ROWS or counts != EXPECTED_SEASONS:
        raise RuntimeError(f"target key drift rows={len(x)} seasons={counts}")
    if x.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate target QB key")
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate target QB team-week")
    return x.sort_values(["season", "week", "team"]).reset_index(drop=True)


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def load_rosters(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots: list[dict] = []
    rows = []
    inv = []
    for season in seasons:
        r = lower(m78.load_weekly_rosters(season, snapshots))
        week_col = first_col(r, ["week"])
        team_col = first_col(r, ["team", "club_code", "recent_team"])
        id_col = first_col(r, ["gsis_id", "player_id"])
        pos_col = first_col(r, ["position", "position_group"])
        name_cols = [c for c in ["full_name", "player_name", "display_name", "football_name"] if c in r.columns]
        inv.append({
            "source": "nflverse_roster_weekly", "season": season, "rows": int(len(r)),
            "week_field": week_col or "", "team_field": team_col or "", "player_id_field": id_col or "",
            "position_field": pos_col or "", "name_fields": "|".join(name_cols),
            "id_coverage": float(r[id_col].notna().mean()) if id_col else 0.0,
            "position_coverage": float(r[pos_col].notna().mean()) if pos_col else 0.0,
        })
        if not all([week_col, team_col, id_col, pos_col]) or not name_cols:
            continue
        for rec in r.to_dict("records"):
            try:
                week = int(float(rec.get(week_col)))
            except Exception:
                continue
            team = canon_team(rec.get(team_col))
            pid = str(rec.get(id_col, "") or "").strip()
            pos = str(rec.get(pos_col, "") or "").upper().strip()
            if not team or not pid or pid.lower() in {"nan", "none"}:
                continue
            for nc in name_cols:
                nk = m78.norm_name(rec.get(nc, ""))
                if nk:
                    rows.append({
                        "season": season, "week": week, "team": team,
                        "player_name_key": nk, "player_id": pid, "position": pos,
                        "name_field": nc,
                    })
    roster = pd.DataFrame(rows).drop_duplicates()
    if roster.empty:
        raise RuntimeError("roster identity table empty")
    return roster, pd.concat([pd.DataFrame(inv), pd.DataFrame(snapshots)], ignore_index=True, sort=False)


def resolve_target_qbs(target: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    key = ["season", "week", "team", "player_name_key"]
    q = roster.loc[roster["position"].astype(str).str.upper().str.startswith("QB")].copy()
    g = q.groupby(key, as_index=False).agg(
        player_ids=("player_id", lambda s: "|".join(sorted(set(map(str, s))))),
        player_id_n=("player_id", lambda s: len(set(map(str, s)))),
    )
    t = target.rename(columns={"player_clean_key": "player_name_key"})
    out = t.merge(g, on=key, how="left", validate="one_to_one")
    out["player_id_n"] = num(out["player_id_n"]).fillna(0).astype(int)
    out["resolution_status"] = np.where(
        out["player_id_n"].eq(1), "EXACT_NORMALIZED_UNIQUE",
        np.where(out["player_id_n"].gt(1), "AMBIGUOUS", "UNMAPPED"),
    )
    out["resolved_player_id"] = np.where(out["player_id_n"].eq(1), out["player_ids"].fillna(""), "")
    return out


def load_pbp(seasons: list[int], roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    qbroster = roster.loc[roster["position"].astype(str).str.upper().str.startswith("QB"),
                          ["season", "week", "team", "player_id"]].drop_duplicates()
    qbset = set(map(tuple, qbroster[["season", "week", "team", "player_id"]].astype(str).to_numpy()))
    games = []
    inv = []
    for season in seasons:
        raw = nfl.load_pbp(seasons=[season])
        p = lower(raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw))
        if "season_type" in p.columns:
            p = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        elif "game_type" in p.columns:
            p = p.loc[p["game_type"].astype(str).str.upper().eq("REG")].copy()
        missing = sorted(REQ_PBP - set(p.columns))
        inv.append({
            "source": "nflverse_pbp", "season": season, "rows": int(len(p)),
            "missing_required_fields": "|".join(missing),
            **{f"coverage_{c}": float(p[c].notna().mean()) if c in p.columns else 0.0 for c in sorted(REQ_PBP)},
        })
        if missing:
            continue
        p["season"] = season
        p["week"] = num(p["week"])
        p["team"] = p["posteam"].map(canon_team)
        p["passer_player_id"] = p["passer_player_id"].astype("string").fillna("").str.strip()
        p["rusher_player_id"] = p["rusher_player_id"].astype("string").fillna("").str.strip()
        db = num(p["qb_dropback"]).fillna(0).eq(1)
        rush = num(p["rush_attempt"]).fillna(0).eq(1)
        scramble = num(p["qb_scramble"]).fillna(0).eq(1)
        kneel = num(p["qb_kneel"]).fillna(0).eq(1)

        d = p.loc[db & p["passer_player_id"].ne(""), ["season", "week", "team", "passer_player_id"]].copy()
        d = d.rename(columns={"passer_player_id": "player_id"})
        d["dropbacks"] = 1
        d = d.groupby(["season", "week", "team", "player_id"], as_index=False)["dropbacks"].sum()

        r = p.loc[rush & p["rusher_player_id"].ne(""),
                  ["season", "week", "team", "rusher_player_id", "qb_scramble", "qb_kneel"]].copy()
        r = r.rename(columns={"rusher_player_id": "player_id"})
        # Use roster QB position to avoid treating non-QB rushers as QB rushes.
        r["_qb"] = [
            (str(int(s)), str(int(w)), str(t), str(pid)) in qbset
            for s, w, t, pid in zip(r["season"], r["week"], r["team"], r["player_id"])
        ]
        r = r.loc[r["_qb"]].copy()
        r["qb_rushes"] = 1
        r["designed_runs"] = (
            num(r["qb_scramble"]).fillna(0).eq(0) & num(r["qb_kneel"]).fillna(0).eq(0)
        ).astype(int)
        r = r.groupby(["season", "week", "team", "player_id"], as_index=False).agg(
            qb_rushes=("qb_rushes", "sum"), designed_runs=("designed_runs", "sum")
        )

        g = d.merge(r, on=["season", "week", "team", "player_id"], how="outer")
        g[["dropbacks", "qb_rushes", "designed_runs"]] = g[["dropbacks", "qb_rushes", "designed_runs"]].fillna(0.0)
        g["snaps"] = g["dropbacks"] + g["qb_rushes"]
        g["designed_run_rate"] = np.where(g["snaps"] > 0, g["designed_runs"] / g["snaps"], np.nan)
        games.append(g)
    hist = pd.concat(games, ignore_index=True) if games else pd.DataFrame()
    if hist.empty:
        raise RuntimeError("QB designed-run history empty")
    return hist, pd.DataFrame(inv)


def prior_history(target: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    idx = {str(pid): g.sort_values(["season", "week"]).reset_index(drop=True)
           for pid, g in hist.groupby("player_id", sort=False)}
    rows = []
    for r in target.itertuples(index=False):
        pid = str(r.resolved_player_id or "")
        h = idx.get(pid, pd.DataFrame())
        if not h.empty:
            h = h.loc[
                (h["season"] < int(r.season)) |
                ((h["season"] == int(r.season)) & (h["week"] < int(r.week)))
            ].sort_values(["season", "week"]).tail(HISTORY_GAMES)
        latest = h.iloc[-1] if len(h) else None
        snaps = float(num(h["snaps"]).sum()) if len(h) else 0.0
        rows.append({
            "season": int(r.season), "week": int(r.week), "team": str(r.team),
            "player_name_key": str(r.player_name_key), "resolved_player_id": pid,
            "resolution_status": str(r.resolution_status),
            "prior_games": int(len(h)),
            "recent8_designed_runs": float(num(h["designed_runs"]).sum()) if len(h) else np.nan,
            "recent8_qb_rushes": float(num(h["qb_rushes"]).sum()) if len(h) else np.nan,
            "recent8_dropbacks": float(num(h["dropbacks"]).sum()) if len(h) else np.nan,
            "recent8_snaps": snaps if len(h) else np.nan,
            "recent8_designed_run_rate": float(num(h["designed_runs"]).sum() / snaps) if snaps > 0 else np.nan,
            "latest_prior_designed_runs": float(latest.designed_runs) if latest is not None else np.nan,
            "latest_prior_qb_rushes": float(latest.qb_rushes) if latest is not None else np.nan,
            "latest_prior_dropbacks": float(latest.dropbacks) if latest is not None else np.nan,
            "latest_prior_snaps": float(latest.snaps) if latest is not None else np.nan,
            "latest_prior_designed_run_rate": float(latest.designed_run_rate) if latest is not None else np.nan,
            "strict_prior_max_season": int(h["season"].max()) if len(h) else np.nan,
            "strict_prior_max_week": int(h.iloc[-1]["week"]) if len(h) else np.nan,
        })
    return pd.DataFrame(rows)


def architecture_audit(repo: Path) -> pd.DataFrame:
    files = {
        "builder": repo / "scripts/build/pbp_features.py",
        "metrics": repo / "scripts/metrics_v2.py",
        "rules": repo / "scripts/modeling/rules_v2.py",
        "contracts": repo / "scripts/artifact_contracts.py",
    }
    text = {k: p.read_text(encoding="utf-8") for k, p in files.items()}
    rows = [
        {"check": "builder_excludes_scrambles", "passed": "qb_scramble\"].eq(0)" in text["builder"], "evidence": "build_qb_run_metrics"},
        {"check": "builder_excludes_kneels", "passed": "qb_kneel\"].eq(0)" in text["builder"], "evidence": "build_qb_run_metrics"},
        {"check": "metrics_strict_prior_week", "passed": "w.lt(int(week))" in text["metrics"], "evidence": "metrics_v2._join_optional"},
        {"check": "rules_fixed_057", "passed": "pass_share = 0.57" in text["rules"], "evidence": "rules_v2.project_game_script"},
        {"check": "artifact_contract_declares_designed_run_rate", "passed": "designed_run_rate" in text["contracts"], "evidence": "artifact_contracts"},
    ]
    # Production-code literal scan. Construction/join/contract are allowed; any other consumer fails orphan claim.
    refs = []
    for p in (repo / "scripts").rglob("*.py"):
        rel = p.relative_to(repo).as_posix()
        if rel.startswith("scripts/backtest/"):
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "designed_run_rate" in body:
            refs.append(rel)
    allowed = {"scripts/build/pbp_features.py", "scripts/metrics_v2.py", "scripts/artifact_contracts.py"}
    unexpected = sorted(set(refs) - allowed)
    rows.append({
        "check": "no_unexpected_downstream_consumer", "passed": len(unexpected) == 0,
        "evidence": "refs=" + "|".join(sorted(refs)) + ";unexpected=" + "|".join(unexpected),
    })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    target = load_target_keys(a.m89_root)
    roster, roster_inv = load_rosters([2023, 2024, 2025])
    resolved = resolve_target_qbs(target, roster)
    hist, pbp_inv = load_pbp([2023, 2024, 2025], roster)
    prior = prior_history(resolved, hist)
    arch = architecture_audit(a.repo_root)

    overall_id = float(resolved["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE").mean())
    season_id = {str(s): float(resolved.loc[resolved.season.eq(s), "resolution_status"].eq("EXACT_NORMALIZED_UNIQUE").mean()) for s in [2024, 2025]}
    rp = prior.loc[prior["resolution_status"].eq("EXACT_NORMALIZED_UNIQUE")].copy()
    prior1 = float(rp["prior_games"].ge(1).mean()) if len(rp) else 0.0
    prior3 = float(rp["prior_games"].ge(3).mean()) if len(rp) else 0.0
    wk1 = rp.loc[rp["week"].eq(1) & rp["prior_games"].ge(1)].copy()
    wk1_safe = bool(len(wk1) > 0 and (num(wk1["strict_prior_max_season"]) < num(wk1["season"])).all())
    pbp_schema = bool(len(pbp_inv) == 3 and pbp_inv["missing_required_fields"].fillna("").eq("").all())
    arch_map = arch.set_index("check")["passed"].astype(bool).to_dict()

    gates = {
        "exact_884_target_keys": len(target) == 884,
        "pbp_2023_2025_required_schema": pbp_schema,
        "target_qb_identity_ge_0_98": overall_id >= 0.98,
        "target_qb_identity_2024_ge_0_97": season_id["2024"] >= 0.97,
        "target_qb_identity_2025_ge_0_97": season_id["2025"] >= 0.97,
        "resolved_targets_prior1_ge_0_92": prior1 >= 0.92,
        "resolved_targets_prior3_ge_0_85": prior3 >= 0.85,
        "week1_prior_season_only": wk1_safe,
        "historical_definition_matches_builder": bool(arch_map.get("builder_excludes_scrambles") and arch_map.get("builder_excludes_kneels")),
        "production_join_strict_prior": bool(arch_map.get("metrics_strict_prior_week")),
        "upstream_pass_share_fixed_0_57": bool(arch_map.get("rules_fixed_057")),
        "no_unexpected_downstream_consumer": bool(arch_map.get("no_unexpected_downstream_consumer")),
        "zero_sportsbook_inputs": True,
        "zero_target_outcomes_read": True,
        "zero_model_fitting": True,
        "zero_production_changes": True,
    }
    all_pass = all(gates.values())
    if all_pass:
        disposition = "QB_DESIGNED_RUN_LINKAGE_SOURCE_ELIGIBLE"
    elif not pbp_schema:
        disposition = "QB_DESIGNED_RUN_LINKAGE_SOURCE_INELIGIBLE_SCHEMA"
    elif not arch_map.get("no_unexpected_downstream_consumer", False):
        disposition = "QB_DESIGNED_RUN_LINKAGE_NOT_ORPHANED_ALREADY_CONSUMED"
    else:
        disposition = "QB_DESIGNED_RUN_LINKAGE_SOURCE_INELIGIBLE_IDENTITY_OR_HISTORY"

    pd.concat([roster_inv, pbp_inv], ignore_index=True, sort=False).to_csv(a.out_dir / "qb_designed_run_source_inventory.csv", index=False)
    resolved.to_csv(a.out_dir / "qb_designed_run_target_identity_audit.csv", index=False)
    prior.to_csv(a.out_dir / "qb_designed_run_prior_history_audit.csv", index=False)
    arch.to_csv(a.out_dir / "qb_designed_run_architecture_audit.csv", index=False)
    result = {
        "migration": "QB_PASS_RATE_DESIGNED_RUN_SOURCE_AUDIT_V1",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "target_rows": int(len(target)),
        "target_season_rows": {str(k): int(v) for k, v in target.season.value_counts().sort_index().to_dict().items()},
        "target_qb_identity_resolution_rate": overall_id,
        "target_qb_identity_resolution_by_season": season_id,
        "resolved_target_prior1_rate": prior1,
        "resolved_target_prior3_rate": prior3,
        "week1_resolved_with_prior_history": int(len(wk1)),
        "week1_prior_season_only_verified": wk1_safe,
        "pbp_source_audit": pbp_inv.to_dict("records"),
        "architecture_checks": {str(r.check): bool(r.passed) for r in arch.itertuples(index=False)},
        "gates": gates,
        "all_gates_pass": all_pass,
        "sportsbook_inputs_used": False,
        "target_outcomes_read": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }
    (a.out_dir / "qb_designed_run_source_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
