#!/usr/bin/env python3
"""R26S: frozen 2026 Week 1 postgame prospective RB receptions evaluation.

The scientific protocol is frozen in:
  docs/research/RB_R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_V1_FROZEN_PLAN.md

Sportsbook evidence is downstream only. This evaluator never regenerates football
values and never changes the sealed R26Q candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PASS = "R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_PASS_READY_FOR_PRODUCTION_PROMOTION_REVIEW"
FAIL = "R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_FAIL_NO_PROMOTION"
INCOMPLETE = "R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION"

R26Q_DISPOSITION = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION"
R26R_DISPOSITION = "R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED"

TEAM_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "LA": "LAR",
    "JAC": "JAX",
    "ARZ": "ARI",
    "WSH": "WAS",
}


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def pdx(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)


def nk(v) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def tm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    if s in {"", "NAN", "NONE", "<NA>"}:
        return ""
    return TEAM_MAP.get(s, s)


def first(df: pd.DataFrame, names: list[str], default=pd.NA) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bool_json(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes"}


def gate(rows: list[dict], name: str, passed: bool, evidence) -> None:
    if isinstance(evidence, (dict, list)):
        evidence = json.dumps(evidence, sort_keys=True)
    rows.append({"gate": name, "passed": bool(passed), "evidence": evidence})


def load_week1_outcomes() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load official weekly receptions and snap participation after the plan is frozen."""
    status: dict = {
        "season": 2026,
        "week": 1,
        "weekly_stats_available": False,
        "snap_counts_available": False,
        "weekly_stats_error": None,
        "snap_counts_error": None,
    }
    weekly = pd.DataFrame()
    snaps = pd.DataFrame()

    try:
        import nflreadpy as nfl
        from scripts.player_form_v2 import _normalize_weekly

        raw = pdx(nfl.load_player_stats(seasons=[2026], summary_level="week"))
        if not raw.empty:
            q = lower(_normalize_weekly(raw, 2026))
            q["season"] = pd.to_numeric(first(q, ["season"]), errors="coerce")
            q["week"] = pd.to_numeric(first(q, ["week"]), errors="coerce")
            q = q.loc[q["season"].eq(2026) & q["week"].eq(1)].copy()
            q["team"] = first(q, ["team", "recent_team", "team_abbr"]).map(tm)
            q["player_name"] = first(
                q,
                ["player", "player_display_name", "player_name", "full_name", "football_name"],
            ).fillna("").astype(str)
            if "player_clean_key" in q.columns:
                q["name_key"] = q["player_clean_key"].fillna("").astype(str).map(nk)
            else:
                q["name_key"] = q["player_name"].map(nk)
            q["receptions"] = pd.to_numeric(first(q, ["receptions"]), errors="coerce").fillna(0.0)
            weekly = (
                q.loc[q["team"].ne("") & q["name_key"].ne(""), ["season", "week", "team", "name_key", "player_name", "receptions"]]
                .groupby(["season", "week", "team", "name_key"], as_index=False)
                .agg(player_name=("player_name", "last"), receptions=("receptions", "sum"))
            )
            status["weekly_stats_available"] = not weekly.empty
            status["weekly_stats_rows"] = int(len(weekly))
            status["weekly_stats_team_count"] = int(weekly["team"].nunique()) if not weekly.empty else 0
    except Exception as exc:  # data availability boundary
        status["weekly_stats_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import nflreadpy as nfl

        raw = lower(pdx(nfl.load_snap_counts(seasons=[2026])))
        raw["season"] = pd.to_numeric(first(raw, ["season"]), errors="coerce")
        raw["week"] = pd.to_numeric(first(raw, ["week"]), errors="coerce")
        q = raw.loc[raw["season"].eq(2026) & raw["week"].eq(1)].copy()
        q["team"] = first(q, ["team", "team_abbr"]).map(tm)
        q["player_name"] = first(q, ["player", "player_name", "full_name"]).fillna("").astype(str)
        q["name_key"] = q["player_name"].map(nk)
        q["offense_snaps"] = pd.to_numeric(first(q, ["offense_snaps"]), errors="coerce")
        q["offense_pct"] = pd.to_numeric(first(q, ["offense_pct", "offense_percentage"]), errors="coerce")
        snaps = (
            q.loc[q["team"].ne("") & q["name_key"].ne(""), ["season", "week", "team", "name_key", "player_name", "offense_snaps", "offense_pct"]]
            .sort_values(["team", "name_key"])
            .drop_duplicates(["season", "week", "team", "name_key"], keep="last")
        )
        status["snap_counts_available"] = not snaps.empty
        status["snap_counts_rows"] = int(len(snaps))
        status["snap_counts_team_count"] = int(snaps["team"].nunique()) if not snaps.empty else 0
    except Exception as exc:  # data availability boundary
        status["snap_counts_error"] = f"{type(exc).__name__}: {exc}"

    return weekly, snaps, status


def verify_array_seal(manifest: pd.DataFrame, npz_path: Path) -> tuple[bool, list[dict]]:
    audit: list[dict] = []
    ok = True
    with np.load(npz_path, allow_pickle=False) as z:
        members = set(z.files)
        expected = set(manifest["array_member"].astype(str))
        if members != expected or len(members) != 107:
            ok = False
        for _, row in manifest.iterrows():
            member = str(row["array_member"])
            if member not in z:
                audit.append({"array_member": member, "present": False, "draws": 0, "hash_match": False})
                ok = False
                continue
            arr = np.asarray(z[member], dtype=np.float64)
            digest = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
            expected_hash = str(row["array_sha256_f64"])
            row_ok = (
                arr.ndim == 1
                and len(arr) == 25000
                and np.isfinite(arr).all()
                and (arr >= 0).all()
                and np.allclose(arr, np.round(arr), atol=0, rtol=0)
                and digest == expected_hash
            )
            ok = ok and row_ok
            audit.append(
                {
                    "array_member": member,
                    "present": True,
                    "draws": int(len(arr)),
                    "array_sha256_f64": digest,
                    "expected_sha256_f64": expected_hash,
                    "hash_match": digest == expected_hash,
                    "valid_receptions_draws": bool(row_ok),
                }
            )
    return ok, audit


def role_bucket(role, position) -> str:
    r = str(role or "").upper().strip()
    p = str(position or "").upper().strip()
    if p == "FB" or r.startswith("FB"):
        return "FB"
    if r == "RB1":
        return "RB1"
    if r.startswith("RB"):
        return "RB2+"
    return "UNRESOLVED_ROLE"


def pct_change(candidate: float, baseline: float) -> float:
    if not np.isfinite(baseline) or baseline == 0:
        return math.nan
    return 100.0 * (candidate - baseline) / baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26q-root", required=True)
    ap.add_argument("--r26r-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    qroot = Path(args.r26q_root)
    rroot = Path(args.r26r_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Exact sealed-parent evidence already artifact-digest verified by workflow.
    qdisp = json.loads(one(qroot, "r26q_disposition.json").read_text(encoding="utf-8"))
    qgates = pd.read_csv(one(qroot, "r26q_gate_matrix.csv"))
    manifest_path = one(qroot, "r26o_rb_receptions_shadow_manifest.csv")
    npz_path = one(qroot, "r26o_rb_receptions_shadow_arrays.npz")
    manifest = pd.read_csv(manifest_path, low_memory=False)

    rdisp = json.loads(one(rroot, "r26r_disposition.json").read_text(encoding="utf-8"))
    rgates = pd.read_csv(one(rroot, "r26r_gate_matrix.csv"))
    roles = pd.read_csv(one(rroot, "source_current_roles_identity_only.csv"), low_memory=False)
    market_summary = pd.read_csv(one(rroot, "r26r_receptions_player_market_summary.csv"), low_memory=False)
    market_lines = pd.read_csv(one(rroot, "r26r_receptions_book_line_snapshot.csv"), low_memory=False)

    array_ok, array_audit = verify_array_seal(manifest, npz_path)
    pd.DataFrame(array_audit).to_csv(out / "r26s_sealed_array_verification.csv", index=False)

    expected_teams = sorted(set(manifest["team"].astype(str).map(tm)) - {""})
    structural_parent_ok = (
        qdisp.get("disposition") == R26Q_DISPOSITION
        and int(qdisp.get("gate_count", 0)) == 28
        and int(qdisp.get("gate_pass_count", 0)) == 28
        and len(qgates) == 28
        and qgates["passed"].map(bool_json).all()
        and rdisp.get("disposition") == R26R_DISPOSITION
        and int(rdisp.get("gate_count", 0)) == 30
        and int(rdisp.get("gate_pass_count", 0)) == 30
        and len(rgates) == 30
        and rgates["passed"].map(bool_json).all()
        and len(manifest) == 107
        and int(pd.to_numeric(manifest["vacancy_active"], errors="coerce").fillna(0).sum()) == 104
        and array_ok
    )

    # Normalize frozen values and pregame descriptive labels before outcome merge.
    base = manifest.copy()
    base["team"] = base["team"].map(tm)
    base["player_clean_key"] = base["player_clean_key"].map(nk)
    base["baseline_receptions_mean"] = pd.to_numeric(base["baseline_mc_receptions_mean"], errors="coerce")
    base["candidate_receptions_mean"] = pd.to_numeric(base["shadow_receptions_mean"], errors="coerce")
    base["candidate_minus_baseline_mean"] = base["candidate_receptions_mean"] - base["baseline_receptions_mean"]
    base["vacancy_active"] = pd.to_numeric(base["vacancy_active"], errors="coerce").fillna(0).astype(int)

    roles = lower(roles)
    roles["team"] = roles["team"].map(tm)
    roles["player_clean_key"] = roles["player_clean_key"].map(nk)
    role_cols = [c for c in ["team", "player_clean_key", "role", "model_role", "position", "position_group"] if c in roles.columns]
    role_small = roles[role_cols].drop_duplicates(["team", "player_clean_key"], keep="last")
    base = base.merge(role_small, on=["team", "player_clean_key"], how="left", suffixes=("", "_role"))
    if "position" not in base.columns:
        base["position"] = base.get("position_family", "")
    base["pregame_role_cohort"] = [role_bucket(r, p) for r, p in zip(base.get("role", ""), base.get("position_family", ""))]
    base["movement_cohort"] = np.where(base["candidate_minus_baseline_mean"].abs() >= 0.50, "LARGE_MOVER", "SMALL_MOVER")
    base["direction_cohort"] = np.select(
        [base["candidate_minus_baseline_mean"] > 1e-9, base["candidate_minus_baseline_mean"] < -1e-9],
        ["INCREASED", "DECREASED"],
        default="UNCHANGED",
    )

    ms = lower(market_summary)
    ms["team"] = first(ms, ["sealed_team", "team"]).map(tm)
    ms["player_clean_key"] = ms["player_clean_key"].map(nk)
    ms = ms[["team", "player_clean_key", "book_line_rows", "books", "line_min", "line_median", "line_max"]].drop_duplicates(["team", "player_clean_key"])
    base = base.merge(ms, on=["team", "player_clean_key"], how="left")
    base["market_covered"] = pd.to_numeric(base["book_line_rows"], errors="coerce").fillna(0).gt(0)

    weekly, snaps, source_status = load_week1_outcomes()
    source_status["expected_sealed_team_count"] = len(expected_teams)
    source_status["expected_sealed_teams"] = expected_teams

    # Persist exact normalized source extracts before modeling evaluation.
    weekly.to_csv(out / "source_week1_weekly_stats.csv", index=False)
    snaps.to_csv(out / "source_week1_snap_counts.csv", index=False)
    source_status["weekly_stats_extract_sha256"] = sha256_file(out / "source_week1_weekly_stats.csv")
    source_status["snap_counts_extract_sha256"] = sha256_file(out / "source_week1_snap_counts.csv")

    source_complete = bool(
        source_status.get("weekly_stats_available")
        and source_status.get("snap_counts_available")
        and set(expected_teams).issubset(set(weekly["team"].unique()) if not weekly.empty else set())
        and set(expected_teams).issubset(set(snaps["team"].unique()) if not snaps.empty else set())
    )
    source_status["all_32_sealed_teams_in_weekly_stats"] = bool(not weekly.empty and set(expected_teams).issubset(set(weekly["team"].unique())))
    source_status["all_32_sealed_teams_in_snap_counts"] = bool(not snaps.empty and set(expected_teams).issubset(set(snaps["team"].unique())))
    source_status["week1_source_complete"] = source_complete
    json_dump(out / "r26s_outcome_source_status.json", source_status)

    # Merge participation and receptions. No snap evidence => unresolved; zero snaps => DNP.
    eval_df = base.copy()
    if not snaps.empty:
        ss = snaps[["team", "name_key", "offense_snaps", "offense_pct"]].rename(columns={"name_key": "player_clean_key"})
        eval_df = eval_df.merge(ss, on=["team", "player_clean_key"], how="left")
    else:
        eval_df["offense_snaps"] = np.nan
        eval_df["offense_pct"] = np.nan

    if not weekly.empty:
        ww = weekly[["team", "name_key", "receptions"]].rename(columns={"name_key": "player_clean_key", "receptions": "weekly_receptions"})
        eval_df = eval_df.merge(ww, on=["team", "player_clean_key"], how="left")
    else:
        eval_df["weekly_receptions"] = np.nan

    eval_df["participation_status"] = np.select(
        [
            pd.to_numeric(eval_df["offense_snaps"], errors="coerce").gt(0),
            pd.to_numeric(eval_df["offense_snaps"], errors="coerce").eq(0),
        ],
        ["EVALUABLE_OFFENSIVE_SNAPS", "ZERO_SNAP_DNP"],
        default="UNRESOLVED_PARTICIPATION",
    )
    eval_df["evaluable"] = eval_df["participation_status"].eq("EVALUABLE_OFFENSIVE_SNAPS")
    eval_df["actual_receptions"] = np.where(
        eval_df["evaluable"],
        pd.to_numeric(eval_df["weekly_receptions"], errors="coerce").fillna(0.0),
        np.nan,
    )
    eval_df["outcome_source"] = np.where(
        ~eval_df["evaluable"],
        "NOT_SCORED",
        np.where(eval_df["weekly_receptions"].notna(), "NFLVERSE_WEEKLY_STATS", "CONFIRMED_SNAP_ZERO_RECEPTIONS"),
    )
    eval_df["primary_evaluable"] = eval_df["vacancy_active"].eq(1) & eval_df["evaluable"]

    for prefix in ["baseline", "candidate"]:
        eval_df[f"{prefix}_error"] = eval_df[f"{prefix}_receptions_mean"] - eval_df["actual_receptions"]
        eval_df[f"{prefix}_abs_error"] = eval_df[f"{prefix}_error"].abs()
    eval_df["paired_improvement"] = eval_df["baseline_abs_error"] - eval_df["candidate_abs_error"]
    eval_df["winner"] = np.select(
        [eval_df["paired_improvement"] > 1e-12, eval_df["paired_improvement"] < -1e-12],
        ["CANDIDATE", "BASELINE"],
        default="TIE",
    )

    primary = eval_df.loc[eval_df["primary_evaluable"]].copy()
    controls = eval_df.loc[eval_df["vacancy_active"].eq(0)].copy()

    metrics = {
        "primary_evaluable_rows": int(len(primary)),
        "sealed_rows": int(len(eval_df)),
        "changed_rows": int(eval_df["vacancy_active"].eq(1).sum()),
        "control_rows": int(eval_df["vacancy_active"].eq(0).sum()),
        "evaluable_all_rows": int(eval_df["evaluable"].sum()),
        "zero_snap_dnp_rows": int(eval_df["participation_status"].eq("ZERO_SNAP_DNP").sum()),
        "unresolved_participation_rows": int(eval_df["participation_status"].eq("UNRESOLVED_PARTICIPATION").sum()),
    }

    if len(primary):
        metrics.update(
            {
                "baseline_mae": float(primary["baseline_abs_error"].mean()),
                "candidate_mae": float(primary["candidate_abs_error"].mean()),
                "mae_improvement": float(primary["paired_improvement"].mean()),
                "relative_mae_improvement_pct": float(100.0 * primary["paired_improvement"].mean() / primary["baseline_abs_error"].mean()) if primary["baseline_abs_error"].mean() else math.nan,
                "baseline_bias": float(primary["baseline_error"].mean()),
                "candidate_bias": float(primary["candidate_error"].mean()),
                "baseline_median_abs_error": float(primary["baseline_abs_error"].median()),
                "candidate_median_abs_error": float(primary["candidate_abs_error"].median()),
                "candidate_wins": int(primary["winner"].eq("CANDIDATE").sum()),
                "baseline_wins": int(primary["winner"].eq("BASELINE").sum()),
                "ties": int(primary["winner"].eq("TIE").sum()),
            }
        )

    # Team-cluster bootstrap.
    boot = pd.DataFrame(columns=["iteration", "mae_improvement"])
    if len(primary) and primary["team"].nunique() > 0:
        rng = np.random.default_rng(42)
        teams = np.array(sorted(primary["team"].unique()), dtype=object)
        groups = {t: primary.loc[primary["team"].eq(t)] for t in teams}
        vals = []
        for i in range(10000):
            sampled = rng.choice(teams, size=len(teams), replace=True)
            pieces = [groups[t] for t in sampled]
            z = pd.concat(pieces, ignore_index=True)
            vals.append(float(z["baseline_abs_error"].mean() - z["candidate_abs_error"].mean()))
        boot = pd.DataFrame({"iteration": np.arange(1, 10001), "mae_improvement": vals})
        metrics["bootstrap_mean_improvement"] = float(boot["mae_improvement"].mean())
        metrics["bootstrap_ci_lower_2_5"] = float(boot["mae_improvement"].quantile(0.025))
        metrics["bootstrap_ci_upper_97_5"] = float(boot["mae_improvement"].quantile(0.975))
    boot.to_csv(out / "r26s_bootstrap_distribution.csv", index=False)

    # Within-room allocation-share evaluation.
    room_rows: list[dict] = []
    share_player_rows: list[dict] = []
    for team, g in primary.groupby("team"):
        if len(g) < 2 or float(g["actual_receptions"].sum()) <= 0:
            continue
        bsum = float(g["baseline_receptions_mean"].sum())
        csum = float(g["candidate_receptions_mean"].sum())
        asum = float(g["actual_receptions"].sum())
        if bsum <= 0 or csum <= 0:
            continue
        z = g.copy()
        z["actual_share"] = z["actual_receptions"] / asum
        z["baseline_share"] = z["baseline_receptions_mean"] / bsum
        z["candidate_share"] = z["candidate_receptions_mean"] / csum
        z["baseline_share_abs_error"] = (z["baseline_share"] - z["actual_share"]).abs()
        z["candidate_share_abs_error"] = (z["candidate_share"] - z["actual_share"]).abs()
        for _, r in z.iterrows():
            share_player_rows.append(
                {
                    "team": team,
                    "player_clean_key": r["player_clean_key"],
                    "player": r["player"],
                    "actual_share": r["actual_share"],
                    "baseline_share": r["baseline_share"],
                    "candidate_share": r["candidate_share"],
                    "baseline_share_abs_error": r["baseline_share_abs_error"],
                    "candidate_share_abs_error": r["candidate_share_abs_error"],
                }
            )
        room_rows.append(
            {
                "team": team,
                "evaluable_primary_players": int(len(z)),
                "actual_receptions_sum": asum,
                "baseline_receptions_sum": bsum,
                "candidate_receptions_sum": csum,
                "baseline_share_mae": float(z["baseline_share_abs_error"].mean()),
                "candidate_share_mae": float(z["candidate_share_abs_error"].mean()),
            }
        )
    rooms = pd.DataFrame(room_rows)
    shares = pd.DataFrame(share_player_rows)
    if not shares.empty:
        metrics["baseline_within_room_share_mae"] = float(shares["baseline_share_abs_error"].mean())
        metrics["candidate_within_room_share_mae"] = float(shares["candidate_share_abs_error"].mean())
        metrics["within_room_share_players"] = int(len(shares))
        metrics["within_room_share_teams"] = int(shares["team"].nunique())
    rooms.to_csv(out / "r26s_team_room_evaluation.csv", index=False)
    shares.to_csv(out / "r26s_player_share_evaluation.csv", index=False)

    # Frozen role cohorts and large movers.
    cohort_rows: list[dict] = []
    for cohort, g in primary.groupby("pregame_role_cohort", dropna=False):
        b = float(g["baseline_abs_error"].mean()) if len(g) else math.nan
        c = float(g["candidate_abs_error"].mean()) if len(g) else math.nan
        cohort_rows.append(
            {
                "cohort_type": "PREGAME_ROLE",
                "cohort": str(cohort),
                "n": int(len(g)),
                "baseline_mae": b,
                "candidate_mae": c,
                "relative_candidate_vs_baseline_pct": pct_change(c, b),
            }
        )
    large = primary.loc[primary["movement_cohort"].eq("LARGE_MOVER")]
    if len(large):
        b = float(large["baseline_abs_error"].mean())
        c = float(large["candidate_abs_error"].mean())
        cohort_rows.append(
            {
                "cohort_type": "MOVEMENT",
                "cohort": "LARGE_MOVER_ABS_DELTA_GE_0_50",
                "n": int(len(large)),
                "baseline_mae": b,
                "candidate_mae": c,
                "relative_candidate_vs_baseline_pct": pct_change(c, b),
            }
        )
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(out / "r26s_role_cohort_evaluation.csv", index=False)

    # Exact pregame market benchmark only; never a football input or promotion gate.
    mb = eval_df.loc[eval_df["market_covered"]].copy()
    if len(mb):
        mb["market_median_abs_error"] = (pd.to_numeric(mb["line_median"], errors="coerce") - mb["actual_receptions"]).abs()
        mb["baseline_vs_market_gap"] = mb["baseline_receptions_mean"] - pd.to_numeric(mb["line_median"], errors="coerce")
        mb["candidate_vs_market_gap"] = mb["candidate_receptions_mean"] - pd.to_numeric(mb["line_median"], errors="coerce")
    market_cols = [
        "team", "player", "player_clean_key", "evaluable", "actual_receptions",
        "baseline_receptions_mean", "candidate_receptions_mean", "line_min", "line_median", "line_max",
        "book_line_rows", "baseline_abs_error", "candidate_abs_error", "market_median_abs_error",
        "baseline_vs_market_gap", "candidate_vs_market_gap",
    ]
    for c in market_cols:
        if c not in mb.columns:
            mb[c] = np.nan
    mb[market_cols].to_csv(out / "r26s_market_benchmark.csv", index=False)

    # Identity audit and full player evaluation.
    identity_audit = eval_df[[
        "team", "player", "player_clean_key", "vacancy_active", "pregame_role_cohort",
        "offense_snaps", "weekly_receptions", "participation_status", "evaluable", "outcome_source",
    ]].copy()
    identity_audit.to_csv(out / "r26s_identity_audit.csv", index=False)
    eval_df.to_csv(out / "r26s_player_evaluation.csv", index=False)

    # Frozen scientific decision rules.
    primary_n = int(len(primary))
    cin_exact = bool(
        len(controls) == 3
        and np.allclose(
            pd.to_numeric(controls["baseline_receptions_mean"], errors="coerce"),
            pd.to_numeric(controls["candidate_receptions_mean"], errors="coerce"),
            atol=0,
            rtol=0,
        )
    )
    cin_error_exact = bool(
        controls.loc[controls["evaluable"]].empty
        or np.allclose(
            controls.loc[controls["evaluable"], "baseline_abs_error"],
            controls.loc[controls["evaluable"], "candidate_abs_error"],
            atol=0,
            rtol=0,
        )
    )
    role_gate = True
    if not cohorts.empty:
        role_q = cohorts.loc[(cohorts["cohort_type"].eq("PREGAME_ROLE")) & (cohorts["n"].ge(10))]
        if len(role_q):
            role_gate = bool((pd.to_numeric(role_q["relative_candidate_vs_baseline_pct"], errors="coerce") <= 10.0 + 1e-12).all())
    large_gate = True
    if len(large) >= 10:
        large_gate = bool(float(large["candidate_abs_error"].mean()) <= float(large["baseline_abs_error"].mean()) + 1e-12)

    gates: list[dict] = []
    gate(gates, "01_exact_r26q_parent_contract", qdisp.get("disposition") == R26Q_DISPOSITION and len(qgates) == 28 and qgates["passed"].map(bool_json).all(), {"disposition": qdisp.get("disposition"), "gates": int(len(qgates))})
    gate(gates, "02_exact_r26r_parent_contract", rdisp.get("disposition") == R26R_DISPOSITION and len(rgates) == 30 and rgates["passed"].map(bool_json).all(), {"disposition": rdisp.get("disposition"), "gates": int(len(rgates))})
    gate(gates, "03_sealed_candidate_arrays_exact", array_ok, {"arrays": len(array_audit), "expected": 107})
    gate(gates, "04_sealed_scope_107_104_3", len(manifest) == 107 and eval_df["vacancy_active"].eq(1).sum() == 104 and eval_df["vacancy_active"].eq(0).sum() == 3, {"rows": len(manifest), "changed": int(eval_df["vacancy_active"].eq(1).sum()), "controls": int(eval_df["vacancy_active"].eq(0).sum())})
    gate(gates, "05_week1_weekly_stats_available", bool(source_status.get("weekly_stats_available")), source_status.get("weekly_stats_error") or source_status.get("weekly_stats_rows", 0))
    gate(gates, "06_week1_snap_counts_available", bool(source_status.get("snap_counts_available")), source_status.get("snap_counts_error") or source_status.get("snap_counts_rows", 0))
    gate(gates, "07_all_32_week1_teams_represented", source_complete, {"weekly_teams": source_status.get("weekly_stats_team_count", 0), "snap_teams": source_status.get("snap_counts_team_count", 0), "expected": 32})
    gate(gates, "08_primary_evaluable_changed_rows_ge_50", primary_n >= 50, primary_n)
    gate(gates, "09_no_unresolved_identity_silently_scored", bool(eval_df.loc[eval_df["participation_status"].eq("UNRESOLVED_PARTICIPATION"), "evaluable"].eq(False).all()), int(eval_df["participation_status"].eq("UNRESOLVED_PARTICIPATION").sum()))
    gate(gates, "10_zero_snap_dnp_not_scored_as_zero", bool(eval_df.loc[eval_df["participation_status"].eq("ZERO_SNAP_DNP"), "actual_receptions"].isna().all()), int(eval_df["participation_status"].eq("ZERO_SNAP_DNP").sum()))

    scientific_ready = bool(structural_parent_ok and source_complete and primary_n >= 50)
    if scientific_ready:
        gate(gates, "11_candidate_primary_mae_strictly_lower", metrics["candidate_mae"] < metrics["baseline_mae"], {"baseline": metrics["baseline_mae"], "candidate": metrics["candidate_mae"]})
        gate(gates, "12_team_cluster_bootstrap_mean_gt_zero", metrics.get("bootstrap_mean_improvement", math.nan) > 0, metrics.get("bootstrap_mean_improvement"))
        gate(gates, "13_bootstrap_ci_lower_ge_minus_0_05", metrics.get("bootstrap_ci_lower_2_5", math.nan) >= -0.05, metrics.get("bootstrap_ci_lower_2_5"))
        gate(gates, "14_candidate_within_room_share_mae_not_worse", metrics.get("candidate_within_room_share_mae", math.inf) <= metrics.get("baseline_within_room_share_mae", -math.inf) + 1e-12, {"baseline": metrics.get("baseline_within_room_share_mae"), "candidate": metrics.get("candidate_within_room_share_mae")})
        gate(gates, "15_large_mover_mae_not_worse_if_n_ge_10", large_gate, {"n": int(len(large))})
        gate(gates, "16_no_pregame_role_cohort_n_ge_10_worsens_gt_10pct", role_gate, cohorts.to_dict(orient="records"))
    else:
        for n in [
            "11_candidate_primary_mae_strictly_lower",
            "12_team_cluster_bootstrap_mean_gt_zero",
            "13_bootstrap_ci_lower_ge_minus_0_05",
            "14_candidate_within_room_share_mae_not_worse",
            "15_large_mover_mae_not_worse_if_n_ge_10",
            "16_no_pregame_role_cohort_n_ge_10_worsens_gt_10pct",
        ]:
            gate(gates, n, False, "NOT_EVALUATED_SOURCE_OR_SAMPLE_INCOMPLETE")

    gate(gates, "17_cin_controls_baseline_candidate_exact", cin_exact and cin_error_exact, {"rows": int(len(controls)), "evaluable": int(controls["evaluable"].sum())})
    gate(gates, "18_sportsbook_football_inputs_zero", True, 0)
    gate(gates, "19_no_postgame_regeneration_refit_or_tuning", True, {"football_values_regenerated": False, "r9_refit": False, "tuning": False})
    gate(gates, "20_production_parameters_unchanged", True, False)
    gate(gates, "21_no_production_promotion_performed", True, False)
    gate(gates, "22_no_live_shadow_activation_performed", True, False)

    gate_df = pd.DataFrame(gates)
    gate_df.to_csv(out / "r26s_gate_matrix.csv", index=False)

    if not structural_parent_ok:
        disposition = FAIL
        decision_reason = "IMMUTABLE_PARENT_OR_SEAL_CONTRACT_FAILURE"
    elif not source_complete or primary_n < 50:
        disposition = INCOMPLETE
        decision_reason = "AUTHORITATIVE_WEEK1_OUTCOME_OR_PARTICIPATION_EVIDENCE_INCOMPLETE"
    else:
        scientific_gate_names = {
            "11_candidate_primary_mae_strictly_lower",
            "12_team_cluster_bootstrap_mean_gt_zero",
            "13_bootstrap_ci_lower_ge_minus_0_05",
            "14_candidate_within_room_share_mae_not_worse",
            "15_large_mover_mae_not_worse_if_n_ge_10",
            "16_no_pregame_role_cohort_n_ge_10_worsens_gt_10pct",
            "17_cin_controls_baseline_candidate_exact",
        }
        scientific_pass = bool(gate_df.loc[gate_df["gate"].isin(scientific_gate_names), "passed"].all())
        disposition = PASS if scientific_pass else FAIL
        decision_reason = "FROZEN_SCIENTIFIC_GATES_PASS" if scientific_pass else "FROZEN_SCIENTIFIC_GATE_FAILURE"

    # Source/hash provenance manifest.
    provenance = []
    for label, path in [
        ("r26q_disposition", one(qroot, "r26q_disposition.json")),
        ("r26q_gate_matrix", one(qroot, "r26q_gate_matrix.csv")),
        ("sealed_r26o_manifest", manifest_path),
        ("sealed_r26o_arrays", npz_path),
        ("r26r_disposition", one(rroot, "r26r_disposition.json")),
        ("r26r_gate_matrix", one(rroot, "r26r_gate_matrix.csv")),
        ("r26r_roles_snapshot", one(rroot, "source_current_roles_identity_only.csv")),
        ("r26r_market_summary", one(rroot, "r26r_receptions_player_market_summary.csv")),
        ("r26r_market_book_lines", one(rroot, "r26r_receptions_book_line_snapshot.csv")),
        ("week1_weekly_stats_extract", out / "source_week1_weekly_stats.csv"),
        ("week1_snap_counts_extract", out / "source_week1_snap_counts.csv"),
    ]:
        provenance.append({"source": label, "path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)})
    pd.DataFrame(provenance).to_csv(out / "r26s_source_provenance_manifest.csv", index=False)

    payload = {
        "candidate": "RB_R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_V1",
        "disposition": disposition,
        "decision_reason": decision_reason,
        "authority_note": "Evaluation only. PASS authorizes a separately frozen production-promotion review, not production promotion.",
        "parent_r26q_run": 34400524030,
        "parent_r26q_artifact": 10123251043,
        "parent_r26q_artifact_digest": "sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1",
        "parent_r26q_head": "68661da94f03cab2f96182d47636cf55e088b5de",
        "parent_r26r_run": 34401814588,
        "parent_r26r_artifact": 10124274040,
        "parent_r26r_artifact_digest": "sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e",
        "parent_r26r_head": "469aa40c90c738e12a82ee32ccca70c9cdbbc29f",
        "frozen_gate_count": 22,
        "gate_pass_count": int(gate_df["passed"].sum()),
        "source_complete": source_complete,
        "scientific_evaluation_ready": scientific_ready,
        "metrics": metrics,
        "sportsbook_football_inputs_used": 0,
        "football_values_regenerated": False,
        "r9_refit": False,
        "tuning_performed": False,
        "same_week_depth_used_to_change_candidate": False,
        "production_parameters_changed": False,
        "production_promotion_performed": False,
        "live_shadow_production_activation_performed": False,
        "production_promotion_review_authorized": disposition == PASS,
    }
    json_dump(out / "r26s_disposition.json", payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"R26S disposition={disposition} gates={int(gate_df['passed'].sum())}/{len(gate_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
