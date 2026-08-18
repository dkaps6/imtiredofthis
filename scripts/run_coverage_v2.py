#!/usr/bin/env python3
"""Production runner for Coverage v2 with explicit Ourlads position normalization.

Ourlads preserves alignment-specific position labels (LWR/SWR/RWR). Coverage v2
needs the broader semantic group WR when constructing the active receiver universe.
This runner adapts that provider-specific schema at the boundary without changing
or degrading the canonical Ourlads artifact used elsewhere in the model.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from scripts.build import build_coverage_v2 as cov
from scripts.runtime_context import resolve_week

WR_SLOT_POSITIONS = {"WR", "LWR", "RWR", "SWR", "WIDE RECEIVER", "SLOT WR"}


def normalize_ourlads_roles_for_coverage(roles: pd.DataFrame) -> pd.DataFrame:
    if roles is None or roles.empty:
        raise RuntimeError("Coverage v2 requires non-empty Ourlads roles")
    out = roles.copy()
    cols = {str(c).strip().lower(): c for c in out.columns}
    pos_col = cols.get("position") or cols.get("pos")
    if not pos_col:
        raise RuntimeError("Ourlads roles missing position column")

    raw = out[pos_col].fillna("").astype(str).str.upper().str.strip()
    # Preserve the source position in a diagnostic column while exposing the
    # semantic position group expected by Coverage v2.
    if "coverage_source_position" not in out.columns:
        out["coverage_source_position"] = raw
    out[pos_col] = raw.where(~raw.isin(WR_SLOT_POSITIONS), "WR")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    season = int(args.season)
    week = int(args.week) if args.week is not None else int(resolve_week())

    roles = cov._safe_csv(cov.ROLES)
    adapted_roles = normalize_ourlads_roles_for_coverage(roles)
    wr_count = int(adapted_roles[adapted_roles["position"].astype(str).str.upper().eq("WR")].shape[0])
    if wr_count == 0:
        source_counts = roles.get("position", pd.Series(dtype="object")).astype(str).value_counts().to_dict()
        raise RuntimeError(f"Coverage v2 Ourlads adapter found zero WRs; source position counts={source_counts}")

    team_map = cov.build_authoritative_team_map(season, week)
    wrs = cov.load_wr_universe(team_map, roles=adapted_roles)
    team_cov = cov.build_team_coverage(season, week, team_map)
    player_cov = cov.build_player_coverage(wrs, team_cov)
    exposure = cov.build_exposure(player_cov, team_cov)

    if exposure.empty or exposure["opponent"].fillna("").eq("").any():
        raise RuntimeError("Coverage v2 lost authoritative WR/opponent identity")

    cov.DATA.mkdir(parents=True, exist_ok=True)
    team_cov.to_csv(cov.TEAM_OUT, index=False)
    player_cov.to_csv(cov.PLAYER_OUT, index=False)
    exposure.to_csv(cov.EXPOSURE_OUT, index=False)

    print(
        f"[coverage_v2] season={season} week={week} scheduled_teams={len(team_map)} "
        f"source_WR_rows={wr_count} active_WRs={len(exposure)}"
    )
    print(
        f"[coverage_v2] team_scheme_available={int(team_cov['coverage_available'].sum())}/{len(team_cov)} "
        f"source={team_cov['coverage_source'].iloc[0] if len(team_cov) else 'unavailable'}"
    )
    print(
        f"[coverage_v2] direct_WR_CB_matchups={int(player_cov['matchup_available'].sum())}/{len(player_cov)} "
        f"alignments={int(player_cov['alignment_available'].sum())}/{len(player_cov)}"
    )
    print("[coverage_v2] Ourlads LWR/SWR/RWR normalized only at coverage boundary; source roles remain unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
