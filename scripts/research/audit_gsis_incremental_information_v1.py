#!/usr/bin/env python3
"""Sanitized GSIS incremental-information audit.

This is a source-information audit, not a predictive test.  It reads only the
identity and play-choice columns from GSIS Lineup Detail and Formation Usage.
It deliberately does not read gains, first downs, touchdowns, turnovers, EPA,
or any other football outcome/efficiency field.

The output contains aggregate counts and distribution summaries only.  It does
not contain player names, team-level report values, exact lineups, URLs, or
authentication/session material.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


TEAM_MAP = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "LA": "LAR",
}
NAME_SUFFIXES = {"JR", "JR.", "SR", "SR.", "II", "III", "IV", "V"}
DOWN_MAP = {
    "FIRST DOWN": 1,
    "SECOND DOWN": 2,
    "THIRD DOWN": 3,
    "FOURTH DOWN": 4,
}


def _team(value: Any) -> str:
    raw = str(value).strip().upper()
    return TEAM_MAP.get(raw, raw)


def _number(value: Any) -> float:
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "—", "N/A"}:
        return float("nan")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else float("nan")


def _player_key(value: Any) -> str:
    text = re.sub(
        r"\b(?:Jr|Sr|II|III|IV|V)\.?$", "", str(value), flags=re.IGNORECASE
    ).strip()
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _split_lineup(value: Any) -> list[str]:
    players: list[str] = []
    for part in [piece.strip() for piece in str(value).split(",") if piece.strip()]:
        if part.upper() in NAME_SUFFIXES and players:
            players[-1] = f"{players[-1]}, {part}"
        else:
            players.append(part)
    return players


def _filter_value(record: dict[str, Any], filter_id: str) -> str:
    for item in record.get("filters", []):
        if item.get("id") == filter_id:
            return str(item.get("value", ""))
    raise RuntimeError(f"missing GSIS filter {filter_id}")


def _header_and_rows(table: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rows = table.get("rows", [])
    for index, row in enumerate(rows):
        cells = row.get("cells", [])
        if sum(cell.get("tag") == "th" for cell in cells) > 1:
            return [str(cell.get("text", "")).strip() for cell in cells], rows[index + 1 :]
    raise RuntimeError("GSIS table has no multi-column header row")


def _median(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _load_snapshot(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("season", 0)) != 2026 or str(payload.get("phase", "")).upper() != "REG":
        raise RuntimeError("audit requires the frozen 2026 REG point-in-time snapshot")
    return payload


def _lineup_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    allowed = {"Lineup", "Plays", "Passing Plays", "Rushing Plays"}
    for record in payload.get("records", []):
        if record.get("report") != "Lineup Detail":
            continue
        team = _team(_filter_value(record, "select2"))
        mode = str(record.get("mode", ""))
        for table in record.get("tables", []):
            headers, rows = _header_and_rows(table)
            missing = allowed.difference(headers)
            if missing:
                raise RuntimeError(f"Lineup Detail missing allowed fields: {sorted(missing)}")
            index = {name: headers.index(name) for name in allowed}
            for row in rows:
                cells = [str(cell.get("text", "")).strip() for cell in row.get("cells", [])]
                if len(cells) != len(headers):
                    continue
                players = _split_lineup(cells[index["Lineup"]])
                item = {
                    "team": team,
                    "mode": mode,
                    "players": frozenset(_player_key(player) for player in players),
                    "player_count": len(players),
                    "plays": _number(cells[index["Plays"]]),
                    "passing_plays": _number(cells[index["Passing Plays"]]),
                    "rushing_plays": _number(cells[index["Rushing Plays"]]),
                }
                (valid if len(players) == 11 else invalid).append(item)
    return valid, invalid


def _lineup_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["mode"] == mode:
            grouped[row["team"]].append(row)

    team_metrics: list[dict[str, float]] = []
    duplicate_sets = 0
    for team_rows in grouped.values():
        ordered = sorted(team_rows, key=lambda item: item["plays"], reverse=True)
        duplicate_sets += len(ordered) - len({item["players"] for item in ordered})
        total_plays = sum(item["plays"] for item in ordered)
        if not ordered or not total_plays:
            continue
        shares = [item["plays"] / total_plays for item in ordered]
        top_players = ordered[0]["players"]
        weighted_substitutions = sum(
            item["plays"] * (11 - len(item["players"].intersection(top_players)))
            for item in ordered
        ) / total_plays

        eligible = [
            item
            for item in ordered
            if item["passing_plays"] + item["rushing_plays"] >= 5
        ]
        eligible_denominator = sum(
            item["passing_plays"] + item["rushing_plays"] for item in eligible
        )
        all_denominator = sum(
            item["passing_plays"] + item["rushing_plays"] for item in ordered
        )
        if eligible_denominator:
            pass_rate = sum(item["passing_plays"] for item in eligible) / eligible_denominator
            pass_rate_mad = sum(
                (item["passing_plays"] + item["rushing_plays"])
                * abs(
                    item["passing_plays"]
                    / (item["passing_plays"] + item["rushing_plays"])
                    - pass_rate
                )
                for item in eligible
            ) / eligible_denominator
        else:
            pass_rate_mad = float("nan")

        team_metrics.append(
            {
                "lineups": float(len(ordered)),
                "top1_share": shares[0],
                "top3_share": sum(shares[:3]),
                "effective_lineups": 1.0 / sum(share * share for share in shares),
                "weighted_substitutions": weighted_substitutions,
                "eligible_rows": float(len(eligible)),
                "eligible_play_share": eligible_denominator / all_denominator
                if all_denominator
                else float("nan"),
                "pass_rate_mad": pass_rate_mad,
            }
        )

    return {
        "teams": len(grouped),
        "valid_rows": sum(len(items) for items in grouped.values()),
        "duplicate_exact_player_sets": duplicate_sets,
        "median_lineups_per_team": _median(item["lineups"] for item in team_metrics),
        "median_top1_play_share": _median(item["top1_share"] for item in team_metrics),
        "median_top3_play_share": _median(item["top3_share"] for item in team_metrics),
        "median_effective_lineups": _median(
            item["effective_lineups"] for item in team_metrics
        ),
        "median_play_weighted_substitutions_from_top_lineup": _median(
            item["weighted_substitutions"] for item in team_metrics
        ),
        "lineups_with_at_least_5_pass_rush_plays": int(
            sum(item["eligible_rows"] for item in team_metrics)
        ),
        "median_eligible_pass_rush_play_share": _median(
            item["eligible_play_share"] for item in team_metrics
        ),
        "median_within_team_lineup_pass_rate_mad": _median(
            item["pass_rate_mad"] for item in team_metrics
        ),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"required live-stack input missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _identity_coverage(
    offense_rows: list[dict[str, Any]], live_stack: Path, snap_trace_path: Path
) -> dict[str, Any]:
    roles = _read_csv(live_stack / "current_player_availability.csv")
    roles["team"] = roles["team"].map(_team)
    roles["audit_key"] = roles["player"].map(_player_key)
    role_index = {
        (row.team, row.audit_key): row for row in roles.itertuples(index=False)
    }

    lineup_players = {
        (row["team"], player)
        for row in offense_rows
        for player in row["players"]
        if player
    }
    matched = {identity for identity in lineup_players if identity in role_index}
    positions = Counter(
        str(role_index[identity].position_group).upper() for identity in matched
    )
    matched_skill = {
        identity
        for identity in matched
        if str(role_index[identity].position_group).upper()
        in {"QB", "RB", "FB", "WR", "TE"}
    }
    matched_wr_te = {
        identity
        for identity in matched
        if str(role_index[identity].position_group).upper() in {"WR", "TE"}
    }

    snap = _read_csv(snap_trace_path)
    snap["team"] = snap["team"].map(_team)
    snap["audit_key"] = snap["player_clean_key"].map(_player_key)
    snap_keys = set(zip(snap["team"], snap["audit_key"]))

    player_form = _read_csv(live_stack / "player_form_consensus.csv")
    player_form["team"] = player_form["team"].map(_team)
    player_form["audit_key"] = player_form["player_clean_key"].map(_player_key)
    player_form_keys = set(zip(player_form["team"], player_form["audit_key"]))

    entitlement = _read_csv(live_stack / "target_entitlement_v1_trace.csv")
    entitlement["team"] = entitlement["team"].map(_team)
    entitlement["audit_key"] = entitlement["player_clean_key"].map(_player_key)
    entitlement_keys = set(zip(entitlement["team"], entitlement["audit_key"]))

    return {
        "unique_gsis_offensive_player_team_identities": len(lineup_players),
        "matched_current_depth_availability_identities": len(matched),
        "matched_current_depth_availability_rate": _ratio(len(matched), len(lineup_players)),
        "matched_position_groups": dict(sorted(positions.items())),
        "matched_skill_identities": len(matched_skill),
        "matched_skill_in_player_form": sum(
            identity in player_form_keys for identity in matched_skill
        ),
        "matched_skill_in_entitlement_trace": sum(
            identity in entitlement_keys for identity in matched_skill
        ),
        "matched_wr_te_identities": len(matched_wr_te),
        "matched_wr_te_in_strict_prior_snap_trace": sum(
            identity in snap_keys for identity in matched_wr_te
        ),
        "joint_lineup_fields_in_live_stack": False,
    }


def _formation_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    selected_teams: set[str] = set()
    nonempty_teams: set[str] = set()
    required = {
        "Yards to Go",
        "# TEs",
        "# WRs",
        "Play Count",
        "Rushing Plays",
        "Passing Plays",
    }
    for record in payload.get("records", []):
        if record.get("report") != "Formation Usage":
            continue
        team = _team(_filter_value(record, "select2"))
        selected_teams.add(team)
        for table in record.get("tables", []):
            table_rows = table.get("rows", [])
            title = ""
            if table_rows:
                title = next(
                    (
                        str(cell.get("text", "")).strip()
                        for cell in table_rows[0].get("cells", [])
                        if cell.get("tag") == "th"
                    ),
                    "",
                )
            down = DOWN_MAP.get(title)
            if down is None:
                raise RuntimeError(f"unknown Formation Usage table title: {title}")
            headers, data_rows = _header_and_rows(table)
            missing = required.difference(headers)
            if missing:
                raise RuntimeError(f"Formation Usage missing allowed fields: {sorted(missing)}")
            index = {name: headers.index(name) for name in required}
            for row in data_rows:
                cells = [str(cell.get("text", "")).strip() for cell in row.get("cells", [])]
                if len(cells) != len(headers):
                    continue
                play_count = _number(cells[index["Play Count"]])
                if not math.isfinite(play_count) or play_count <= 0:
                    continue
                tight_ends = int(_number(cells[index["# TEs"]]))
                wide_receivers = int(_number(cells[index["# WRs"]]))
                running_backs = 5 - tight_ends - wide_receivers
                personnel = (
                    f"{running_backs}{tight_ends}"
                    if 0 <= running_backs <= 4 and 0 <= tight_ends <= 4
                    else "OTHER"
                )
                nonempty_teams.add(team)
                rows.append(
                    {
                        "team": team,
                        "down": down,
                        "yards_to_go": cells[index["Yards to Go"]],
                        "personnel": personnel,
                        "plays": play_count,
                        "passing_plays": _number(cells[index["Passing Plays"]]),
                        "rushing_plays": _number(cells[index["Rushing Plays"]]),
                    }
                )
    return rows, sorted(selected_teams.difference(nonempty_teams))


def _formation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["team"]].append(row)

    team_metrics: list[dict[str, float]] = []
    personnel_team_coverage: Counter[str] = Counter()
    for team_rows in grouped.values():
        personnel = {row["personnel"] for row in team_rows}
        personnel_team_coverage.update(personnel)
        play_total = sum(row["plays"] for row in team_rows)
        by_personnel: Counter[str] = Counter()
        for row in team_rows:
            by_personnel[row["personnel"]] += row["plays"]
        shares = [value / play_total for value in by_personnel.values()]

        eligible = [
            row
            for row in team_rows
            if row["passing_plays"] + row["rushing_plays"] >= 5
        ]
        eligible_denominator = sum(
            row["passing_plays"] + row["rushing_plays"] for row in eligible
        )
        all_denominator = sum(
            row["passing_plays"] + row["rushing_plays"] for row in team_rows
        )
        if eligible_denominator:
            pass_rate = sum(row["passing_plays"] for row in eligible) / eligible_denominator
            pass_rate_mad = sum(
                (row["passing_plays"] + row["rushing_plays"])
                * abs(
                    row["passing_plays"]
                    / (row["passing_plays"] + row["rushing_plays"])
                    - pass_rate
                )
                for row in eligible
            ) / eligible_denominator
        else:
            pass_rate_mad = float("nan")
        team_metrics.append(
            {
                "cells": float(len(team_rows)),
                "personnel": float(len(personnel)),
                "top_share": max(shares),
                "effective_personnel": 1.0 / sum(share * share for share in shares),
                "eligible_rows": float(len(eligible)),
                "eligible_play_share": eligible_denominator / all_denominator
                if all_denominator
                else float("nan"),
                "pass_rate_mad": pass_rate_mad,
            }
        )

    # Hold team, down and yards-to-go fixed, then measure pass-rate dispersion
    # across personnel cells with at least three pass/rush plays.
    situations: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["passing_plays"] + row["rushing_plays"] >= 3:
            situations[(row["team"], row["down"], row["yards_to_go"])].append(row)
    team_mad_numerator: Counter[str] = Counter()
    team_mad_denominator: Counter[str] = Counter()
    qualified_situations = 0
    qualified_cells = 0
    for (team, _down, _yards), situation_rows in situations.items():
        if len({row["personnel"] for row in situation_rows}) < 2:
            continue
        denominator = sum(
            row["passing_plays"] + row["rushing_plays"] for row in situation_rows
        )
        pass_rate = sum(row["passing_plays"] for row in situation_rows) / denominator
        mad = sum(
            (row["passing_plays"] + row["rushing_plays"])
            * abs(
                row["passing_plays"]
                / (row["passing_plays"] + row["rushing_plays"])
                - pass_rate
            )
            for row in situation_rows
        ) / denominator
        team_mad_numerator[team] += mad * denominator
        team_mad_denominator[team] += denominator
        qualified_situations += 1
        qualified_cells += len(situation_rows)
    within_situation_team_mad = [
        team_mad_numerator[team] / denominator
        for team, denominator in team_mad_denominator.items()
    ]

    return {
        "teams_with_nonempty_rows": len(grouped),
        "nonempty_rows": len(rows),
        "personnel_codes_observed": sorted(personnel_team_coverage),
        "personnel_code_team_coverage": dict(sorted(personnel_team_coverage.items())),
        "median_situation_personnel_cells_per_team": _median(
            item["cells"] for item in team_metrics
        ),
        "median_personnel_groups_per_team": _median(
            item["personnel"] for item in team_metrics
        ),
        "median_top_personnel_play_share": _median(
            item["top_share"] for item in team_metrics
        ),
        "median_effective_personnel_groups": _median(
            item["effective_personnel"] for item in team_metrics
        ),
        "cells_with_at_least_5_pass_rush_plays": int(
            sum(item["eligible_rows"] for item in team_metrics)
        ),
        "median_eligible_pass_rush_play_share": _median(
            item["eligible_play_share"] for item in team_metrics
        ),
        "median_within_team_cell_pass_rate_mad": _median(
            item["pass_rate_mad"] for item in team_metrics
        ),
        "same_team_down_distance_multi_personnel_situations": qualified_situations,
        "same_situation_qualified_cells": qualified_cells,
        "same_situation_qualified_pass_rush_plays": int(sum(team_mad_denominator.values())),
        "median_within_situation_personnel_pass_rate_mad": _median(
            within_situation_team_mad
        ),
    }


def _team_form_summary(live_stack: Path) -> dict[str, Any]:
    team_form = _read_csv(live_stack / "team_form.csv")
    twelve = pd.to_numeric(team_form.get("12p_rate"), errors="coerce")
    proe = pd.to_numeric(team_form.get("proe"), errors="coerce")
    pass_rate = pd.to_numeric(team_form.get("pass_rate_off"), errors="coerce")
    return {
        "rows": len(team_form),
        "twelve_personnel_rate_nonnull": int(twelve.notna().sum()),
        "twelve_personnel_rate_unique_values": int(twelve.nunique(dropna=True)),
        "twelve_personnel_rate_all_zero": bool(twelve.notna().all() and twelve.eq(0).all()),
        "proe_nonnull": int(proe.notna().sum()),
        "proe_unique_values": int(proe.nunique(dropna=True)),
        "offensive_pass_rate_nonnull": int(pass_rate.notna().sum()),
        "offensive_pass_rate_unique_values": int(pass_rate.nunique(dropna=True)),
    }


def build_audit(
    snapshot_path: Path, live_stack: Path, snap_trace_path: Path
) -> dict[str, Any]:
    payload = _load_snapshot(snapshot_path)
    lineups, invalid_lineups = _lineup_rows(payload)
    offense = [row for row in lineups if row["mode"] == "Offense"]
    formation, empty_formation_teams = _formation_rows(payload)

    return {
        "audit_version": "GSIS_INCREMENTAL_INFORMATION_AUDIT_V1",
        "audit_type": "source_quality_and_incremental_information_only",
        "model_fit_performed": False,
        "production_changed": False,
        "full_slate_changed": False,
        "week3_performance_outcome_fields_read": False,
        "gsis_reports_read": ["Lineup Detail", "Formation Usage"],
        "gsis_fields_read": {
            "Lineup Detail": [
                "Lineup",
                "Plays",
                "Passing Plays",
                "Rushing Plays",
            ],
            "Formation Usage": [
                "Down",
                "Yards to Go",
                "# TEs",
                "# WRs",
                "Play Count",
                "Passing Plays",
                "Rushing Plays",
            ],
        },
        "lineup_detail": {
            "offense": _lineup_summary(lineups, "Offense"),
            "defense": _lineup_summary(lineups, "Defense"),
            "invalid_or_ambiguous_player_count_rows": len(invalid_lineups),
            "identity_coverage_against_live_stack": _identity_coverage(
                offense, live_stack, snap_trace_path
            ),
        },
        "formation_usage": {
            **_formation_summary(formation),
            "source_empty_teams": empty_formation_teams,
            "team_form_comparator": _team_form_summary(live_stack),
        },
        "temporal_limitations": {
            "gsis_report_has_week_filter": False,
            "single_cumulative_snapshot_can_measure_temporal_churn": False,
            "single_cumulative_snapshot_can_time_replacement_emergence": False,
            "prospective_immutable_snapshots_required": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsis-snapshot", type=Path, required=True)
    parser.add_argument("--live-stack-dir", type=Path, required=True)
    parser.add_argument("--snap-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.gsis_snapshot, args.live_stack_dir, args.snap_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "disposition": "GSIS_INCREMENTAL_INFORMATION_AUDIT_COMPLETE",
                "output": str(args.output),
                "raw_gsis_values_emitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
