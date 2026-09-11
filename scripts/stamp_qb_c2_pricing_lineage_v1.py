#!/usr/bin/env python3
"""Stamp exact QB C2 distribution lineage onto priced Full Slate rows.

This is a projection-neutral governance adapter. It runs *after* pricing and
copies only football-model provenance from the already-certified C2 production
audit into ``outputs/props_priced_clean.csv``. No projection, distribution,
book, line, odds, fair probability, or edge value is modified.

The stamp makes player-by-player routing explicit:
- M89/M90 continues to own the final pass-yard mean.
- the frozen Phase-J selector decides whether the C2 distribution specialist was
  consumed for that QB.
- unselected QBs are explicitly marked as canonical-distribution fallback.
- non-pass-yards markets carry zero/blank QB-distribution fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.eligible_team_set_v1 import validate_current_team_set
from scripts.utils.player_identity_v3 import player_name_key

DATA = Path("data")
OUTPUTS = Path("outputs")
PRICED = OUTPUTS / "props_priced_clean.csv"
C2_CSV = DATA / "qb_c2_production_integration_audit.csv"
C2_JSON = DATA / "qb_c2_production_integration_audit.json"
OUT_AUDIT = DATA / "qb_c2_pricing_lineage_stamp_audit.json"

SPECIALIST_VERSION = "C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1"
CANONICAL_FALLBACK = "CANONICAL_QB_DISTRIBUTION"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"QB C2 pricing lineage required artifact missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    if out.empty:
        raise RuntimeError(f"QB C2 pricing lineage required artifact has zero rows: {path}")
    return out


def _key(value: object) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _validate_current_c2_scope(
    status: dict,
    c2: pd.DataFrame,
    *,
    active_roles_path: Path | None = None,
) -> dict:
    """Validate one current C2 starter row for every certified eligible team.

    In legacy mode the shared helper retains the historical 32-team contract.
    With explicit current availability, current-output coverage is the exact
    certified eligible-team set. This does not alter the separate 32-team C2
    state-context source-integrity contract upstream.
    """
    if "team" not in c2.columns:
        raise RuntimeError("QB C2 production CSV missing team column")
    scope = validate_current_team_set(
        c2["team"],
        active_roles_path=active_roles_path,
        label="QB C2 current production audit",
    )
    expected_qbs = int(scope["observed_teams"])
    team_keys = c2["team"].map(canon_team)
    if team_keys.eq("").any():
        raise RuntimeError("QB C2 production CSV contains unresolvable team identity")
    if len(c2) != expected_qbs or team_keys.nunique() != expected_qbs:
        raise RuntimeError(
            "QB C2 production CSV must contain exactly one QB per certified eligible team; "
            f"rows={len(c2)} teams={team_keys.nunique()} expected={expected_qbs}"
        )
    if int(status.get("football_qb_rows", 0)) != expected_qbs:
        raise RuntimeError(
            "QB C2 production audit row count != certified eligible-team scope; "
            f"status_rows={status.get('football_qb_rows')} expected={expected_qbs}"
        )
    return scope


def _validate_priced_c2_subset(
    lookup: dict[tuple[str, str], pd.Series],
    matched_identities: set[tuple[str, str]],
    selected_players: int,
) -> dict:
    """Validate sportsbook-priced QBs as a downstream subset of football QBs.

    Sportsbook offer availability may omit an otherwise valid football starter.
    The sportsbook is therefore never allowed to define the upstream football QB
    universe. Every priced QB must exist in the certified C2 football audit, but
    certified football QBs without a pass-yard offer are explicitly allowed and
    recorded as downstream coverage gaps.
    """
    audit_identities = set(lookup)
    if not matched_identities:
        raise RuntimeError("QB C2 pricing lineage matched zero priced pass-yard QBs")
    extra = sorted(matched_identities - audit_identities)
    if extra:
        raise RuntimeError(f"priced QB identities absent from C2 football audit: {extra[:20]}")

    selected_audit_identities = {
        identity
        for identity, row in lookup.items()
        if int(row.get("selector_c2_selected")) == 1
    }
    expected_selected_priced = int(len(selected_audit_identities & matched_identities))
    if int(selected_players) != expected_selected_priced:
        raise RuntimeError(
            "QB C2 stamped selected-player count drift within priced subset "
            f"expected={expected_selected_priced} actual={selected_players}"
        )

    missing_priced = sorted(audit_identities - matched_identities)
    return {
        "football_qbs": int(len(audit_identities)),
        "priced_pass_yard_qbs": int(len(matched_identities)),
        "football_qbs_without_priced_pass_yard_offer": int(len(missing_priced)),
        "football_qbs_without_priced_pass_yard_offer_identities": [
            {"team": team, "player_key": player_key} for team, player_key in missing_priced
        ],
        "c2_selected_football_qbs": int(len(selected_audit_identities)),
        "c2_selected_priced_qbs": int(expected_selected_priced),
        "c2_selected_unpriced_qbs": int(len(selected_audit_identities - matched_identities)),
    }


def main() -> int:
    priced = _read(PRICED)
    c2 = _read(C2_CSV)
    if not C2_JSON.exists() or C2_JSON.stat().st_size <= 0:
        raise RuntimeError(f"QB C2 production audit missing: {C2_JSON}")
    status = json.loads(C2_JSON.read_text(encoding="utf-8"))
    if status.get("disposition") != "QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS":
        raise RuntimeError(f"QB C2 production integration not certified: {status.get('disposition')}")
    if float(status.get("max_raw_qb_mean_gap", 1.0)) > 1e-10:
        raise RuntimeError("QB C2 production audit has raw mean drift")
    if int(status.get("state_capture_changed_arrays", -1)) != 0:
        raise RuntimeError("QB C2 state-capture seam is not exact")
    for field in (
        "sportsbook_inputs_to_starter_selection",
        "sportsbook_inputs_to_selector",
        "sportsbook_inputs_to_c2_generation",
    ):
        if int(status.get(field, 1)) != 0:
            raise RuntimeError(f"QB C2 production audit leakage flag {field}={status.get(field)}")

    required_c2 = {
        "team", "player", "player_clean_key", "selector_version",
        "selector_delta_pass_attempts", "selector_c2_selected",
        "starter_authority_source", "raw_mean_gap",
    }
    missing = sorted(required_c2 - set(c2.columns))
    if missing:
        raise RuntimeError(f"QB C2 production CSV missing columns: {missing}")
    scope = _validate_current_c2_scope(status, c2)
    expected_qbs = int(scope["observed_teams"])

    # Build an identity key independently from display-name punctuation/suffixes.
    c2["_team_key"] = c2["team"].map(canon_team)
    c2["_player_key"] = c2["player"].map(_key)
    if c2["_player_key"].eq("").any() or c2.duplicated(["_team_key", "_player_key"]).any():
        raise RuntimeError("QB C2 production CSV has blank/duplicate canonical identities")

    # Preserve every pre-stamp value so this adapter can prove it is projection-neutral.
    protected_cols = [
        c for c in (
            "event_id", "player", "player_clean_key", "team", "opponent",
            "market", "source_market", "vegas_line", "model_proj", "mc_proj",
            "model_sd", "fair_prob", "market_prob", "vegas_odds", "fair_odds",
            "edge_pct", "edge_abs", "book", "book_title", "side",
            "qb_synthesis_proj", "qb_synthesis_version", "qb_synthesis_applied",
        ) if c in priced.columns
    ]
    before = priced[protected_cols].copy(deep=True)

    stamp_cols = [
        "qb_distribution_specialist_applied",
        "qb_distribution_specialist_version",
        "qb_distribution_candidate_version",
        "qb_distribution_selector_version",
        "qb_distribution_selector_delta_pass_attempts",
        "qb_distribution_starter_authority_source",
        "qb_distribution_route",
        "qb_distribution_raw_mean_gap",
    ]
    # Fail closed rather than silently overwriting a potentially different stamp.
    already = [c for c in stamp_cols if c in priced.columns]
    if already:
        raise RuntimeError(f"priced output already contains QB C2 lineage columns; refusing overwrite: {already}")

    priced["qb_distribution_specialist_applied"] = 0
    priced["qb_distribution_specialist_version"] = ""
    priced["qb_distribution_candidate_version"] = ""
    priced["qb_distribution_selector_version"] = ""
    priced["qb_distribution_selector_delta_pass_attempts"] = np.nan
    priced["qb_distribution_starter_authority_source"] = ""
    priced["qb_distribution_route"] = ""
    priced["qb_distribution_raw_mean_gap"] = np.nan

    source_market = priced.get("source_market", pd.Series("", index=priced.index)).astype(str)
    pass_mask = source_market.eq("player_pass_yds")
    if int(pass_mask.sum()) <= 0:
        raise RuntimeError("QB C2 pricing lineage found zero pass-yard side rows")

    # Use Series rows rather than namedtuples because leading-underscore helper
    # column names are not stable namedtuple attributes in pandas.
    lookup = {
        (str(r["_team_key"]), str(r["_player_key"])): r
        for _, r in c2.iterrows()
    }
    missing_rows: list[dict] = []
    matched_identities: set[tuple[str, str]] = set()
    for idx, row in priced.loc[pass_mask].iterrows():
        identity = (canon_team(row.get("team")), _key(row.get("player")))
        cr = lookup.get(identity)
        if cr is None:
            missing_rows.append({"team": row.get("team"), "player": row.get("player")})
            continue
        matched_identities.add(identity)
        selected = int(cr.get("selector_c2_selected"))
        priced.at[idx, "qb_distribution_specialist_applied"] = selected
        priced.at[idx, "qb_distribution_specialist_version"] = SPECIALIST_VERSION if selected else ""
        priced.at[idx, "qb_distribution_candidate_version"] = SPECIALIST_VERSION
        priced.at[idx, "qb_distribution_selector_version"] = str(cr.get("selector_version"))
        priced.at[idx, "qb_distribution_selector_delta_pass_attempts"] = float(cr.get("selector_delta_pass_attempts"))
        priced.at[idx, "qb_distribution_starter_authority_source"] = str(cr.get("starter_authority_source"))
        priced.at[idx, "qb_distribution_route"] = "C2_SELECTED" if selected else CANONICAL_FALLBACK
        priced.at[idx, "qb_distribution_raw_mean_gap"] = float(cr.get("raw_mean_gap"))

    if missing_rows:
        raise RuntimeError(f"priced pass-yard rows missing QB C2 audit identity: {missing_rows[:20]}")

    pass_rows = priced.loc[pass_mask].copy()
    selected_players = int(
        pass_rows.loc[pd.to_numeric(pass_rows["qb_distribution_specialist_applied"], errors="coerce").eq(1), ["team", "player"]]
        .drop_duplicates().shape[0]
    )
    priced_scope = _validate_priced_c2_subset(lookup, matched_identities, selected_players)
    if int(priced_scope["football_qbs"]) != expected_qbs:
        raise RuntimeError(
            "QB C2 football audit identity count drifted from certified eligible-team scope; "
            f"audit={priced_scope['football_qbs']} expected={expected_qbs}"
        )
    if int(priced_scope["c2_selected_football_qbs"]) != int(status.get("selected_qb_rows", -1)):
        raise RuntimeError(
            "QB C2 selected football-QB count differs between CSV and JSON audit; "
            f"csv={priced_scope['c2_selected_football_qbs']} json={status.get('selected_qb_rows')}"
        )
    if not pass_rows["qb_distribution_candidate_version"].eq(SPECIALIST_VERSION).all():
        raise RuntimeError("not every pass-yard row records the frozen C2 candidate version")
    if pass_rows["qb_distribution_selector_version"].astype(str).str.strip().eq("").any():
        raise RuntimeError("pass-yard row missing QB C2 selector version")
    gap = pd.to_numeric(pass_rows["qb_distribution_raw_mean_gap"], errors="coerce")
    if gap.isna().any() or float(gap.abs().max()) > 1e-10:
        raise RuntimeError("stamped QB C2 raw mean-neutrality drift")

    non_pass = priced.loc[~pass_mask]
    if not pd.to_numeric(non_pass["qb_distribution_specialist_applied"], errors="coerce").fillna(0).eq(0).all():
        raise RuntimeError("non-pass-yard rows were incorrectly stamped with QB C2")
    for col in (
        "qb_distribution_specialist_version", "qb_distribution_candidate_version",
        "qb_distribution_selector_version", "qb_distribution_starter_authority_source",
        "qb_distribution_route",
    ):
        if non_pass[col].fillna("").astype(str).str.strip().ne("").any():
            raise RuntimeError(f"non-pass-yard rows contain QB C2 lineage field {col}")

    # Prove no existing pricing/model values changed.
    for col in protected_cols:
        left = before[col]
        right = priced[col]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            a = pd.to_numeric(left, errors="coerce").to_numpy(float)
            b = pd.to_numeric(right, errors="coerce").to_numpy(float)
            if not np.allclose(a, b, rtol=0, atol=0, equal_nan=True):
                raise RuntimeError(f"QB C2 lineage stamp changed protected numeric column={col}")
        else:
            if not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
                raise RuntimeError(f"QB C2 lineage stamp changed protected text column={col}")

    priced.to_csv(PRICED, index=False)
    payload = {
        "disposition": "QB_C2_PRICING_LINEAGE_STAMP_CERTIFIED",
        "priced_side_rows": int(len(priced)),
        "pass_yard_side_rows": int(pass_mask.sum()),
        "pass_yard_qbs": int(len(matched_identities)),
        "football_qbs": int(priced_scope["football_qbs"]),
        "football_qbs_without_priced_pass_yard_offer": int(priced_scope["football_qbs_without_priced_pass_yard_offer"]),
        "football_qbs_without_priced_pass_yard_offer_identities": priced_scope["football_qbs_without_priced_pass_yard_offer_identities"],
        "current_team_scope_mode": str(scope["mode"]),
        "current_team_scope_expected": int(scope["expected_teams"]),
        "c2_selected_qbs": int(priced_scope["c2_selected_priced_qbs"]),
        "c2_selected_football_qbs": int(priced_scope["c2_selected_football_qbs"]),
        "c2_selected_unpriced_qbs": int(priced_scope["c2_selected_unpriced_qbs"]),
        "c2_unselected_priced_qbs": int(len(matched_identities) - selected_players),
        "specialist_version": SPECIALIST_VERSION,
        "selector_version": str(pass_rows["qb_distribution_selector_version"].iloc[0]),
        "max_raw_mean_gap": float(gap.abs().max()),
        "non_pass_specialist_rows": 0,
        "protected_columns_changed": 0,
        "sportsbook_inputs_used_to_stamp": False,
        "sportsbook_offer_coverage_defines_football_universe": False,
        "pricing_values_modified": False,
        "source_c2_audit": str(C2_JSON),
        "output": str(PRICED),
    }
    OUT_AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[qb_c2_pricing_lineage_stamp] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
