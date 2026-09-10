#!/usr/bin/env python3
"""Build the three predeclared availability integration fixtures from immutable evidence.

The baseline current_player_availability.csv is never overwritten. Each fixture
copies it, injects only a definitive-unavailable status for the selected current
player(s), re-ranks through the already-locked resolver, then materializes active
roles and applies the already-locked game timing eligibility filter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.build.build_current_player_availability_v1 import rerank
from scripts.build.build_reconciled_active_roles_v1 import build as build_active
from scripts.build.build_production_eligible_active_roles_v1 import build as build_eligible

DATA = Path("data")


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"fixture input missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _pick_rb1(active: pd.DataFrame):
    x = active.loc[active.position_group.astype(str).str.upper().eq("RB")].copy()
    for team, g in x.groupby("team", sort=True):
        g = g.sort_values(["depth_index", "player_clean_key"])
        rb1 = g.loc[g.role.astype(str).eq("RB1")]
        if len(rb1) == 1 and len(g) >= 2:
            return rb1.iloc[0], g.iloc[1]
    raise RuntimeError("no eligible RB1/RB2 fixture room found")


def _pick_qb1(active: pd.DataFrame):
    x = active.loc[active.position_group.astype(str).str.upper().eq("QB")].copy()
    for team, g in x.groupby("team", sort=True):
        g = g.sort_values(["depth_index", "player_clean_key"])
        qb1 = g.loc[g.role.astype(str).eq("QB1")]
        if len(qb1) == 1 and len(g) >= 2:
            return qb1.iloc[0], g.iloc[1]
    raise RuntimeError("no eligible QB1/QB2 fixture room found")


def _pick_wr_te(active: pd.DataFrame):
    for team in sorted(active.team.dropna().astype(str).unique()):
        wr = active.loc[(active.team.astype(str).eq(team)) & active.position_group.astype(str).str.upper().eq("WR")].copy()
        te = active.loc[(active.team.astype(str).eq(team)) & active.position_group.astype(str).str.upper().eq("TE")].copy()
        wr2p = wr.loc[~wr.role.astype(str).eq("WR1")].sort_values(["depth_index", "player_clean_key"])
        wr1 = wr.loc[wr.role.astype(str).eq("WR1")]
        te1 = te.loc[te.role.astype(str).eq("TE1")]
        if len(wr1) == 1 and len(wr2p) >= 2 and len(te1) == 1 and len(te) >= 2:
            return wr2p.iloc[0], te1.iloc[0]
    raise RuntimeError("no eligible WR2+/TE1 fixture room found")


def _mark_unavailable(avail: pd.DataFrame, targets: list[tuple[str, str]]) -> pd.DataFrame:
    x = avail.copy()
    target_set = {(str(t), str(k)) for t, k in targets}
    mask = pd.Series([(str(t), str(k)) in target_set for t, k in zip(x.team, x.player_clean_key)], index=x.index)
    if int(mask.sum()) != len(target_set):
        raise RuntimeError(f"fixture target resolution mismatch requested={target_set} matched={int(mask.sum())}")
    x.loc[mask, "final_availability_state"] = "UNAVAILABLE_FIXTURE_INACTIVE"
    x.loc[mask, "availability_authority"] = "frozen_certification_fixture"
    x.loc[mask, "availability_reason"] = "predeclared 35-gate definitive-unavailable fixture"
    x.loc[mask, "definitive_unavailable"] = 1
    x.loc[mask, "eligible_for_opportunity"] = 0
    x = rerank(x)
    return x


def _write(name: str, avail: pd.DataFrame, cert: pd.DataFrame, targets: list[dict], out_root: Path) -> dict:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    active, active_meta = build_active(avail)
    eligible, eligible_meta = build_eligible(active, cert)
    avail.to_csv(out_dir / "current_player_availability.csv", index=False)
    active.to_csv(out_dir / "roles_ourlads_active_v1.csv", index=False)
    eligible.to_csv(out_dir / "roles_current_production_eligible_v1.csv", index=False)
    payload = {
        "fixture": name,
        "targets": targets,
        "active_meta": active_meta,
        "eligible_meta": eligible_meta,
        "sportsbook_inputs_used": 0,
    }
    (out_dir / "fixture_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DATA / "availability_cert_fixtures")
    args = ap.parse_args()
    avail = _load(DATA / "current_player_availability.csv")
    active = _load(DATA / "roles_current_production_eligible_v1.csv")
    cert = _load(DATA / "current_player_availability_game_certification.csv")

    old_rb1, prior_rb2 = _pick_rb1(active)
    rb_avail = _mark_unavailable(avail, [(old_rb1.team, old_rb1.player_clean_key)])
    rb_active, _ = build_active(rb_avail)
    rb_eligible, _ = build_eligible(rb_active, cert)
    rb_room = rb_eligible.loc[(rb_eligible.team.astype(str).eq(str(old_rb1.team))) & rb_eligible.position_group.astype(str).str.upper().eq("RB")].sort_values(["depth_index", "player_clean_key"])
    successor = rb_room.loc[rb_room.role.astype(str).eq("RB1")]
    if len(successor) != 1 or str(successor.iloc[0].player_clean_key) != str(prior_rb2.player_clean_key):
        raise RuntimeError("RB1 fixture did not deterministically promote prior RB2")
    rb_payload = _write("rb1_out", rb_avail, cert, [{
        "team": str(old_rb1.team), "player": str(old_rb1.player), "player_clean_key": str(old_rb1.player_clean_key),
        "old_role": "RB1", "expected_successor": str(prior_rb2.player_clean_key),
    }], args.out_root)

    old_qb1, prior_qb2 = _pick_qb1(active)
    qb_avail = _mark_unavailable(avail, [(old_qb1.team, old_qb1.player_clean_key)])
    qb_active, _ = build_active(qb_avail)
    qb_eligible, _ = build_eligible(qb_active, cert)
    qb_room = qb_eligible.loc[(qb_eligible.team.astype(str).eq(str(old_qb1.team))) & qb_eligible.position_group.astype(str).str.upper().eq("QB")]
    successor = qb_room.loc[qb_room.role.astype(str).eq("QB1")]
    if len(successor) != 1 or str(successor.iloc[0].player_clean_key) != str(prior_qb2.player_clean_key):
        raise RuntimeError("QB1 fixture did not deterministically promote prior QB2")
    qb_payload = _write("qb1_inactive", qb_avail, cert, [{
        "team": str(old_qb1.team), "player": str(old_qb1.player), "player_clean_key": str(old_qb1.player_clean_key),
        "old_role": "QB1", "expected_successor": str(prior_qb2.player_clean_key),
    }], args.out_root)

    old_wr, old_te = _pick_wr_te(active)
    wt_avail = _mark_unavailable(avail, [(old_wr.team, old_wr.player_clean_key), (old_te.team, old_te.player_clean_key)])
    wt_payload = _write("wr_te_unavailable", wt_avail, cert, [
        {"team": str(old_wr.team), "player": str(old_wr.player), "player_clean_key": str(old_wr.player_clean_key), "old_role": str(old_wr.role), "family": "WR"},
        {"team": str(old_te.team), "player": str(old_te.player), "player_clean_key": str(old_te.player_clean_key), "old_role": "TE1", "family": "TE"},
    ], args.out_root)

    payload = {"disposition": "CURRENT_PLAYER_AVAILABILITY_CERTIFICATION_FIXTURES_BUILT", "rb1_out": rb_payload, "qb1_inactive": qb_payload, "wr_te_unavailable": wt_payload, "sportsbook_inputs_used": 0}
    (args.out_root / "fixtures_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
