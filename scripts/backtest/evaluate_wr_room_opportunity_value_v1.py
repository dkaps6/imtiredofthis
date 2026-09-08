#!/usr/bin/env python3
"""WR-R13: conserved WR-room opportunity + production-value entitlement.

Materially different follow-up to failed WR-R12. R12 showed that recent same-team
target counts improved target MAE but harmed receiving-yard MAE and tails. R13 asks
whether strictly-prior receiving production adds information about *which* current WR
should receive a fixed WR-room opportunity share.

Frozen before results:
- same 2025 W1-18 walk-forward evaluation and M38 baseline as WR-R12;
- same four most recent completed same-team WR games;
- no sportsbook inputs;
- WR-room mass conserved exactly;
- evidence score uses equal-weight target evidence and receiving-yard evidence
  converted into target-equivalents at the prior WR room's own yards/target;
- exactly one average prior WR-room game of M38 pseudo-target mass;
- same frozen gates as WR-R12; no tuning after seeing results.

Passing only authorizes a separate integration confirmation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.backtest import evaluate_wr_room_empirical_bayes_v1 as r12


def apply_candidate(base: pd.DataFrame, logs: pd.DataFrame, season: int, week: int):
    pieces = []
    audits = []
    for (event_id, team), group in base.groupby(["event_id", "team"], dropna=False, sort=False):
        g = group.copy()
        pos = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().str.strip()
        wr_mask = pos.isin(r12.WR_POS)
        raw = pd.to_numeric(g.get("rules_tgt_share"), errors="coerce").fillna(0.0).clip(0.0, 0.95)
        wr_total = float(raw.loc[wr_mask].sum())
        if wr_mask.sum() <= 1 or wr_total <= 0:
            pieces.append(g)
            continue

        wr_idx = list(g.index[wr_mask])
        baseline_room = raw.loc[wr_idx].to_numpy(float) / wr_total
        hist = r12.prior_team_wr_games(logs, season, week, str(team))
        n_games = int(hist[["season", "week"]].drop_duplicates().shape[0]) if not hist.empty else 0
        if hist.empty:
            pieces.append(g)
            continue

        hist = hist.copy()
        hist["_targets"] = pd.to_numeric(hist.get("targets", 0.0), errors="coerce").fillna(0.0)
        hist["_rec_yards"] = pd.to_numeric(hist.get("receiving_yards", hist.get("rec_yards", 0.0)), errors="coerce").fillna(0.0)
        total_targets = float(hist["_targets"].sum())
        total_yards = float(hist["_rec_yards"].sum())
        avg_room_targets = total_targets / n_games if n_games > 0 and total_targets > 0 else 20.0
        room_ypt = total_yards / total_targets if total_targets > 0 and total_yards > 0 else np.nan

        by_player = hist.groupby("player_clean_key", dropna=False)[["_targets", "_rec_yards"]].sum()
        current_keys = g.loc[wr_idx, "player_clean_key"].astype(str).tolist()
        obs_t = np.asarray([float(by_player.loc[k, "_targets"]) if k in by_player.index else 0.0 for k in current_keys], dtype=float)
        obs_y = np.asarray([float(by_player.loc[k, "_rec_yards"]) if k in by_player.index else 0.0 for k in current_keys], dtype=float)

        if np.isfinite(room_ypt) and room_ypt > 0:
            yard_target_equiv = obs_y / room_ypt
            evidence = 0.5 * (obs_t + yard_target_equiv)
        else:
            evidence = obs_t.copy()

        score = evidence + avg_room_targets * baseline_room
        candidate_room = score / float(score.sum()) if float(score.sum()) > 0 else baseline_room.copy()
        candidate = candidate_room * wr_total
        g.loc[wr_idx, "rules_tgt_share"] = candidate
        audits.append({
            "week": int(week), "event_id": str(event_id), "team": str(team),
            "history_games": n_games, "history_wr_targets": total_targets,
            "history_wr_rec_yards": total_yards, "history_wr_yards_per_target": room_ypt,
            "pseudo_wr_targets": float(avg_room_targets),
            "baseline_wr_room_mass": wr_total, "candidate_wr_room_mass": float(candidate.sum()),
            "mass_gap": float(candidate.sum() - wr_total),
            "max_player_room_share_move": float(np.max(np.abs(candidate_room - baseline_room))),
        })
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True), audits


def relabel_outputs(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for p in list(out_dir.glob("wr_r12_*.csv")):
        text = p.read_text(encoding="utf-8")
        text = text.replace("WR_R12_EB4", "WR_R13_OPP_VALUE")
        text = text.replace("WR_R12_CONSERVED_ENTITLEMENT", "WR_R13_OPPORTUNITY_VALUE")
        new = p.with_name(p.name.replace("wr_r12_", "wr_r13_"))
        new.write_text(text, encoding="utf-8")
        p.unlink()


def main() -> int:
    r12.apply_candidate = apply_candidate
    out_dir = Path("data/backtests/wr_room_opportunity_value_v1")
    args = list(sys.argv[1:])
    if "--out-dir" not in args:
        args.extend(["--out-dir", str(out_dir)])
    else:
        out_dir = Path(args[args.index("--out-dir") + 1])
    sys.argv = [sys.argv[0], *args]
    rc = int(r12.main())
    relabel_outputs(out_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
