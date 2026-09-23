#!/usr/bin/env python3
"""Full slate backtest report over one or more graded weeks.

Reads the graded detail CSV that ``grade_market_track_record_gsis_v1.py``
emits with ``--detail-out`` and prints the decomposition we actually need to
judge the board: overall, by market, by position, by side, plus the
falsification checks that a headline hit rate hides (odds uniformity, slate
coverage, side imbalance).

Read-only. Never touches pricing, projection or model selection.
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NFL_TEAMS = 32
QB_MARKET = "pass_yards"
# One bet per player-market-week; the board can carry several book rows.
DEDUP_KEY = ["season", "week", "player_clean_key", "market"]


def _decided(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["bet_result"].isin(["WIN", "LOSS"])]


def _row_stats(df: pd.DataFrame) -> dict:
    dec = _decided(df)
    w = int(dec["bet_result"].eq("WIN").sum())
    n = int(len(dec))
    units = float(pd.to_numeric(dec["unit_result"], errors="coerce").fillna(0).sum())
    return {
        "bets": n,
        "W": w,
        "L": n - w,
        "hit": (w / n) if n else np.nan,
        "units": units,
        "roi": (units / n) if n else np.nan,
        "model_mae": float(dec["model_error"].abs().mean()),
        "vegas_mae": float(dec["vegas_error"].abs().mean()),
        "closer": float(dec["model_closer_than_vegas"].mean()),
    }


def _fmt_table(rows: dict[str, dict]) -> str:
    hdr = f"{'':<18}{'bets':>6}{'W-L':>9}{'hit':>8}{'units':>9}{'roi':>8}{'mMAE':>8}{'vMAE':>8}{'closer':>8}"
    out = [hdr, "-" * len(hdr)]
    for name, s in rows.items():
        record = "{}-{}".format(s["W"], s["L"])
        out.append(
            f"{name:<18}{s['bets']:>6}{record:>9}"
            f"{s['hit']:>8.3f}{s['units']:>+9.2f}{s['roi']:>+8.3f}"
            f"{s['model_mae']:>8.2f}{s['vegas_mae']:>8.2f}{s['closer']:>8.3f}"
        )
    return "\n".join(out)


def cluster_score_pvalue(df: pd.DataFrame, probabilities) -> tuple[float, int]:
    """One-sided score test over independent NFL game clusters."""
    required = ["season", "week", "event_id"]
    if any(c not in df.columns for c in required):
        return float("nan"), 0
    frame = df[required].copy()
    for c in required:
        if frame[c].isna().any() or frame[c].astype(str).str.strip().eq("").any():
            return float("nan"), 0

    p = np.asarray(list(probabilities), dtype=float)
    if len(p) != len(df) or np.any(~np.isfinite(p)):
        return float("nan"), 0

    frame["_score"] = df["bet_result"].eq("WIN").astype(float).to_numpy() - p
    scores = frame.groupby(required, dropna=False)["_score"].sum().to_numpy(dtype=float)
    n_clusters = int(len(scores))
    if n_clusters < 8:
        return float("nan"), n_clusters

    sd = float(np.std(scores, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan"), n_clusters
    t_stat = float(np.mean(scores)) / (sd / np.sqrt(n_clusters))
    return float(stats.t.sf(t_stat, df=n_clusters - 1)), n_clusters


def breakeven(odds: float) -> float:
    o = float(odds)
    return (-o) / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report(detail_path: Path) -> int:
    d = pd.read_csv(detail_path, low_memory=False)
    d.columns = [c.lower() for c in d.columns]
    for col in ("model_error", "vegas_error", "model_closer_than_vegas"):
        if col not in d.columns:
            raise SystemExit(f"graded detail is missing required column {col!r}")

    if d.empty:
        section("SLATE BACKTEST — ZERO SELECTED PRODUCTION BETS")
        print("selected settlement rows: 0; decided=0; void=0")
        print("No publishable positive-EV bets were selected; report exits cleanly.")
        return 0

    weeks = sorted(pd.to_numeric(d["week"], errors="coerce").dropna().astype(int).unique())
    section(f"SLATE BACKTEST — season {d['season'].iloc[0]}, weeks {weeks}")
    print(
        f"selected settlement rows: {len(d)}; "
        f"decided={int(d['bet_result'].isin(['WIN','LOSS']).sum())}; "
        f"void={int(d['bet_result'].eq('VOID').sum())}"
    )
    print(f"graded detail columns: {sorted(d.columns)}")
    if "position" not in d.columns:
        print("\n  ** position is absent from the graded detail, so the per-position")
        print("  ** tables below are skipped. That is a grader defect, not an")
        print("  ** empty result -- do not read its absence as 'no position data'.")

    section("OVERALL")
    print(_fmt_table({"all markets": _row_stats(d)}))

    section("BY WEEK")
    print(_fmt_table({f"week {w}": _row_stats(g) for w, g in d.groupby("week")}))

    section("BY MARKET")
    by_mkt = {str(m): _row_stats(g) for m, g in d.groupby("market")}
    print(_fmt_table(dict(sorted(by_mkt.items(), key=lambda kv: -kv[1]["hit"]))))

    if "position" in d.columns:
        pos_order = ["QB", "RB", "WR", "TE"]
        def _ordered(frame):
            seen = [p for p in pos_order if (frame["position"] == p).any()]
            rest = sorted(set(frame["position"].astype(str)) - set(pos_order))
            return seen + rest

        section("BY POSITION")
        print(_fmt_table({p: _row_stats(d[d["position"] == p]) for p in _ordered(d)}))

        for wk, gw in d.groupby("week"):
            section(f"BY POSITION — WEEK {int(wk)}")
            print(_fmt_table({p: _row_stats(gw[gw["position"] == p]) for p in _ordered(gw)}))

        section("BY POSITION x MARKET")
        rows = {}
        for p in _ordered(d):
            gp = d[d["position"] == p]
            for m, gm in gp.groupby("market"):
                if len(_decided(gm)) >= 5:
                    rows[f"{p} {m}"] = _row_stats(gm)
        print(_fmt_table(rows))
        print("\n(cells with fewer than 5 decided bets are omitted)")

    section("BY SIDE")
    print(_fmt_table({str(s): _row_stats(g) for s, g in d.groupby(d["side"].astype(str).str.upper())}))

    # ---- falsification checks -------------------------------------------
    section("FALSIFICATION CHECK 1 — odds uniformity")
    odds = d["vegas_odds"].value_counts(dropna=False)
    print(f"distinct vegas_odds values: {len(odds)}")
    print(odds.head(12).to_string())
    if len(odds) <= 2:
        print("\n  ** WARNING: odds are effectively constant across the board.")
        print("  ** That is a placeholder price, not captured market juice.")
        print("  ** Hit rates below remain valid; units/ROI are built on an assumed price.")

    section("FALSIFICATION CHECK 2 — slate coverage (who is missing)")
    for (season, week), g in d.groupby(["season", "week"]):
        teams = set(g["team"].astype(str).str.upper())
        print(f"\n  {int(season)} wk{int(week):02d}: {len(teams)}/{NFL_TEAMS} teams have any graded row")
        q = g[g["market"].astype(str).eq(QB_MARKET)]
        qteams = set(q["team"].astype(str).str.upper())
        print(f"    {QB_MARKET}: {len(qteams)} teams, {len(q.drop_duplicates(subset=[c for c in DEDUP_KEY if c in q.columns]))} unique bets")
        missing = sorted(teams - qteams)
        if missing:
            print(f"    teams with a board row but NO graded {QB_MARKET} bet: {', '.join(missing)}")
        if len(qteams) < NFL_TEAMS:
            print(f"    -> {NFL_TEAMS - len(qteams)} teams unaccounted for; "
                  f"selection bias is unresolved until each is explained")

    section("FALSIFICATION CHECK 3 — side balance by market")
    bal = (d.assign(side=d["side"].astype(str).str.upper())
             .groupby(["market", "side"])["bet_result"]
             .value_counts().unstack(fill_value=0))
    print(bal.to_string())

    section("FALSIFICATION CHECK 4 — Best Snapshot quote vs consensus")
    print("The deployed betting layer chooses the highest-EV real book+line offer")
    print("and its higher-EV side; nonpositive-EV player-markets PASS. Consensus")
    print("line is retained only as a diagnostic. The spread below measures how far")
    print("the production-selected quote sat from the cross-book median.")
    if "consensus_line" in d.columns and "vegas_line" in d.columns:
        gap = (pd.to_numeric(d["vegas_line"], errors="coerce")
               - pd.to_numeric(d["consensus_line"], errors="coerce")).abs()
        off = int((gap > 1e-9).sum())
        print(f"\n  graded rows priced away from consensus: {off} / {len(d)} ({off / len(d):.1%})")
        if off:
            print(f"  median absolute gap where it happens : {gap[gap > 1e-9].median():.2f}")
        straddle = d.loc[
            (pd.to_numeric(d["model_proj"], errors="coerce")
             - pd.to_numeric(d["consensus_line"], errors="coerce")).abs() < gap.fillna(0)
        ]
        print(f"  rows where the projection sits inside the book spread: {len(straddle)}")
        print("  (mean-vs-line direction is diagnostic only; production side is EV-selected)")
    else:
        print("\n  consensus_line is absent from the graded detail -- cannot bound this.")

    # ---- the QB pass-yards hypothesis -----------------------------------
    q = d[d["market"].astype(str).eq(QB_MARKET)].copy()
    if not q.empty:
        key = [c for c in DEDUP_KEY if c in q.columns]
        dq = q.drop_duplicates(subset=key, keep="first")
        section(f"HYPOTHESIS UNDER TEST — {QB_MARKET}, deduplicated on {key}")
        print(f"rows {len(q)} -> unique bets {len(dq)} (collapsed {len(q) - len(dq)})")
        print()
        print(_fmt_table({"combined": _row_stats(dq),
                          **{f"week {w}": _row_stats(g) for w, g in dq.groupby("week")}}))

        dec = _decided(dq)
        n, w = len(dec), int(dec["bet_result"].eq("WIN").sum())
        if n:
            null_p = np.array([breakeven(o) for o in dec["vegas_odds"]], dtype=float)
            be = float(null_p.mean())
            p = w / n
            ci = 1.96 * sqrt(p * (1 - p) / n)
            cluster_p, n_clusters = cluster_score_pvalue(dec, null_p)
            print(f"\n  mean individual breakeven : {be:.4f}")
            print(f"  observed hit rate          : {p:.4f}   95% CI [{p - ci:.4f}, {p + ci:.4f}]")
            print(f"  independent game clusters  : {n_clusters}")
            if np.isfinite(cluster_p):
                print(f"  cluster-aware one-tail p   : {cluster_p:.5f}")
            else:
                print("  cluster-aware one-tail p   : NA (insufficient/invalid clusters)")
            print(
                f"\n  NOTE: the confidence interval is descriptive at the bet-row level; "
                "the hypothesis p-value above is computed from game-cluster score sums."
            )

        print(f"\nevery {QB_MARKET} bet:")
        cols = [c for c in ["week", "player", "team", "opponent", "side", "vegas_line",
                            "model_proj", "actual", "bet_result", "unit_result"] if c in dq.columns]
        print(dq[cols].sort_values(["week", "player"]).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail", type=Path, required=True,
                    help="graded detail CSV from grade_market_track_record_gsis_v1.py --detail-out")
    return report(ap.parse_args().detail)


if __name__ == "__main__":
    raise SystemExit(main())
