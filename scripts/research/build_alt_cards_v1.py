#!/usr/bin/env python3
"""Build alternate-line betting cards under hard, non-negotiable screens.

Every screen here was learned from a specific week-9 failure. The point of this
script is that no leg reaches a card because someone -- human or model -- judged
it reasonable in the moment. A leg is emitted only if it survives every gate,
and every rejection is logged with the gate that killed it.

GATES (all must pass; order is the order football logic demands)

  G1 ON THE FIELD      snap_pct >= SNAP_MIN (mean offense_pct, strictly prior).
                       Failure case: Roschon Johnson, 15.6% snaps, RB3,
                       shipped as a three-method clear at +294.

  G2 REAL ROLE         absolute main_line floor by market. The book's small
                       line is information, not an opportunity.
                       Failure case: 4.5-yard receiving rungs whose players
                       busted to zero 29.4% of the time.

  G3 NOT AGAINST SELF  cushion = model_proj / alt_threshold >= CUSHION_MIN.
                       A cushion below 1.0 is a bet against our own projection.
                       Failure case: Emeka Egbuka alt rec >=5, cushion 0.92,
                       placed on four of five cards. It lost; it was the
                       weakest leg on the board by our own number.

  G4 REAL ALTERNATE    frac_of_line outside [NEAR_MAIN_LO, NEAR_MAIN_HI] and
                       <= FRAC_MAX. A rung sitting on the main line at a
                       main-line price is not an alternate.
                       Failure case: Roschon rec_yards >2.5 off a 3.0 main,
                       presented as an alt line that did not exist by kickoff.

  G5 MARKET ALLOWED    receptions excluded. Blind ROI -30.0% on 1260 week-9
                       rungs; even inside the cushion rule it was -0.6%.

  G6 OWN EVIDENCE      own_games >= OWN_GAMES_MIN and own_rate >= OWN_RATE_MIN,
                       denominator = games PLAYED (snaps > 0), not games with a
                       recorded stat. The stat-line denominator rewards cameos.

  G7 MATCHUP           opponent not in the toughest DEF_EXCLUDE_Q quantile of
                       prior yards allowed to the position group. Against the
                       toughest quartile the measured edge was -1.8%.

  G8 AVAILABLE         not Out/Doubtful/Questionable on the current report, and
                       not on the manual quarantine list.

  G9 BOOK              offered by the requested book, at a price in
                       [PRICE_MIN, PRICE_MAX].

CARD CONSTRUCTION
  * No leg appears on more than one card. Tiers that share legs are not tiers;
    they are correlated copies of one bet. Egbuka on four of five cards took
    out 80% of the board at once.
  * One rung per player per card.
  * Cards are built from the highest-cushion survivors outward.

MARKET EVIDENCE (week 9 COMPLETE -- all 13 alt-covered games, 5115 rungs)
  Paired on the same pool (FanDuel, this builder's price window), so blind and
  gated describe the same takeable offers:

                 blind n   blind ROI   gated n   gated ROI   gated hit
    rec_yards        528       +0.5%        53       +6.2%       88.7%
    rush_yards       247      -15.8%        23       +0.3%       87.0%
    pass_yards       108       -2.6%         0          --          --
    receptions       279       -8.0%         0          --          --

  Whole pool: blind -5.3% (n=1162) -> gated +4.4% (n=76, 88.2% hit).

  Cluster bootstrap by game (13 games, 20k draws, game as the resampling unit
  because legs inside a game are correlated): gated +4.4% [-1.8%, +10.5%],
  blind -4.9% [-16.7%, +3.7%], gap +9.3% [-2.1%, +23.7%], P(gap>0) = 93.8%.

  READ THAT HONESTLY. +4.4% on n=76 over one week is NOT distinguishable from
  zero, and neither is the gap at the 95% level. The gates move the board about
  nine points in the right direction and 94% of draws agree on the sign. That is
  not an established edge and nothing here is promotable until weeks 10+ confirm
  it. This builder reports; it does not certify.

  This block was previously written from a PARTIAL slate (11 of 13 games) and
  cited rec_yards +5.0%/n=48 gated against +2.9% blind. The two late games graded
  at -28.1% and moved blind rec_yards to +0.5%. Partial-slate numbers are not to
  be cited as evidence again.

  See PRISM_ALT_TABLE_SUPPORT_AUDIT_V1 (and its 2026-11-10 addendum) for why the
  clearance table is separately disqualified for most markets -- that audit
  constrains table-priced selection, which this builder does not do. Its
  profitability claim is withdrawn; its ranking and depth rules held.

STRUCTURAL LIMIT -- PARTLY AN ARTIFACT OF THE OLD RANKING RULE (CORRECTED)
  This block previously read that the builder "cannot serve a lottery tier"
  because all 76 week-9 survivors priced between -225 and -1800 with ZERO
  plus-money legs, and concluded that cushion plus the main-line floor
  "mathematically selects heavy favorites."

  That was the RANKING rule, not the gates. Cushion-descending ordering keeps
  the deepest rung on every ladder, and deepest is most expensive. Ranking by
  leg_value instead (see leg_value) leaves every gate untouched and produces
  materially cheaper cards from the same survivors -- on the week-10 TNF board,
  +115 and +151 where cushion-ranking gave -506 and -466, needing 68-74% per
  leg instead of 91-94%. The price distribution of SURVIVORS is a real
  constraint; the price of the CARDS was mostly self-inflicted.

  What stands unchanged: HIGH/LOTTERY still cannot be certified by anything
  validated here. They come from the >+300 price band (-46.5% on 1018 week-9
  rungs) and the >1.6x depth band (-53.5%), and FRAC_MAX/PRICE_MAX exclude both
  by design. Label them as uncertified speculation; do not present them as a
  deeper version of a gated leg. A five-tier ladder also needs 20 disjoint
  players, which a single-game slate cannot supply.

This script decides nothing. It reports what survives and what did not.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

SNAP_MIN = 0.30
CUSHION_MIN = 1.50
# Rung depth cap. PRISM_ALT_TABLE_SUPPORT_AUDIT_V1 measured rungs beyond 1.6x of
# the line at -55.1% ROI across all markets (n=593), with the clearance table
# overstating them by -0.0286. 1.50 already sits inside that and is kept.
FRAC_MAX = 1.50
NEAR_MAIN_LO, NEAR_MAIN_HI = 0.85, 1.15
OWN_GAMES_MIN = 5
OWN_RATE_MIN = 0.60
DEF_EXCLUDE_Q = 0.25
PRICE_MIN, PRICE_MAX = -2500, 130
# Markets this builder will not emit.
#
# Measured on week 9 with the exclusion lifted, so these are the builder's own
# numbers rather than a verdict imported from the table audit:
#
#   rec_yards        53 legs, 88.7% hit, +6.2% ROI
#   rush_yards       23 legs, 87.0% hit, +0.3% ROI
#   pass_yards        0 legs -- eliminated by the role floor and cushion gates
#   rush_rec_yards    0 legs -- same
#   receptions        0 legs -- same
#
# Regraded 2026-11-10 on the complete 13-game board. Both emitting markets are
# >= 0 gated, so no exclusion change is warranted.
#
# `rush_yards` is NOT excluded despite returning -15.8% blind and being one of
# the markets PRISM_ALT_TABLE_SUPPORT_AUDIT_V1 disqualifies for TABLE-priced
# selection. That audit is about the clearance table's calibration; this builder
# never reads the table. Gated, the market returned +0.3%, so excluding it would
# discard a non-negative. n=23 is thin and that figure may not survive a slate.
#
# `rush_rec_yards` and `receptions` produce zero legs here anyway. They are
# listed explicitly so that a future loosening of the role floor or cushion
# cannot silently admit them: blind they returned -28.8% and -30.7%, and
# rush_rec_yards is the worst-calibrated cell family in the repository at
# -0.0809, every frac band negative from 0.7x upward.
EXCLUDED_MARKETS = ("receptions", "rush_rec_yards", "anytime_td")
MAIN_LINE_FLOOR = {"rec_yards": 18.0, "rush_yards": 15.0,
                   "pass_yards": 30.0, "rush_rec_yards": 30.0}


def _implied(american: float) -> float:
    a = float(american)
    return (-a) / ((-a) + 100.0) if a < 0 else 100.0 / (a + 100.0)


def leg_value(row) -> float:
    """Own-evidence edge over the offered price: own_rate - implied probability.

    THIS IS THE RANKING KEY, AND CUSHION IS NOT.
    ------------------------------------------------------------------
    G3 cushion is a correct FILTER -- below CUSHION_MIN a leg is a bet against
    our own projection and must go. But this builder also used cushion as the
    ORDERING key, which selects, for every player, the DEEPEST rung on his
    ladder. Deepest is also most expensive, so the ordering was
    anti-correlated with value and the docstring's "structural limit" (all 76
    week-9 survivors between -225 and -1800, zero plus-money legs, a three-leg
    card paying about +60) was an artifact of the ranking rule rather than a
    property of the gates.

    Measured on the 2026 week-10 TNF board (run 43327080728, real model_proj,
    all nine gates passed by every leg below), shrinking own_rate toward the
    market price by the measured PRISM overdispersion slope (0.159):

        cushion-ranked  2 legs   -506   needs 91.4%/leg   ROI  +3.0%
        cushion-ranked  3 legs   -466   needs 93.7%/leg   ROI  +0.3%
        value-ranked    2 legs   +115   needs 68.3%/leg   ROI +15.4%
        value-ranked    3 legs   +151   needs 73.6%/leg   ROI +18.5%

    Same gates, same board, same players. The cushion-ranked cards need per-leg
    hit rates of 91-94% when the gates' own measured week-9 rate was 88.2%; the
    value-ranked cards need 68-74%.

    Ranking by the raw edge is deliberate even though week 9 showed own_rate
    overstates by ~12 points: shrinkage toward market is a monotone transform of
    (own_rate - implied), so it does not change the ORDER, only the magnitude.
    The overstatement matters for sizing and for believing a quoted ROI. It does
    not matter for choosing which rung on a ladder to take.
    """
    og = row.get("own_games")
    oh = row.get("own_hits")
    px = row.get("offered_american")
    if not og or px is None:
        return float("-inf")
    try:
        return ((oh or 0) / og) - _implied(px)
    except Exception:
        return float("-inf")


def _fail(log, row, gate, detail):
    log.append({"player": row.get("player"), "market": row.get("market"),
                "alt_threshold": row.get("alt_threshold"),
                "offered_american": row.get("offered_american"),
                "gate": gate, "detail": detail})


def apply_gates(df: pd.DataFrame, book: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (survivors, rejection_log). Pure function of the input frame."""
    log: list[dict] = []
    keep = []
    dq = df.get("def_allowed")
    cut = float(np.nanquantile(dq, DEF_EXCLUDE_Q)) if dq is not None and dq.notna().any() else None
    for row in df.to_dict("records"):
        if book and str(row.get("book", "")).lower() != book.lower():
            _fail(log, row, "G9_book", f"book={row.get('book')}"); continue
        px = row.get("offered_american")
        if px is None or not np.isfinite(px) or not (PRICE_MIN <= px <= PRICE_MAX):
            _fail(log, row, "G9_price", f"price={px}"); continue
        if str(row.get("market")) in EXCLUDED_MARKETS:
            _fail(log, row, "G5_market", str(row.get("market"))); continue
        snap = row.get("snap_pct")
        if snap is None or not np.isfinite(snap):
            _fail(log, row, "G1_snap_unknown", "no snap data"); continue
        if snap < SNAP_MIN:
            _fail(log, row, "G1_snap", f"snap_pct={snap:.3f} < {SNAP_MIN}"); continue
        floor = MAIN_LINE_FLOOR.get(str(row.get("market")))
        ml = row.get("main_line")
        if floor is None or ml is None or not np.isfinite(ml):
            _fail(log, row, "G2_role_unknown", f"main_line={ml}"); continue
        if ml < floor:
            _fail(log, row, "G2_role", f"main_line={ml:.1f} < {floor:.0f}"); continue
        proj, thr = row.get("model_proj"), row.get("alt_threshold")
        if not (proj and thr) or not np.isfinite(proj) or not np.isfinite(thr) or thr <= 0:
            _fail(log, row, "G3_cushion_unknown", f"proj={proj} thr={thr}"); continue
        cush = proj / thr
        if cush < CUSHION_MIN:
            _fail(log, row, "G3_cushion", f"cushion={cush:.2f} < {CUSHION_MIN}"
                  + ("  (BETTING AGAINST OUR OWN PROJECTION)" if cush < 1.0 else "")); continue
        frac = row.get("frac_of_line")
        if frac is None or not np.isfinite(frac):
            _fail(log, row, "G4_frac_unknown", "no frac_of_line"); continue
        if frac > FRAC_MAX:
            _fail(log, row, "G4_frac_deep", f"frac={frac:.2f} > {FRAC_MAX}"); continue
        if NEAR_MAIN_LO <= frac <= NEAR_MAIN_HI:
            _fail(log, row, "G4_near_main", f"frac={frac:.2f} is the main line in disguise"); continue
        og, oh = row.get("own_games"), row.get("own_hits")
        if og is None or not np.isfinite(og) or og < OWN_GAMES_MIN:
            _fail(log, row, "G6_own_games", f"own_games={og} < {OWN_GAMES_MIN}"); continue
        rate = (oh or 0) / og
        if rate < OWN_RATE_MIN:
            _fail(log, row, "G6_own_rate", f"own_rate={rate:.2f} < {OWN_RATE_MIN}"); continue
        if cut is not None:
            da = row.get("def_allowed")
            if da is None or not np.isfinite(da):
                _fail(log, row, "G7_matchup_unknown", "no opponent data"); continue
            if da < cut:
                _fail(log, row, "G7_matchup", f"opponent allows {da:.1f} < {cut:.1f} (toughest quartile)"); continue
        st = str(row.get("injury_status") or "").strip().lower()
        if st in {"out", "doubtful", "questionable"}:
            _fail(log, row, "G8_injury", f"status={st}"); continue
        if bool(row.get("quarantined")):
            _fail(log, row, "G8_quarantine", "manual quarantine"); continue
        row["cushion"] = cush
        row["own_rate"] = rate
        row["leg_value"] = leg_value(row)
        keep.append(row)
    surv = pd.DataFrame(keep)
    if not surv.empty:
        # Cushion has already done its job as a gate. Order by value so the
        # rung kept for each player is the one worth betting, not the deepest.
        surv = surv.sort_values("leg_value", ascending=False).drop_duplicates(["player", "market"])
    return surv, pd.DataFrame(log)


def build_cards(surv: pd.DataFrame, sizes=(2, 3, 4, 5, 6)) -> pd.DataFrame:
    """Disjoint cards: no leg is reused across tiers."""
    if surv.empty:
        return pd.DataFrame()
    pool = surv.sort_values("leg_value", ascending=False).drop_duplicates("player").copy()
    names = ["SAFE", "MODERATE", "MEDIUM", "HIGH RISK", "LOTTERY"]
    out, used = [], set()
    for name, n in zip(names, sizes):
        avail = pool[~pool.player.isin(used)]
        if len(avail) < n:
            break
        legs = avail.head(n).copy()
        used.update(legs.player)
        dec = np.prod([1 + (a / 100 if a > 0 else 100 / -a) for a in legs.offered_american]) - 1
        legs["tier"] = name
        legs["card_odds"] = int(round(dec * 100)) if dec >= 1 else int(-round(100 / dec))
        legs["card_legs"] = n
        out.append(legs)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True, help="CSV of candidate alt legs with the gate columns")
    ap.add_argument("--book", default=None, help="restrict to one book, e.g. fanduel")
    ap.add_argument("--out-cards", default="alt_cards.csv")
    ap.add_argument("--out-rejects", default="alt_rejections.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.legs, low_memory=False)
    surv, rej = apply_gates(df, a.book)
    cards = build_cards(surv)
    Path(a.out_cards).parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(a.out_cards, index=False)
    rej.to_csv(a.out_rejects, index=False)
    print(f"candidates {len(df)} -> survivors {len(surv)} -> carded {len(cards)}")
    if not rej.empty:
        print("\nrejections by gate:")
        print(rej.gate.value_counts().to_string())
    if cards.empty:
        print("\nNO CARDS. Not a failure -- the slate did not offer qualifying legs.")
        return 0
    for tier, g in cards.groupby("tier", sort=False):
        print(f"\n{tier}: {g.card_legs.iat[0]} legs @ {g.card_odds.iat[0]:+d}")
        for r in g.itertuples():
            print(f"   {r.player:20s}{r.market:12s}>{r.alt_threshold:6.1f} "
                  f"(main {r.main_line:5.1f}) {int(r.offered_american):+6d} "
                  f"cush {r.cushion:4.2f}x snap {r.snap_pct:.0%} own {int(r.own_hits)}/{int(r.own_games)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
