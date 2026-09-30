"""Verification-only pricing/comparison helpers for RNG isolation candidate.

Copied narrowly from the frozen downstream-materiality methodology so the
production-repair branch does not depend on the closed research branch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.discrete_count_alignment_v1 import align_prealigned_outcomes
from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.modeling.qb_pass_synthesis_v1 import (
    build_feature_dict,
    predict_correction as predict_qb_synthesis,
)
from scripts.operations.grade_market_track_record_v1 import _ev_roi, select_model_bet
from scripts.simulation_v2 import MARKET_MAP
from scripts.utils.player_identity_v3 import player_name_key

ITERATIONS = 25000
SUPPORTED = {"pass_yards", "rush_yards", "rec_yards", "receptions"}
KEY = ["event_id", "player_clean_key", "market"]
PM_KEY = ["season", "week", "event_id", "player", "market"]
QUOTE_KEY = [
    "season", "week", "event_id", "player_clean_key",
    "market", "_book_key", "vegas_line",
]


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def canon_market(value: object) -> str:
    raw = str(value or "").lower().strip()
    return MARKET_MAP.get(raw, raw)


def finite(value: object, default=np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def provider_aliases(paid: pd.DataFrame) -> dict[str, str]:
    p = paid.copy()
    p["team"] = p["team"].map(canon_team)
    p["opponent"] = p["opponent"].map(canon_team)

    def canonical(r):
        a, b = sorted([str(r["team"]), str(r["opponent"])])
        return f"{int(r['season'])}_{int(r['week']):02d}_{a}_{b}"

    p["_canonical_event_id"] = p.apply(canonical, axis=1)
    aliases = {}
    for cg, g in p.groupby("_canonical_event_id", sort=False):
        ids = sorted(set(g["event_id"].astype(str)))
        if len(ids) != 1:
            raise RuntimeError(
                f"provider event identity ambiguous canonical={cg}: {ids}"
            )
        aliases[str(cg)] = ids[0]
    return aliases


def provider_identity_aliases(
    paid: pd.DataFrame,
    event_aliases: dict[str, str],
) -> dict[tuple[str, str], tuple[str, str]]:
    aliases = {}
    cols = [
        "season", "week", "team", "opponent",
        "event_id", "player", "player_clean_key",
    ]
    for r in paid[cols].drop_duplicates().itertuples(index=False):
        a, b = sorted([canon_team(r.team), canon_team(r.opponent)])
        canonical_game = f"{int(r.season)}_{int(r.week):02d}_{a}_{b}"
        canonical_player = str(
            player_name_key(r.player, strip_suffix=True) or ""
        ).strip()
        provider_game = str(r.event_id)
        provider_player = str(r.player_clean_key)
        if not canonical_player or not provider_player:
            raise RuntimeError(
                f"blank provider/canonical player identity player={r.player}"
            )
        key = (canonical_game, canonical_player)
        value = (provider_game, provider_player)
        prior = aliases.get(key)
        if prior is not None and prior != value:
            raise RuntimeError(
                f"ambiguous provider player identity canonical={key} "
                f"prior={prior} new={value}"
            )
        aliases[key] = value
        if event_aliases.get(canonical_game) != provider_game:
            raise RuntimeError(
                f"provider event/player alias disagreement canonical={canonical_game}"
            )
    return aliases


def install_provider_aliases(
    result,
    event_aliases: dict[str, str],
    identity_aliases: dict[tuple[str, str], tuple[str, str]],
) -> None:
    additions = {}
    for (game, pkey, market), values in list(result.values.items()):
        provider = event_aliases.get(str(game))
        if provider:
            additions[(provider, pkey, market)] = values
        ident = identity_aliases.get((str(game), str(pkey)))
        if ident:
            provider_game, provider_player = ident
            additions[(str(game), provider_player, market)] = values
            additions[(provider_game, provider_player, market)] = values
    result.values.update(additions)


def representative_rule_rows(root: Path) -> dict[tuple[str, str, str], pd.Series]:
    r = read_csv(
        root / "data/model_rule_simulation_inputs.csv",
        "model rule simulation inputs",
    )
    r["_canonical_market"] = r["market"].map(canon_market)
    out = {}
    for _, row in r.sort_values(
        ["event_id", "player_clean_key", "_canonical_market", "book"],
        kind="mergesort",
    ).iterrows():
        key = (
            str(row["event_id"]),
            str(row["player_clean_key"]),
            str(row["_canonical_market"]),
        )
        out.setdefault(key, row)
    return out


def target_mean_full(
    *,
    market: str,
    mc_proj: float,
    paid_meta: pd.Series,
    rule_row: pd.Series | None,
    weights: pd.DataFrame,
    qb_bundle: dict,
) -> tuple[float, float]:
    comp = pd.DataFrame(
        [
            {
                "market": market,
                "mc_proj": mc_proj,
                "ml_proj": paid_meta.get("ml_proj"),
                "state_proj": paid_meta.get("state_proj"),
            }
        ]
    )
    ens = apply_ensemble(comp, weights=weights).iloc[0]
    ensemble = float(ens["ensemble_proj"])
    if market != "pass_yards":
        return ensemble, ensemble
    if rule_row is None:
        raise RuntimeError(
            "missing QB rule row key="
            f"{(paid_meta.get('event_id'), paid_meta.get('player_clean_key'), market)}"
        )
    features = build_feature_dict(
        rule_row,
        base_proj=ensemble,
        mc_proj=mc_proj,
        team_context=qb_bundle["team_context"],
        player_logs=qb_bundle["player_logs"],
        weather=qb_bundle["weather"],
        season=2026,
        week=3,
    )
    synth, _, _ = predict_qb_synthesis(
        features, artifact=qb_bundle["artifact"]
    )
    return float(synth), ensemble


def price_stage(
    result,
    paid: pd.DataFrame,
    rule_rows: dict[tuple[str, str, str], pd.Series],
    weights: pd.DataFrame,
    qb_bundle: dict,
) -> dict[str, pd.DataFrame]:
    meta = paid.drop_duplicates(KEY, keep="first").copy()
    paid_by_key = {
        tuple(map(str, r)): g.copy()
        for r, g in paid.groupby(KEY, sort=False)
    }
    rows = {
        "SHAPE_ONLY_FIXED_FINAL_MEAN": [],
        "FULL_DOWNSTREAM_PROPAGATION": [],
    }

    for mr in meta.itertuples(index=False):
        key = (str(mr.event_id), str(mr.player_clean_key), str(mr.market))
        arr = result.values.get(key)
        if arr is None or len(arr) != ITERATIONS:
            raise RuntimeError(f"missing simulated pricing key={key}")
        base = np.asarray(arr, dtype=float)
        if str(mr.market) == "pass_yards":
            conv = finite(getattr(mr, "qb_attempt_conversion", np.nan))
            share = finite(getattr(mr, "qb_pass_att_share", 1.0), 1.0)
            if not np.isfinite(conv):
                raise RuntimeError(f"missing QB attempt conversion key={key}")
            base = base * conv * share

        mc = float(base.mean())
        pm = paid_by_key[key].iloc[0]
        shape_target = float(pm["model_proj"])
        rule = rule_rows.get(key)
        full_target, ensemble = target_mean_full(
            market=str(mr.market),
            mc_proj=mc,
            paid_meta=pm,
            rule_row=rule,
            weights=weights,
            qb_bundle=qb_bundle,
        )

        for surface, target in [
            ("SHAPE_ONLY_FIXED_FINAL_MEAN", shape_target),
            ("FULL_DOWNSTREAM_PROPAGATION", full_target),
        ]:
            eligible = bool(np.isfinite(mc) and mc > 0 and np.isfinite(target))
            adjusted = (
                base * max(0.0, target / mc) if eligible else base.copy()
            )
            adjusted, _ = align_prealigned_outcomes(
                adjusted,
                market=str(mr.market),
                eligible=eligible,
                target_mean=target,
            )
            group = paid_by_key[key]
            over_by_line = {
                float(line): float(np.mean(adjusted > float(line)))
                for line in pd.to_numeric(
                    group["vegas_line"], errors="raise"
                ).unique()
            }
            for rr in group.itertuples(index=False):
                line = float(rr.vegas_line)
                po = over_by_line[line]
                prob = po if str(rr.side).upper() == "OVER" else 1.0 - po
                rec = rr._asdict()
                rec["fair_prob"] = float(prob)
                rec["stage_mc_proj"] = mc
                rec["stage_ensemble_proj"] = ensemble
                rec["stage_target_mean"] = float(target)
                rec["stage_model_proj"] = float(np.mean(adjusted))
                rec["stage_ev_roi"] = float(_ev_roi(prob, rr.vegas_odds))
                rows[surface].append(rec)

    return {k: pd.DataFrame(v) for k, v in rows.items()}


def quote_state(board: pd.DataFrame) -> pd.DataFrame:
    b = board.copy()
    book = (
        b.get("book", pd.Series("", index=b.index))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    title = (
        b.get("book_title", pd.Series("", index=b.index))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    b["_book_key"] = book.mask(book.eq(""), title).mask(
        lambda s: s.eq(""), "~missing-book"
    )
    b["_ev"] = [
        _ev_roi(p, o)
        for p, o in zip(
            pd.to_numeric(b["fair_prob"], errors="coerce"),
            pd.to_numeric(b["vegas_odds"], errors="coerce"),
        )
    ]
    b["_side_rank"] = (
        b["side"].astype(str).str.upper().map({"OVER": 0, "UNDER": 1}).fillna(9)
    )
    keys = [c for c in QUOTE_KEY if c in b.columns]
    q = b.sort_values(
        keys + ["_ev", "_side_rank"],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    )
    q = q.drop_duplicates(keys, keep="first").copy()
    q["quote_has_edge"] = q["_ev"].gt(0)
    return q


def best_ev_table(board: pd.DataFrame) -> pd.DataFrame:
    q = quote_state(board)
    key = [c for c in PM_KEY if c in q.columns]
    q["_book_key"] = (
        q.get("book", pd.Series("", index=q.index))
        .astype("string")
        .fillna("")
        .str.lower()
    )
    q = q.sort_values(
        key + ["_ev", "_book_key", "vegas_line"],
        ascending=[True] * len(key) + [False, True, True],
        kind="mergesort",
    )
    out = q.drop_duplicates(key, keep="first").copy()
    out["best_ev"] = out["_ev"]
    return out


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(float)
    rb = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def qtile(series: pd.Series, p: float) -> float:
    return (
        float(pd.to_numeric(series, errors="coerce").quantile(p))
        if len(series)
        else float("nan")
    )


def compare_boards(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protected_keys: set[tuple[str, str]],
    *,
    comparison: str,
    surface: str,
    kind: str,
) -> dict:
    l = left.loc[
        [
            (str(e), str(p)) in protected_keys
            for e, p in zip(left["event_id"], left["player_clean_key"])
        ]
    ].copy()
    r = right.loc[
        [
            (str(e), str(p)) in protected_keys
            for e, p in zip(right["event_id"], right["player_clean_key"])
        ]
    ].copy()
    if set(l["paid_row_id"]) != set(r["paid_row_id"]):
        raise RuntimeError(
            f"paid-row universe mismatch {comparison} {surface}"
        )

    m = l[
        [
            "paid_row_id", "fair_prob", "stage_ev_roi", "event_id",
            "player", "player_clean_key", "team", "market", "book",
            "vegas_line", "side",
        ]
    ].merge(
        r[["paid_row_id", "fair_prob", "stage_ev_roi"]],
        on="paid_row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    m["abs_prob_delta"] = (
        m["fair_prob_right"] - m["fair_prob_left"]
    ).abs()
    m["abs_ev_delta"] = (
        m["stage_ev_roi_right"] - m["stage_ev_roi_left"]
    ).abs()

    ql = quote_state(l)
    qr = quote_state(r)
    qkeys = [c for c in QUOTE_KEY if c in ql.columns]
    qm = ql[qkeys + ["side", "_ev", "quote_has_edge"]].merge(
        qr[qkeys + ["side", "_ev", "quote_has_edge"]],
        on=qkeys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    quote_side_flips = int(
        qm["side_left"].astype(str).ne(qm["side_right"].astype(str)).sum()
    )
    quote_edge_flips = int(
        qm["quote_has_edge_left"].ne(qm["quote_has_edge_right"]).sum()
    )

    sl = select_model_bet(l)
    sr = select_model_bet(r)
    pkeys = [c for c in PM_KEY if c in l.columns]
    all_pm = l[pkeys].drop_duplicates().merge(
        r[pkeys].drop_duplicates(), on=pkeys, how="outer"
    )

    def sel_map(s: pd.DataFrame) -> dict[tuple, tuple]:
        out = {}
        for rr in s.itertuples(index=False):
            k = tuple(getattr(rr, c) for c in pkeys)
            out[k] = (
                str(getattr(rr, "side", "")),
                str(getattr(rr, "book", "")),
                float(getattr(rr, "vegas_line")),
                float(getattr(rr, "vegas_odds")),
            )
        return out

    lm, rm = sel_map(sl), sel_map(sr)
    bet_pass = side_flip = identity = 0
    for rr in all_pm.itertuples(index=False):
        k = tuple(getattr(rr, c) for c in pkeys)
        a, b = lm.get(k), rm.get(k)
        if (a is None) != (b is None):
            bet_pass += 1
        if a is not None and b is not None:
            if a[0] != b[0]:
                side_flip += 1
            if a != b:
                identity += 1

    bl = best_ev_table(l)
    br = best_ev_table(r)
    bm = bl[pkeys + ["best_ev"]].merge(
        br[pkeys + ["best_ev"]],
        on=pkeys,
        how="inner",
        suffixes=("_left", "_right"),
    )
    corr = (
        rank_corr(bm["best_ev_left"], bm["best_ev_right"])
        if len(bm)
        else float("nan")
    )
    best_abs = (
        (bm["best_ev_right"] - bm["best_ev_left"]).abs()
        if len(bm)
        else pd.Series(dtype=float)
    )

    def top_turnover(frame_a: pd.DataFrame, frame_b: pd.DataFrame, k: int) -> int:
        aa = frame_a.sort_values("best_ev", ascending=False).head(k)
        bb = frame_b.sort_values("best_ev", ascending=False).head(k)
        ka = {tuple(x) for x in aa[pkeys].itertuples(index=False, name=None)}
        kb = {tuple(x) for x in bb[pkeys].itertuples(index=False, name=None)}
        return int(min(len(ka), len(kb)) - len(ka & kb))

    return {
        "kind": kind,
        "comparison": comparison,
        "surface": surface,
        "protected_side_rows": int(len(m)),
        "protected_quotes": int(len(qm)),
        "protected_player_markets": int(len(all_pm)),
        "mean_abs_prob_delta": float(m["abs_prob_delta"].mean()),
        "median_abs_prob_delta": float(m["abs_prob_delta"].median()),
        "p90_abs_prob_delta": qtile(m["abs_prob_delta"], .90),
        "p95_abs_prob_delta": qtile(m["abs_prob_delta"], .95),
        "p99_abs_prob_delta": qtile(m["abs_prob_delta"], .99),
        "max_abs_prob_delta": float(m["abs_prob_delta"].max()),
        "mean_abs_ev_delta": float(m["abs_ev_delta"].mean()),
        "p95_abs_ev_delta": qtile(m["abs_ev_delta"], .95),
        "p99_abs_ev_delta": qtile(m["abs_ev_delta"], .99),
        "max_abs_ev_delta": float(m["abs_ev_delta"].max()),
        "quote_preferred_side_flips": quote_side_flips,
        "quote_has_edge_pass_flips": quote_edge_flips,
        "best_snapshot_bet_pass_flips": int(bet_pass),
        "best_snapshot_side_flips": int(side_flip),
        "best_snapshot_identity_changes": int(identity),
        "mean_abs_best_ev_delta": float(best_abs.mean()),
        "max_abs_best_ev_delta": float(best_abs.max()),
        "best_ev_spearman": corr,
        "top10_turnover": top_turnover(bl, br, 10),
        "top25_turnover": top_turnover(bl, br, 25),
    }
