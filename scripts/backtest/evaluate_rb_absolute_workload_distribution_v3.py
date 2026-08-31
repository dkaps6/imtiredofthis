"""M95E mechanical input-completeness corrections.

This wrapper changes no model family, feature set, blend grid, selection score,
tail threshold grid, distribution calibration rule, or 2025 tuning protocol.
It only (1) constrains the frozen M94D 2024 comparison file to W13-18,
(2) reconciles player keys by exact season/week/team/player display identity
when the frozen sources use different key normalization, and (3) rebuilds
holdout/validation workload labels from the complete frozen RB comparison
universe so low-volume RB/FB rows missing from M95B feature enrichment do not
produce NA truth labels.
"""
import scripts.backtest.evaluate_rb_absolute_workload_distribution as m

_original_load = m.load_m94d_rb
_original_prepare = m.prepare_eval_frame


def _load_filtered(root):
    hold, validation = _original_load(root)
    hold = hold.loc[m.num(hold["week"]).ge(13)].copy()
    return hold, validation


def _reconcile_keys(feature_trace, rb_baseline):
    rb = rb_baseline.copy()
    if "player" not in rb.columns or "player" not in feature_trace.columns:
        return rb
    keymap = feature_trace[["season", "week", "team", "player", "player_clean_key"]].drop_duplicates(
        ["season", "week", "team", "player"]
    ).rename(columns={"player_clean_key": "_feature_player_key"})
    rb = rb.merge(keymap, on=["season", "week", "team", "player"], how="left", validate="many_to_one")
    rb["player_clean_key"] = rb["_feature_player_key"].fillna(rb["player_clean_key"])
    return rb.drop(columns=["_feature_player_key"])


def _prepare_complete(feature_trace, rb_baseline, team_env):
    rb = _reconcile_keys(feature_trace, rb_baseline)
    out = _original_prepare(feature_trace, rb, team_env)

    # Complete observed workload truth comes from the frozen M94D comparison
    # universe. Rebuild all derived labels from that truth rather than leaving
    # rows without M95B enrichment as NA.
    out["actual_carries"] = m.num(out["actual_rush_att"])
    out["actual_20plus"] = out["actual_carries"].ge(20).astype(int)
    out["actual_25plus"] = out["actual_carries"].ge(25).astype(int)
    out["actual_rb_pool"] = out.groupby(m.TEAM_KEYS)["actual_carries"].transform("sum")
    out["actual_player_rb_share"] = out["actual_carries"] / out["actual_rb_pool"].replace(0, float("nan"))
    out["actual_total_team_rush_pbp"] = m.num(out["actual_rush_att_pbp"])
    out["actual_room_share"] = out["actual_rb_pool"] / out["actual_total_team_rush_pbp"].replace(0, float("nan"))
    out["actual_abs_team_share"] = out["actual_carries"] / out["actual_total_team_rush_pbp"].replace(0, float("nan"))
    return out


m.load_m94d_rb = _load_filtered
m.prepare_eval_frame = _prepare_complete

if __name__ == "__main__":
    raise SystemExit(m.main())
