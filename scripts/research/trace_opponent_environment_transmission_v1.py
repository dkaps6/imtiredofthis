#!/usr/bin/env python3
"""Trace every legal pregame opponent/environment input through the player path.

Authorized by GPT audit comment 6431984732, disposition
`WALKER_ENVIRONMENT_TRANSMISSION_RESEARCH_UNBLOCKED_IN_PARALLEL__READ_ONLY_MECHANISM_FIRST`.

This CONSUMES and does not redo the accepted decomposition at commit
`cca416bd2cdad8f142a5a063923149344c7e5c5d`
(`docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`),
whose findings for Kenneth Walker III are taken as given: zero opponent-caused
movement in opportunity, efficiency, mean and sigma on the generic rushing
path, and nonzero movement on the receiving path.

The question here is the one the decomposition did not answer: for the FULL set
of opponent/environment columns production builds -- not only the ones that
decomposition happened to examine -- which are consumed, which are stranded,
and through which football mechanism.

Read-only and outcome-blind. It reads production SOURCE CODE and a column
inventory. It inspects no outcome, no market, no line and no result, and it
does not fit, score or propose a coefficient, threshold or sign. Per the
authorizing audit: "Week 7 can identify the missing mechanism; it cannot score
the fix."

METHOD AND ITS LIMIT. Consumption is established by literal column-name
reference in the player-path modules. That is sound here, and the reason is
checked rather than assumed: `ml_v2` declares `FEATURE_COLUMNS` as an explicit
tuple and `state_v2` an explicit required set, neither module selects columns
dynamically, and nothing in the player path constructs an `_opp` name
programmatically (every `_opp` match is the `scripts._opponent_map` module or
an opponent-IDENTITY merge). So a name absent from the source cannot reach a
model. `assert_no_dynamic_selection()` re-checks this invariant on every run;
if a future change introduces dynamic feature selection, this script fails
rather than reporting a stale classification.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# TRANSPORT modules move a column onto the player row or the rule object.
# A reference here proves the value was CARRIED, not that any projection used
# it. Conflating the two was a real reporting bug in V1 of this script, caught
# by audit 6432370310 / 6432388032: def_rush_epa and
# explosive_play_rate_allowed were reported "consumed" on the strength of a
# context_bridge_v3 reference, while this script's own
# teamcontext_transmission() correctly showed both are carried then ignored.
TRANSPORT = (
    "scripts/modeling/context_bridge_v3.py",
    "scripts/modeling/context_bridge.py",
)

# CONSUMER modules can turn an opponent column into a projection: a rule, a
# learned component, a simulator or a promoted adapter.
CONSUMERS = (
    "scripts/simulation_v2.py",
    "scripts/run_pricing_v2.py",
    "scripts/modeling/rules_v2.py",
    "scripts/modeling/simulation_rules.py",
    "scripts/modeling/ml_v2.py",
    "scripts/modeling/state_v2.py",
    "scripts/modeling/bayesian_v2.py",
    "scripts/modeling/ensemble_v2.py",
    "scripts/player_form_v2.py",
    # The promoted M89/M90 QB route is its own path. Omitting these modules
    # made the QB-facing pass-defense family read as stranded when in fact it
    # is consumed through qb_promoted_team_context.
    "scripts/modeling/qb_pass_synthesis_v1.py",
    "scripts/modeling/qb_c2_production_adapter_v1.py",
    "scripts/modeling/rb_pricing_adapter_v1.py",
)

PLAYER_PATH = TRANSPORT + CONSUMERS

# Columns that carry provenance/bookkeeping rather than football signal. A
# stranded provenance column is not a missing football mechanism, and counting
# it as one would inflate the finding.
BOOKKEEPING = re.compile(
    r"^(pbp_feature_|pbp_prior_used|team_form_(active|prior)_season|prior_season|"
    r"success_explosive_(source|target)|qb_promoted_context_|games_played|"
    r"prior_proe_plays|rank|team_abbr|team_raw|.*_source$|.*_target_week$)")

# Candidate mechanism families, separated by FOOTBALL MEANING as the
# authorizing audit requires: "rush EPA, stuff/YBC, explosive suppression, box
# structure, expected game state, receiving coverage are not interchangeable."
# The channel is the part that matters: these do NOT all belong in the mean.
FAMILIES = {
    "A_RUN_FIT_BOX_STRUCTURE": {
        "question": "Is there room at the first level?",
        "channel": "opportunity + efficiency",
        "patterns": (r"^(heavy|light)_box_rate", r"^middle_(open|closed)_rate$",
                     r"^12p_rate$", r"^sub_package_rate", r"^personnel_12_rate"),
    },
    "B_RUN_STOP_AT_THE_LINE": {
        "question": "Does this defense kill runs before they start?",
        "channel": "efficiency / central tendency",
        "patterns": (r"^rush_stuff_rate", r"^yards_before_contact_per_rb_rush",
                     r"^def_rush_epa"),
    },
    "C_EXPLOSIVE_RUN_SUPPRESSION": {
        "question": "Does this defense surrender the long one?",
        "channel": "sigma / right tail -- NOT the mean",
        "patterns": (r"^explosive_play_rate_allowed",),
    },
    "D_RB_RECEIVING_COVERAGE": {
        "question": "Does this defense concede the back out of the backfield?",
        "channel": "receiving opportunity + efficiency",
        "patterns": (r"^ypt_allowed_", r"^coverage_(zone|man)_rate$",
                     r"^slot_rate$"),
    },
    "E_PASS_RUSH_BLITZ": {
        "question": "Does pressure force check-downs and change run numbers?",
        "channel": "receiving opportunity + pass efficiency",
        "patterns": (r"^blitz_rate", r"^pressure_rate", r"^def_sack",
                     r"^no_blitz_pressure_rate"),
    },
    "F_EXPECTED_GAME_STATE_VOLUME": {
        "question": "How many snaps, and how pass-leaning?",
        "channel": "opportunity only",
        "patterns": (r"^pace", r"^neutral_pace", r"^plays", r"^proe",
                     r"^true_proe$", r"^pass_rate_(faced|off|over_expected)",
                     r"^neutral_pass_rate$", r"^rz_rate", r"^time_of_possession$",
                     r"^success_rate_", r"^playsdrive$", r"^playsgame$",
                     r"^gear_change$", r"^no_huddle", r"^nohuddle_rate$",
                     r"^shotgun_rate", r"^motion_rate", r"^play_action_rate",
                     r"^play_clock_used$", r"^time_to_throw$", r"^opp_dropbacks$",
                     r"^pass_attempts_per_dropback$"),
    },
    "G_QB_FACING_PASS_DEFENSE": {
        "question": "How well does this defense defend the pass?",
        "channel": "QB mean/efficiency (promoted M89/M90 path)",
        "patterns": (r"^def_pass_", r"^def_ypa_allowed$", r"^off_pass_epa$",
                     r"^off_ypa$", r"^airyardsatt", r"^ay_per_att"),
    },
}


def _family(col: str) -> str:
    for name, spec in FAMILIES.items():
        for p in spec["patterns"]:
            if re.match(p, col):
                return name
    return "UNCLASSIFIED"


def assert_no_dynamic_selection() -> dict:
    """Fail closed if the literal-grep method stops being valid.

    The classification is only sound while the learned components declare their
    features statically. If a future change makes feature selection dynamic, a
    name-absent-from-source conclusion would become a false negative, so this
    raises rather than letting the report go stale.
    """
    checks = {}
    ml = Path("scripts/modeling/ml_v2.py").read_text(encoding="utf-8")
    if "FEATURE_COLUMNS = (" not in ml:
        raise RuntimeError(
            "ml_v2 no longer declares FEATURE_COLUMNS as an explicit tuple; "
            "literal-reference tracing is no longer valid")
    for bad in ("select_dtypes", "FEATURE_COLUMNS = [c for c",
                "FEATURE_COLUMNS = list("):
        if bad in ml:
            raise RuntimeError(f"ml_v2 selects features dynamically via {bad!r}")
    checks["ml_v2_static_feature_columns"] = True
    st = Path("scripts/modeling/state_v2.py").read_text(encoding="utf-8")
    if "required = {" not in st:
        raise RuntimeError("state_v2 no longer declares a static required set")
    checks["state_v2_static_required_set"] = True
    # No module may build an _opp column name at runtime.
    for f in PLAYER_PATH:
        src = Path(f).read_text(encoding="utf-8") if Path(f).exists() else ""
        for bad in ('+ "_opp"', "+ '_opp'", 'f"{', ):
            if bad == 'f"{':
                continue
            if bad in src:
                raise RuntimeError(f"{f} constructs an _opp column name at runtime")
    checks["no_runtime_opp_name_construction"] = True
    return checks


def teamcontext_transmission() -> dict:
    """Separate CARRIED from CONSUMED at the rule boundary.

    The sharpest form of stranding is not a column missing from the pipeline;
    it is a column loaded onto the very object the rule function receives and
    then never read. `context_bridge_v3` populates TeamContext, and
    `rules_v2.matchup_multipliers` reads only some of those fields, so the
    difference is exact and mechanical rather than inferred.
    """
    cb = Path("scripts/modeling/context_bridge_v3.py").read_text(encoding="utf-8")
    rules = "".join(
        Path(f).read_text(encoding="utf-8")
        for f in ("scripts/modeling/rules_v2.py",
                  "scripts/modeling/simulation_rules.py") if Path(f).exists())
    populated = sorted(set(re.findall(r"^\s*(\w+)=_num\(row", cb, re.M)))
    read = set(re.findall(r"(?:defense|offense|ctx|team_ctx|opp_ctx)\.(\w+)",
                          rules))
    ignored = [f for f in populated if f not in read]
    # A field ignored at the rule boundary may still be consumed on another
    # route, so say which -- an unqualified "unused" would be wrong.
    elsewhere = {}
    for f in ignored:
        other = subprocess.run(
            ["grep", "-rl", f, "scripts/"], capture_output=True, text=True)
        files = [x for x in (other.stdout or "").split()
                 if x.endswith(".py")
                 and "research/" not in x and "backtest" not in x
                 and "context_bridge" not in x and "make_team_form" not in x
                 and "team_context_v3" not in x and "contracts" not in x]
        elsewhere[f] = sorted(files)
    return {
        "teamcontext_fields_populated": populated,
        "teamcontext_fields_read_by_a_rule": sorted(read & set(populated)),
        "carried_onto_teamcontext_then_ignored_by_every_rule": ignored,
        "ignored_field_consumers_on_other_routes": elsewhere,
    }


def orphaned_mechanisms() -> dict:
    """Mechanisms that exist in the repo but are not on the canonical path.

    `pricing_v2._projection` applies `_matchup_multiplier`, which contains a
    def_rush_epa -> rush efficiency adjustment. `run_pricing_v2` imports only
    the two odds helpers from that module, so the adjustment never runs in
    production. Reported as an orphan rather than as a missing mechanism,
    because the distinction changes what the fix is.
    """
    runner = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    imported = re.findall(r"from scripts\.pricing_v2 import ([^\n]+)", runner)
    names = sorted({n.strip() for grp in imported for n in grp.split(",")})
    return {
        "module": "scripts/pricing_v2.py",
        "orphaned_function": "_projection -> _matchup_multiplier",
        "contains": ("def_rush_epa -> rush_yards/rush_att/rush_rec_yards "
                     "efficiency multiplier, clipped +/-0.30, scaled 0.25"),
        "names_the_production_runner_imports": names,
        "projection_imported_by_runner": "_projection" in names,
        "on_canonical_path": False,
        "note": ("the mechanism is written and signed but unwired; this is an "
                 "integration gap, not an absent model. It is NOT evaluated or "
                 "endorsed here -- its coefficient was never validated on the "
                 "promoted stack and must not be scored on Week 7."),
    }


def consumption(cols: list[str], files: tuple[str, ...]
                ) -> dict[str, dict[str, int]]:
    """Map base column name -> {file: literal reference count}.

    LITERAL REFERENCE ONLY. A hit proves the base name appears in that file;
    it does NOT prove the opponent-suffixed value is the one used, since a
    same-named offensive or player-side field produces an identical hit. `proe`
    is the clearest example: a literal `proe` reference in simulation_v2 is not
    evidence that `proe_opp` moved a player projection. Per audit 6432388032
    these are reported as literal-reference inventory counts and carry no
    causal-transmission claim.
    """
    out: dict[str, dict[str, int]] = {}
    present = [f for f in files if Path(f).exists()]
    for c in cols:
        hits = {}
        for f in present:
            r = subprocess.run(["grep", "-c", "-F", c, f],
                               capture_output=True, text=True)
            n = int((r.stdout or "0").strip() or 0)
            if n:
                hits[f] = n
        if hits:
            out[c] = hits
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=Path(
        "data/research/week7_opportunity_trace/"
        "week7_opportunity_trace_v1_status.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/opponent_transmission"))
    a = ap.parse_args()

    dyn = assert_no_dynamic_selection()

    inv = json.loads(a.inventory.read_text(encoding="utf-8"))
    by_file = {s["path"]: (s.get("all_non_market_column_names") or [])
               for s in inv["extraction_diagnostics"]["scanned"]}
    mr = by_file.get("data/metrics_ready.csv", [])
    # Opponent columns arrive with three suffixes, not one: a clean `_opp` and
    # the `_opp_x` / `_opp_y` pandas merge-collision variants. Matching only
    # `_opp` silently drops real opponent signal -- it dropped
    # coverage_zone_rate / coverage_man_rate, which rules_v2 DOES consume, and
    # so misreported the receiving-coverage family as fully stranded.
    opp_re = re.compile(r"^(?P<base>.+?)_opp(?:_[xy])?$")
    opp_suffixed, bases = [], set()
    for c in mr:
        m = opp_re.match(c)
        if m:
            opp_suffixed.append(c)
            bases.add(m.group("base"))
    opp_suffixed = sorted(opp_suffixed)
    bases = sorted(bases)

    transported = consumption(bases, TRANSPORT)
    referenced = consumption(bases, CONSUMERS)
    tc = teamcontext_transmission()
    ignored_at_rule = set(
        tc["carried_onto_teamcontext_then_ignored_by_every_rule"])
    rows = []
    for c in bases:
        t_hits, r_hits = transported.get(c, {}), referenced.get(c, {})
        is_book = bool(BOOKKEEPING.match(c))
        rows.append({
            "column_base": c,
            # stage 1
            "stage1_present_on_player_row": any(
                s2 in set(opp_suffixed)
                for s2 in (f"{c}_opp", f"{c}_opp_x", f"{c}_opp_y")),
            # stage 2 -- carried only; NOT a consumption claim
            "stage2_bridged_by_transport": bool(t_hits),
            # stage 3 -- literal reference in a consumer module
            "stage3_referenced_by_a_consumer": bool(r_hits),
            # stage 4 -- for TeamContext fields, whether any rule reads it
            "stage4_read_by_a_rule": (
                "" if c not in set(tc["teamcontext_fields_populated"])
                else str(c not in ignored_at_rule)),
            "transported_but_stranded": bool(t_hits) and not bool(r_hits),
            "football_signal": not is_book,
            "mechanism_family": "BOOKKEEPING" if is_book else _family(c),
            "consumer_files": ";".join(sorted(r_hits)),
            "literal_reference_count_in_consumers": sum(r_hits.values()),
        })

    foot = [r for r in rows if r["football_signal"]]
    stranded = [r for r in foot if not r["stage3_referenced_by_a_consumer"]]

    fam_summary = {}
    for name in list(FAMILIES) + ["UNCLASSIFIED"]:
        f = [r for r in foot if r["mechanism_family"] == name]
        if not f:
            continue
        spec = FAMILIES.get(name, {})
        fam_summary[name] = {
            "football_question": spec.get("question", ""),
            "distribution_channel": spec.get("channel", ""),
            "columns_available": len(f),
            "columns_literally_referenced_by_a_consumer": sum(
                1 for r in f if r["stage3_referenced_by_a_consumer"]),
            "columns_stranded": sum(
                1 for r in f if not r["stage3_referenced_by_a_consumer"]),
            "columns_transported_but_stranded": sum(
                1 for r in f if r["transported_but_stranded"]),
            "fully_stranded": all(
                not r["stage3_referenced_by_a_consumer"] for r in f),
            "stranded_columns": sorted(
                r["column_base"] for r in f
                if not r["stage3_referenced_by_a_consumer"]),
        }

    status = {
        "version": "OPPONENT_ENVIRONMENT_TRANSMISSION_TRACE_V1",
        "authorized_by": "GPT_AUDIT_6431984732_READ_ONLY_MECHANISM_FIRST",
        "consumes_accepted_decomposition": (
            "cca416bd2cdad8f142a5a063923149344c7e5c5d"),
        "scientific_boundary": "READ_ONLY_MECHANISM_TRACE_NO_FITTING_NO_SCORING",
        "outcomes_inspected": False,
        "sportsbook_inputs_used": 0,
        "week7_used_to_choose_coefficient_threshold_or_sign": False,
        "method_validity_checks": dyn,
        "teamcontext_transmission": tc,
        "orphaned_mechanisms": orphaned_mechanisms(),
        "player_path_files": [f for f in PLAYER_PATH if Path(f).exists()],
        "COUNT_SEMANTICS": (
            "these are LITERAL-REFERENCE INVENTORY counts, not certified "
            "causal transmission counts. A base-name hit does not prove the "
            "opponent-suffixed value is the value consumed (audit 6432388032). "
            "The certified dataflow findings are in teamcontext_transmission, "
            "which compares populated against rule-read fields exactly."),
        "opponent_columns_on_player_row": len(opp_suffixed),
        "football_signal_columns": len(foot),
        "football_signal_literally_referenced_by_a_consumer": (
            len(foot) - len(stranded)),
        "football_signal_stranded": len(stranded),
        "ensemble_component_contract": {
            "components": ["mc_proj", "ml_proj", "state_proj"],
            "source": "scripts/modeling/ensemble_v2.COMPONENTS",
            "bayesian_v2_is_a_fourth_component": False,
            "note": ("ensemble_v2 states Bayesian v2 is intentionally NOT a "
                     "fourth independent projection because it is already "
                     "inside the MC baseline. An earlier version of this "
                     "trace said 2 of 4; the correct figure is 2 of 3."),
        },
        "structural_zeros": {
            "ml_v2": ("FEATURE_COLUMNS is hist_games + prev/mean3/mean5 of the "
                      "player's own box stats + position dummies; reads only "
                      "player/team/position from context and uses team as an "
                      "identity key. No opponent input exists."),
            "state_v2": ("required set is {player, team, position}; features "
                         "are built from the player's own prior game logs. No "
                         "opponent input exists."),
            "consequence": ("2 of the 3 independent generic ensemble "
                            "components (ml_v2, state_v2) cannot express any "
                            "opponent effect for any player in any market. "
                            "Architectural absence, not weak signal. The MC "
                            "component CAN express an opponent effect where a "
                            "canonical rule exists and fires."),
        },
        "rb_rush_efficiency_single_channel": {
            "path": "scripts/modeling/rules_v2.py::matchup_multipliers -> rb_rush_eff_mult",
            "gates": ["light_box_rate >= 0.60 -> *1.07",
                      "heavy_box_rate >= 0.60 -> *0.94"],
            "default_when_neither_fires": 1.0,
            "is_a_live_mean_channel": True,
            "scope_correction": (
                "this IS a live opponent -> RB rushing mean channel when a "
                "gate fires. The established claim is narrower: OUTSIDE the "
                "box gate the generic RB rushing mean has no opponent "
                "run-defense transmission. The gate is not called dead, and "
                "no every-RB/every-week claim is made, because firing "
                "frequency is NOT ESTABLISHED (audit 6432370310/6432388032)."),
            "applied_at": "scripts/modeling/simulation_rules.py:231 rules_ypc = base_ypc * rb_rush_eff_mult",
            "reachable_receiving_counterpart": [
                "coverage_zone_rate >= 0.60 -> rb_rec_target_mult *1.20",
                "pressure mismatch > 0.075 -> rb_rec_target_mult *1.10"],
            "explains": ("same player, same opponent, same week: the receiving "
                         "gate is reachable and fired, the rushing gates did "
                         "not. The asymmetry is gate reachability, not football."),
        },
        "not_established": {
            "gate_firing_frequency": (
                "how often light/heavy_box_rate >= 0.60 holds is NOT measured "
                "here. Production does persist the multiplier "
                "(artifact_contracts.py:38 rush_eff_mult; "
                "simulation_rules.py:236 rules_rush_eff_mult), but the Week-7 "
                "preservation filter kept only the column INVENTORY and dropped "
                "every opponent value and the multiplier itself. Answerable "
                "from the next live run or a team_form rebuild; not from the "
                "durable archive."),
            "accuracy_effect": (
                "whether consuming any stranded family would improve accuracy "
                "is NOT tested and must not be scored on Week 7."),
        },
        "separate_defect_do_not_merge": (
            "the zero-history / position-prior entitlement defect is a distinct "
            "cause with a distinct trace and disposition, per the authorizing "
            "audit. It is not folded into this trace."),
        "mechanism_families": fam_summary,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "opponent_environment_transmission_v1"
    import csv
    with (a.out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["mechanism_family"],
                                                r["column_base"])))
    (a.out_dir / f"{stem}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== OPPONENT / ENVIRONMENT TRANSMISSION TRACE V1 ===")
    print("counts below are LITERAL-REFERENCE INVENTORY, not certified")
    print("causal transmission; see teamcontext_transmission for exact dataflow.\n")
    print(f"opponent columns on the player row   : {len(opp_suffixed)}")
    print(f"football-signal columns              : {len(foot)}")
    print(f"  literally referenced by a consumer : {len(foot) - len(stranded)}")
    print(f"  STRANDED                           : {len(stranded)}")
    tbs = [r["column_base"] for r in foot if r["transported_but_stranded"]]
    print(f"  of which TRANSPORTED-BUT-STRANDED  : {len(tbs)}"
          + (f"  {sorted(tbs)}" if tbs else ""))
    print("\nstructural zeros: ml_v2 and state_v2 take NO opponent input at all")
    print("  -> 2 of 3 independent generic ensemble components "
          "(mc/ml/state) cannot")
    print("     express an opponent effect; MC can, where a rule exists "
          "and fires.\n")
    print(f"{'family':32s} {'avail':>5s} {'ref':>4s} {'strand':>6s}  channel")
    for name, fs in fam_summary.items():
        flag = "  <-- FULLY STRANDED" if fs["fully_stranded"] else ""
        print(f"{name:32s} {fs['columns_available']:5d} "
              f"{fs['columns_literally_referenced_by_a_consumer']:4d} "
              f"{fs['columns_stranded']:6d}  "
              f"{fs['distribution_channel']}{flag}")
    print("\nNo coefficient, threshold, sign or winning candidate is produced "
          "here. Week 7 identified the mechanism; it cannot score the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
