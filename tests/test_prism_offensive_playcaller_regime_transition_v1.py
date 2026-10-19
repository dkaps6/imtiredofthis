import pandas as pd

from scripts.research import audit_prism_offensive_playcaller_regime_transition_v1 as mod


def test_historical_caller_state_detects_week1_and_midseason_change(monkeypatch):
    def fake_caller(season, week, team):
        if season == 2023:
            return "Old Caller"
        if season == 2024:
            if week < 4:
                return "New Caller"
            return "Second Caller"
        if season == 2025:
            return "Second Caller"
        return ""
    monkeypatch.setattr(mod.m68, "caller_for", fake_caller)

    tw = pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA"},
        {"season":2024,"week":2,"team":"AAA"},
        {"season":2024,"week":3,"team":"AAA"},
        {"season":2024,"week":4,"team":"AAA"},
    ])
    out,audit = mod.historical_caller_state(tw)
    assert audit["chronology_violations"] == 0
    w1 = out.loc[out["week"].eq(1)].iloc[0]
    assert bool(w1["playcaller_changed"])
    assert bool(w1["offseason_week1_transition"])
    w4 = out.loc[out["week"].eq(4)].iloc[0]
    assert bool(w4["playcaller_changed"])
    assert bool(w4["inseason_transition"])
    assert w4["previous_game_playcaller"] == "New Caller"
    assert w4["current_playcaller"] == "Second Caller"


def test_attach_players_uses_only_strictly_prior_history():
    health = pd.DataFrame([
        {"season":2023,"week":17,"team":"AAA","stable_id":"p1","family":"WR","positive_participation":True},
        {"season":2023,"week":18,"team":"AAA","stable_id":"p1","family":"WR","positive_participation":True},
        {"season":2024,"week":1,"team":"AAA","stable_id":"p1","family":"WR","positive_participation":True},
        {"season":2024,"week":2,"team":"AAA","stable_id":"p1","family":"WR","positive_participation":True},
        {"season":2024,"week":3,"team":"AAA","stable_id":"p1","family":"WR","positive_participation":True},
    ])
    callers = pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA","current_playcaller":"C2",
         "previous_game_playcaller":"C1","previous_game_label":"2023_W18",
         "caller_transition_state":"PLAYCALLER_REGIME_CHANGED","playcaller_changed":True,
         "current_caller_effective_start_week":1,"offseason_week1_transition":True,
         "inseason_transition":False},
        {"season":2024,"week":2,"team":"AAA","current_playcaller":"C2",
         "previous_game_playcaller":"C2","previous_game_label":"2024_W1",
         "caller_transition_state":"PLAYCALLER_REGIME_CONTINUED","playcaller_changed":False,
         "current_caller_effective_start_week":1,"offseason_week1_transition":False,
         "inseason_transition":False},
        {"season":2024,"week":3,"team":"AAA","current_playcaller":"C3",
         "previous_game_playcaller":"C2","previous_game_label":"2024_W2",
         "caller_transition_state":"PLAYCALLER_REGIME_CHANGED","playcaller_changed":True,
         "current_caller_effective_start_week":3,"offseason_week1_transition":False,
         "inseason_transition":True},
    ])
    current = health.loc[health["season"].eq(2024)].copy()
    out,audit = mod.attach_players(health,current.merge(
        callers,on=["season","week","team"],how="left"
    )[callers.columns])
    assert audit["strictly_prior_history_chronology_violations"] == 0
    w1 = out.loc[out["week"].eq(1)].iloc[0]
    assert w1["same_player_same_team_prior_positive_rows"] == 2
    assert not bool(w1["prior_positive_support_ge3"])
    w3 = out.loc[out["week"].eq(3)].iloc[0]
    assert w3["same_player_same_team_prior_positive_rows"] == 4
    assert bool(w3["prior_positive_support_ge3"])


def test_prospective_snapshot_requires_32_week1_teams_and_override():
    teams=[f"T{i:02d}" for i in range(32)]
    rows=[
        {"season":2026,"team":t,"effective_week":1,"playcaller":f"C{i}",
         "source_type":"season_opening","source_url":"https://example.com"}
        for i,t in enumerate(teams)
    ]
    rows.append({
        "season":2026,"team":"T00","effective_week":5,"playcaller":"New Caller",
        "source_type":"midseason_override","source_url":"https://example.com/override"
    })
    out=mod.prospective_parity(pd.DataFrame(rows))
    assert out["prospective_parity_ready"]
    assert out["week1_teams"] == 32
    assert out["override_rows"] == 1


def test_caller_state_does_not_require_player_outcomes(monkeypatch):
    health = pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA","stable_id":"p1","family":"RB",
         "positive_participation":False},
        {"season":2024,"week":2,"team":"AAA","stable_id":"p1","family":"RB",
         "positive_participation":False},
    ])
    def fake_caller(season, week, team):
        return "C1" if season == 2023 else "C2"
    monkeypatch.setattr(mod.m68, "caller_for", fake_caller)
    tw=mod.team_weeks_from_health(health)
    callers,_=mod.historical_caller_state(tw)
    assert callers["current_playcaller"].eq("C2").all()
    assert callers.loc[callers["week"].eq(1),"playcaller_changed"].all()
