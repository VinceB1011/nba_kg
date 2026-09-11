"""
Baut alle kanonischen Tabellen einer Saison und validiert sie.

Aufruf:  python src/build_all.py
"""

import pandas as pd

from load_data import load_all
from normalize import build_team_map, build_game_map, build_player_map
from build_tables import (
    build_teams,
    build_games,
    build_players,
    build_player_game_stats,
    build_player_game_absences,
    save,
)

SEASON = 2025


def validate(tables: dict[str, pd.DataFrame]) -> None:
    teams = tables["teams"]
    players = tables["players"]
    games = tables["games"]
    stats = tables["player_game_stats"]
    absences = tables["player_game_absences"]

    # Primaerschluessel
    assert teams["team_id"].is_unique
    assert players["player_id"].is_unique
    assert games["game_id"].is_unique

    assert len(teams) == 30, f"erwartet 30 Teams, sind {len(teams)}"
    assert len(games) == 1230, f"erwartet 1230 Games, sind {len(games)}"

    # Jedes Team spielt 82 Regular-Season-Spiele
    per_team = pd.concat(
        [games["home_team_id"], games["away_team_id"]]
    ).value_counts()
    assert set(per_team.unique()) == {82}, f"Spiele pro Team: {per_team.unique()}"

    assert games[["home_score", "away_score"]].notna().all().all()

    # Konvention: season = Startjahr. Spiele ab Januar liegen im Folgejahr.
    assert set(games["season"].unique()) == {SEASON}, games["season"].unique()
    assert games["date"].dt.year.isin([SEASON, SEASON + 1]).all()

    # Fremdschluessel
    for name in ("player_game_stats", "player_game_absences"):
        df = tables[name]
        assert df["game_id"].isin(games["game_id"]).all(), f"{name}: game_id"
        assert df["player_id"].isin(players["player_id"]).all(), f"{name}: player_id"
        assert df["team_id"].isin(teams["team_id"]).all(), f"{name}: team_id"
        assert not df.duplicated(["game_id", "player_id"]).any(), f"{name}: dupes"

    # Ein Spieler steht pro Game entweder in stats oder in absences
    overlap = stats.merge(absences, on=["game_id", "player_id"])
    assert overlap.empty, f"{len(overlap)} Zeilen in beiden Tabellen"

    assert stats.notna().all().all(), "player_game_stats enthaelt NULLs"

    # Genau 10 Starter pro Game (5 pro Team)
    starters = stats[stats["starter"]].groupby("game_id").size()
    assert set(starters.unique()) == {10}, f"Starter pro Game: {starters.unique()}"

    print("Alle Validierungen bestanden.")


def main() -> None:
    raw = load_all(SEASON)

    team_map = build_team_map(raw["schedule"], raw["matchups"])
    game_map = build_game_map(raw["schedule"], raw["shots"])
    player_map = build_player_map(
        player_boxscores=raw["player_box"],
        shots=raw["shots"],
        matchups=raw["matchups"],
    )

    games = build_games(game_map, raw["schedule"], SEASON)

    tables = {
        "teams": build_teams(team_map),
        "players": build_players(raw["rosters"], raw["player_box"], player_map),
        "games": games,
        "player_game_stats": build_player_game_stats(raw["player_box"], games),
        "player_game_absences": build_player_game_absences(raw["player_box"], games),
    }

    for name, df in tables.items():
        save(df, name)

    # Mappingtabellen ebenfalls persistieren
    save(team_map, "team_id_map")
    save(game_map, "game_id_map")
    save(player_map, "player_id_map")

    print()
    validate(tables)


if __name__ == "__main__":
    main()
