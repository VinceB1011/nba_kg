"""
Explorationsskript: prueft die drei ID-Mappings einer Saison.

Kein Teil der Pipeline - nur zum Nachvollziehen der Entity Resolution.
Aufruf:  python src/check_mappings.py
"""

import pandas as pd

from load_data import load_all
from normalize import build_team_map, build_game_map, build_player_map

SEASON = 2025


def report_teams(team_map: pd.DataFrame) -> None:
    print("\n=== TEAMS ===")
    print(team_map["_merge"].value_counts().to_string())

    # All-Star-Auswahlen sind durch den Regular-Season-Filter schon raus.
    unmatched = team_map[team_map["_merge"] != "both"]

    if unmatched.empty:
        print("Beide Quellen kennen dieselben 30 Teams.")
    else:
        print(f"\nNicht gematcht ({len(unmatched)}):")
        print(unmatched[["espn_team_id", "team_abbreviation", "espn_team_name"]].to_string(index=False))


def report_games(game_map: pd.DataFrame) -> None:
    print("\n=== GAMES ===")

    matched = game_map["nba_game_id"].notna().sum()
    total = len(game_map)
    print(f"Gematcht: {matched}/{total} ({matched / total:.2%})")

    assert not game_map["espn_game_id"].duplicated().any(), "espn_game_id nicht eindeutig"

    # All-Star und NBA-Cup-Finale werden inzwischen schon in
    # filter_regular_season() entfernt - hier sollte nichts mehr uebrig sein.
    unmatched = game_map[game_map["nba_game_id"].isna()]

    if unmatched.empty:
        print("Alle Regular-Season-Spiele haben ein NBA-Gegenstueck.")
    else:
        print(f"\nNicht gematcht ({len(unmatched)}):")
        print(
            unmatched[
                ["espn_game_id", "game_date", "home_abbreviation", "away_abbreviation"]
            ].to_string(index=False)
        )


def report_players(player_map: pd.DataFrame, player_box: pd.DataFrame) -> None:
    print("\n=== PLAYERS ===")
    print(player_map["_merge"].value_counts().to_string())

    both = player_map[player_map["_merge"] == "both"]

    assert not both.groupby("espn_player_id")["nba_player_id"].nunique().gt(1).any()
    assert not both.groupby("nba_player_id")["espn_player_id"].nunique().gt(1).any()
    print("Keine 1:n-Kollisionen in beide Richtungen.")

    unmatched = player_map[player_map["_merge"] == "left_only"]

    # Ein unmatchter Spieler ist nur dann ein echtes Problem, wenn er
    # ueberhaupt gespielt hat - reine DNP-Eintraege koennen in
    # shotdetail/matchups gar nicht vorkommen.
    minutes = (
        player_box[player_box["athlete_id"].isin(unmatched["espn_player_id"])]
        .groupby("athlete_display_name")["minutes"]
        .sum()
    )

    print(f"\nNicht gematchte ESPN-Spieler: {len(unmatched)}")
    print(minutes.to_string())

    played = minutes[minutes > 0]
    if played.empty:
        print("\nAlle davon haben 0 Minuten gespielt -> unkritisch.")
    else:
        print(f"\nACHTUNG: {len(played)} davon haben gespielt:")
        print(played.to_string())


if __name__ == "__main__":
    raw = load_all(SEASON)

    report_teams(build_team_map(raw["schedule"], raw["matchups"]))
    report_games(build_game_map(raw["schedule"], raw["shots"]))
    report_players(
        build_player_map(
            player_boxscores=raw["player_box"],
            shots=raw["shots"],
            matchups=raw["matchups"],
        ),
        raw["player_box"],
    )
