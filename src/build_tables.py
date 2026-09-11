"""
Baut die kanonischen Tabellen einer Saison aus den Rohquellen.

Canonical IDs sind durchgaengig die ESPN-IDs:
    player_id = ESPN athlete_id
    team_id   = ESPN team_id
    game_id   = ESPN game_id

NBA-IDs werden nur als zusaetzliche Mapping-Spalten mitgefuehrt.
"""

from pathlib import Path

import pandas as pd

from normalize import build_team_map, build_game_map, build_player_map

PROCESSED_DIR = Path("data/processed")


def _to_int64(series: pd.Series) -> pd.Series:
    """ESPN-IDs kommen teils als float an ('123.0') - vereinheitlichen."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def build_teams(team_map: pd.DataFrame) -> pd.DataFrame:
    """
    Die 30 echten NBA-Teams. All-Star-Teams (STARS/STRIPES/WORLD)
    haben kein NBA-Gegenstueck und fallen ueber den Merge-Indikator raus.
    """
    teams = team_map[team_map["_merge"] == "both"].copy()

    teams["team_id"] = _to_int64(teams["espn_team_id"])
    teams["nba_team_id"] = _to_int64(teams["nba_team_id"])

    teams = teams.rename(
        columns={
            "team_abbreviation": "abbreviation",
            "espn_team_name": "name",
            "team_city": "city",
        }
    )

    return (
        teams[["team_id", "nba_team_id", "name", "abbreviation", "city"]]
        .sort_values("team_id")
        .reset_index(drop=True)
    )


def build_games(
    game_map: pd.DataFrame,
    schedule: pd.DataFrame,
    season_start_year: int,
) -> pd.DataFrame:
    """
    Die 1230 Regular-Season-Games, fuer die auch NBA-Daten existieren.

    Nicht enthalten: All-Star-Spiele und das NBA-Cup-Finale
    (siehe docs/nba_kg_documentation.md, Abschnitt 11).

    `season` ist immer das Startjahr der Saison: 2025 steht fuer 2025/26.
    Spiele von Januar bis April tragen also das Vorjahr. ESPN liefert in
    seiner eigenen `season`-Spalte stattdessen das Endjahr (2026) - die
    wird hier bewusst verworfen, damit im ganzen Projekt eine einzige
    Konvention gilt.
    """
    games = game_map[game_map["nba_game_id"].notna()].copy()

    scores = schedule[
        ["game_id", "home_id", "away_id", "home_score", "away_score"]
    ].copy()
    scores["game_id"] = _to_int64(scores["game_id"])

    games["game_id"] = _to_int64(games["espn_game_id"])
    games = games.merge(scores, on="game_id", how="left")

    # `game_date` ist US-Lokalzeit (Tip-off-Datum am Spielort), genau wie
    # GAME_DATE auf der NBA-Seite. Die UTC-Spalte `date` waere ein anderer
    # Tag: ein Abendspiel um 19:30 ET liegt in UTC bereits am Folgetag.
    games["date"] = pd.to_datetime(games["game_date"], format="%Y%m%d")
    games["season"] = season_start_year
    games["home_team_id"] = _to_int64(games["home_id"])
    games["away_team_id"] = _to_int64(games["away_id"])

    return (
        games[
            [
                "game_id",
                "nba_game_id",
                "date",
                "season",
                "season_type",
                "home_team_id",
                "away_team_id",
                "home_score",
                "away_score",
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_players(
    rosters: pd.DataFrame,
    player_boxscores: pd.DataFrame,
    player_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Union aus Roster und Boxscores.

    Der Roster ist ein Snapshot vom Saisonende: Spieler, die vorher
    getradet oder gewaived wurden, fehlen dort. Umgekehrt stehen
    saisonlang Verletzte im Roster ohne je gespielt zu haben.
    Beide Gruppen sollen einen Node bekommen.
    """
    roster = rosters.copy()
    roster["player_id"] = _to_int64(roster["athlete_id"])

    roster = roster[
        [
            "player_id",
            "display_name",
            "first_name",
            "last_name",
            "position_abbreviation",
            "height",
            "weight",
            "date_of_birth",
            "birth_place_country",
            "experience_years",
        ]
    ].rename(
        columns={
            "display_name": "name",
            "position_abbreviation": "position",
        }
    )

    roster = roster.drop_duplicates(subset=["player_id"])

    # Fallback-Attribute fuer Spieler ohne Roster-Eintrag.
    # Die Boxscore-Position stimmt bei allen gemeinsamen Spielern
    # mit der Roster-Position ueberein.
    from_box = (
        player_boxscores.assign(player_id=lambda d: _to_int64(d["athlete_id"]))
        .sort_values("game_date")
        .groupby("player_id")
        .agg(
            box_name=("athlete_display_name", "last"),
            box_position=("athlete_position_abbreviation", "last"),
        )
        .reset_index()
    )

    players = from_box.merge(roster, on="player_id", how="outer")

    players["in_end_of_season_roster"] = players["name"].notna()
    players["name"] = players["name"].fillna(players["box_name"])
    players["position"] = players["position"].fillna(players["box_position"])

    # NBA-ID aus dem Namensmapping anhaengen
    mapping = player_map[player_map["_merge"] == "both"][
        ["espn_player_id", "nba_player_id"]
    ].copy()
    mapping["player_id"] = _to_int64(mapping["espn_player_id"])
    mapping["nba_player_id"] = _to_int64(mapping["nba_player_id"])

    players = players.merge(
        mapping[["player_id", "nba_player_id"]],
        on="player_id",
        how="left",
    )

    return (
        players[
            [
                "player_id",
                "nba_player_id",
                "name",
                "first_name",
                "last_name",
                "position",
                "height",
                "weight",
                "date_of_birth",
                "birth_place_country",
                "experience_years",
                "in_end_of_season_roster",
            ]
        ]
        .sort_values("player_id")
        .reset_index(drop=True)
    )


def save(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"[saved] {path}  {df.shape}")


# Spaltennamen im ESPN-Boxscore -> kanonische Namen
BOX_STAT_COLUMNS = {
    "minutes": "minutes",
    "points": "points",
    "rebounds": "rebounds",
    "offensive_rebounds": "offensive_rebounds",
    "defensive_rebounds": "defensive_rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "fouls": "fouls",
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m",
    "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
    "plus_minus": "plus_minus",
}


def _box_with_keys(player_boxscores: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Boxscore auf die Games der kanonischen Tabelle einschraenken und
    die drei Fremdschluessel auf Int64 normalisieren.

    Der Boxscore enthaelt auch Playoffs (season_type 3) und Play-In (5);
    `games` enthaelt nur die Regular Season, deshalb der Filter.
    """
    box = player_boxscores.copy()

    box["game_id"] = _to_int64(box["game_id"])
    box["player_id"] = _to_int64(box["athlete_id"])
    box["team_id"] = _to_int64(box["team_id"])

    return box[box["game_id"].isin(set(games["game_id"]))]


def _drop_rows_without_player(box: pd.DataFrame) -> pd.DataFrame:
    """
    ESPN liefert fuer Chicago 22 namenlose Platzhalter-Zeilen ohne
    athlete_id (alle Stats 0, reason COACH'S DECISION). Ohne player_id
    kann daraus weder eine Statline noch eine Absence werden.
    """
    missing = box["player_id"].isna()

    if missing.any():
        print(f"[warn] {missing.sum()} Boxscore-Zeilen ohne athlete_id verworfen")

    return box[~missing]


def build_player_game_stats(
    player_boxscores: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Eine Zeile pro Spieler und Game - aber nur, wenn er gespielt hat.

    Nicht-Eintraege stehen in `player_game_absences`. Damit bleibt diese
    Tabelle eine reine Statline-Tabelle ohne NaN-Zeilen.

    `team_id` steht bewusst drin: bei einem Trade mitten in der Saison
    ist die Team-Zugehoerigkeit eine Eigenschaft der Player-Game-Kombination,
    nicht des Spielers.
    """
    box = _drop_rows_without_player(_box_with_keys(player_boxscores, games))

    # `did_not_play` ist in der Quelle lueckenhaft: es gibt Zeilen mit
    # did_not_play=False, aber active=False und minutes=NaN.
    # `minutes` ist das verlaessliche Kriterium - wer gespielt hat,
    # hat eine Minutenzahl.
    played = box[box["minutes"].notna()].copy()

    played = played.rename(columns=BOX_STAT_COLUMNS)
    played["starter"] = played["starter"].fillna(False).astype(bool)

    columns = ["game_id", "player_id", "team_id", "starter"] + list(
        BOX_STAT_COLUMNS.values()
    )

    return (
        played[columns]
        .sort_values(["game_id", "team_id", "player_id"])
        .reset_index(drop=True)
    )


def build_player_game_absences(
    player_boxscores: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Spieler, die im Kader standen, aber nicht gespielt haben.

    `reason` unterscheidet die analytisch wichtigen Faelle:
    COACH'S DECISION vs. Verletzung vs. REST vs. Sperre.
    """
    box = _drop_rows_without_player(_box_with_keys(player_boxscores, games))
    absent = box[box["minutes"].isna()].copy()

    absent["reason"] = absent["reason"].fillna("UNKNOWN")

    return (
        absent[["game_id", "player_id", "team_id", "reason"]]
        .sort_values(["game_id", "team_id", "player_id"])
        .reset_index(drop=True)
    )
