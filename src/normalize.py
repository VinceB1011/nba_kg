import pandas as pd
import re
import unicodedata
from ftfy import fix_text

TEAM_ABBREVIATION_MAP = {
    # ESPN -> NBA canonical
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def normalize_team_abbreviation(value):

    if pd.isna(value):
        return value

    value = str(value).strip().upper()
    return TEAM_ABBREVIATION_MAP.get(value, value)

def normalize_player_name(name: str) -> str | None:
    if pd.isna(name):
        return None

    # repair broken unicode / mojibake first
    name = fix_text(str(name))

    name = name.strip().lower()

    # remove accents / diacritics
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
    )

    name = name.replace(".", "")
    name = name.replace("'", "")
    name = name.replace("-", " ")

    name = re.sub(r"\s+", " ", name).strip()

    return name


# Suffixe wie "II" oder "Jr." sind bei ESPN oft vorhanden und bei
# NBA/shotdetail nicht (z. B. "Darius Brown II" vs. "Darius Brown").
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def strip_name_suffix(name: str | None) -> str | None:
    """
    "darius brown ii" -> "darius brown"

    Der Vorname bleibt immer erhalten, damit aus einem
    zweiteiligen Namen kein einteiliger wird.
    """
    if name is None:
        return None

    parts = name.split()

    while len(parts) > 2 and parts[-1] in NAME_SUFFIXES:
        parts.pop()

    return " ".join(parts)


def add_match_name(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Ergaenzt die Join-Spalte `match_name` und stellt sicher, dass das
    Entfernen der Suffixe keine zwei echten Spieler zusammenwirft.
    """
    df = df.copy()
    df["match_name"] = df["normalized_name"].apply(strip_name_suffix)

    collisions = df[df.duplicated("match_name", keep=False)]

    if not collisions.empty:
        raise ValueError(
            f"Suffix-Entfernung erzeugt mehrdeutige Namen auf Seite '{side}':\n"
            f"{collisions.sort_values('match_name').to_string(index=False)}"
        )

    return df

def build_team_map(
    schedule: pd.DataFrame,
    matchups: pd.DataFrame,
) -> pd.DataFrame:

    espn_home = schedule[
        ["home_id", "home_abbreviation", "home_display_name"]
    ].rename(
        columns={
            "home_id": "espn_team_id",
            "home_abbreviation": "team_abbreviation",
            "home_display_name": "espn_team_name",
        }
    )

    espn_away = schedule[
        ["away_id", "away_abbreviation", "away_display_name"]
    ].rename(
        columns={
            "away_id": "espn_team_id",
            "away_abbreviation": "team_abbreviation",
            "away_display_name": "espn_team_name",
        }
    )

    espn_teams = pd.concat(
        [espn_home, espn_away],
        ignore_index=True,
    )

    espn_teams["team_abbreviation"] = (
        espn_teams["team_abbreviation"]
        .apply(normalize_team_abbreviation)
    )

    espn_teams = (
        espn_teams
        .drop_duplicates()
        .reset_index(drop=True)
    )

    nba_teams = (
        matchups[
            ["team_id", "team_tricode", "team_name", "team_city"]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "team_id": "nba_team_id",
                "team_tricode": "team_abbreviation",
                "team_name": "nba_team_name",
            }
        )
    )

    nba_teams["team_abbreviation"] = (
        nba_teams["team_abbreviation"]
        .apply(normalize_team_abbreviation)
    )

    return espn_teams.merge(
        nba_teams,
        on="team_abbreviation",
        how="outer",
        indicator=True,
    )

def build_game_map(
    schedule: pd.DataFrame,
    shots: pd.DataFrame,
) -> pd.DataFrame:

    # NBA archive file is regular season ("rg"),
    # therefore compare it with ESPN regular season only.
    # Variante A: nur tatsaechlich gespielte Regular-Season-Games.
    # Verschobene Spiele stehen als Platzhalter mit Score 0 im Schedule
    # und tauchen spaeter unter dem Makeup-Date erneut auf.
    espn_games = schedule[
        (schedule["season_type"] == 2)
        & (schedule["status_type_completed"])
    ][
        [
            "game_id",
            "game_date",
            "home_abbreviation",
            "away_abbreviation",
            "season",
            "season_type",
        ]
    ].copy()

    espn_games = espn_games.rename(
        columns={
            "game_id": "espn_game_id",
        }
    )

    espn_games["home_abbreviation"] = (
        espn_games["home_abbreviation"]
        .apply(normalize_team_abbreviation)
    )

    espn_games["away_abbreviation"] = (
        espn_games["away_abbreviation"]
        .apply(normalize_team_abbreviation)
    )

    espn_games["game_date"] = (
        pd.to_datetime(espn_games["game_date"])
        .dt.strftime("%Y%m%d")
    )

    nba_games = (
        shots[
            [
                "GAME_ID",
                "GAME_DATE",
                "HTM",
                "VTM",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "GAME_ID": "nba_game_id",
                "GAME_DATE": "game_date",
                "HTM": "home_abbreviation",
                "VTM": "away_abbreviation",
            }
        )
    )

    nba_games["home_abbreviation"] = (
        nba_games["home_abbreviation"]
        .apply(normalize_team_abbreviation)
    )

    nba_games["away_abbreviation"] = (
        nba_games["away_abbreviation"]
        .apply(normalize_team_abbreviation)
    )

    nba_games["game_date"] = (
        nba_games["game_date"]
        .astype(str)
    )

    return espn_games.merge(
        nba_games,
        on=[
            "game_date",
            "home_abbreviation",
            "away_abbreviation",
        ],
        how="left",
        indicator=True,
    )

def build_nba_players(
    shots: pd.DataFrame,
    matchups: pd.DataFrame,
) -> pd.DataFrame:

    shot_players = (
        shots[
            ["PLAYER_ID", "PLAYER_NAME"]
        ]
        .rename(
            columns={
                "PLAYER_ID": "nba_player_id",
                "PLAYER_NAME": "nba_player_name",
            }
        )
        .dropna()
    )

    matchup_a = matchups[
        ["person_id", "first_name", "family_name"]
    ].copy()

    matchup_a["nba_player_name"] = (
        matchup_a["first_name"].fillna("")
        + " "
        + matchup_a["family_name"].fillna("")
    ).str.strip()

    matchup_a = matchup_a.rename(
        columns={"person_id": "nba_player_id"}
    )[
        ["nba_player_id", "nba_player_name"]
    ]

    matchup_b = matchups[
        [
            "matchups_person_id",
            "matchups_first_name",
            "matchups_family_name",
        ]
    ].copy()

    matchup_b["nba_player_name"] = (
        matchup_b["matchups_first_name"].fillna("")
        + " "
        + matchup_b["matchups_family_name"].fillna("")
    ).str.strip()

    matchup_b = matchup_b.rename(
        columns={
            "matchups_person_id": "nba_player_id"
        }
    )[
        ["nba_player_id", "nba_player_name"]
    ]

    nba_players = pd.concat(
        [
            shot_players,
            matchup_a,
            matchup_b,
        ],
        ignore_index=True,
    )

    nba_players = nba_players.dropna(
        subset=["nba_player_id", "nba_player_name"]
    )

    nba_players = nba_players[
        nba_players["nba_player_id"] != 0
    ]

    nba_players["normalized_name"] = (
        nba_players["nba_player_name"]
        .apply(normalize_player_name)
    )

    return (
        nba_players
        .drop_duplicates(subset=["nba_player_id"])
        .reset_index(drop=True)
    )

def build_espn_players(
    player_boxscores: pd.DataFrame,
) -> pd.DataFrame:

    players = (
        player_boxscores[
            [
                "athlete_id",
                "athlete_display_name",
            ]
        ]
        .dropna(subset=["athlete_id", "athlete_display_name"])
        .drop_duplicates()
        .rename(
            columns={
                "athlete_id": "espn_player_id",
                "athlete_display_name": "espn_player_name",
            }
        )
    )

    players["normalized_name"] = (
        players["espn_player_name"]
        .apply(normalize_player_name)
    )

    return players

def build_player_map(
    player_boxscores,
    shots,
    matchups,
):

    espn_players = build_espn_players(
        player_boxscores
    )

    nba_players = build_nba_players(
        shots,
        matchups,
    )

    espn_players = add_match_name(espn_players, side="espn")
    nba_players = add_match_name(nba_players, side="nba")

    return espn_players.merge(
        nba_players,
        on="match_name",
        how="outer",
        suffixes=("_espn", "_nba"),
        indicator=True,
    )

