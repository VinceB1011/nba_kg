from pathlib import Path
import pandas as pd
import requests
from normalize import build_team_map, build_game_map

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def season_to_sdv_year(start_year: int) -> int:
    """
    2025 -> 2026 for the 2025/26 NBA season.
    """
    return start_year + 1


def load_archive_parquet(
    dataset: str,
    season_start_year: int,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load one dataset from the cdechoch/shufinskiy NBA archive.

    Examples:
        dataset="nbastatsv3"
        dataset="matchups"
        dataset="shotdetail"
        dataset="pbpstats"
    """
    url = (
        "https://huggingface.co/datasets/"
        "cdechoch/nba-data-archive/resolve/main/"
        f"per_season/{dataset}/{season_start_year}.parquet"
    )

    cache_path = DATA_DIR / f"{dataset}_{season_start_year}.parquet"

    if cache and cache_path.exists():
        print(f"[cache] {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[download] {dataset}: {url}")

    df = pd.read_parquet(url)

    if cache:
        df.to_parquet(cache_path, index=False)

    return df


def load_pbp(season_start_year: int) -> pd.DataFrame:
    return load_archive_parquet(
        "nbastatsv3",
        season_start_year,
    )


def load_matchups(season_start_year: int) -> pd.DataFrame:
    return load_archive_parquet(
        "matchups",
        season_start_year,
    )


def load_shots(season_start_year: int) -> pd.DataFrame:
    return load_archive_parquet(
        "shotdetail",
        season_start_year,
    )


def inspect(name: str, df: pd.DataFrame, rows: int = 3) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nExample rows:")
    print(df.head(rows).to_string())


if __name__ == "__main__":
    # 2025 means season 2025/26 in the shufinskiy/cdechoch archive.
    season = 2025

    pbp = load_pbp(season)
    inspect("PLAY BY PLAY", pbp)

    matchups = load_matchups(season)
    inspect("MATCHUPS", matchups)

    shots = load_shots(season)
    inspect("SHOTS", shots)

print("\nACTION TYPES")
print(pbp["actionType"].value_counts().to_string())

print("\nSUBSTITUTIONS")
subs = pbp[
    pbp["actionType"]
    .astype(str)
    .str.lower()
    .str.contains("sub")
]

print(subs.head(15).to_string())

def load_sdv_parquet(
    release_tag: str,
    filename: str,
    cache_name: str,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load a SportsDataverse NBA parquet release from GitHub.
    """

    url = (
        "https://github.com/sportsdataverse/"
        "sportsdataverse-data/releases/download/"
        f"{release_tag}/{filename}"
    )

    cache_path = DATA_DIR / cache_name

    if cache and cache_path.exists():
        print(f"[cache] {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[download] {url}")

    df = pd.read_parquet(url)

    if cache:
        df.to_parquet(cache_path, index=False)

    return df


def load_schedule(season_start_year: int) -> pd.DataFrame:
    """
    season_start_year=2025 means NBA season 2025/26.

    SportsDataverse uses 2026 for the 2025/26 season.
    """
    sdv_year = season_start_year + 1

    return load_sdv_parquet(
        release_tag="espn_nba_schedules",
        filename=f"nba_schedule_{sdv_year}.parquet",
        cache_name=f"sdv_schedule_{season_start_year}.parquet",
    )


def load_player_boxscores(season_start_year: int) -> pd.DataFrame:
    sdv_year = season_start_year + 1

    return load_sdv_parquet(
        release_tag="espn_nba_player_boxscores",
        filename=f"player_box_{sdv_year}.parquet",
        cache_name=f"sdv_player_box_{season_start_year}.parquet",
    )


def load_rosters(season_start_year: int) -> pd.DataFrame:
    sdv_year = season_start_year + 1

    return load_sdv_parquet(
        release_tag="espn_nba_rosters",
        filename=f"rosters_{sdv_year}.parquet",
        cache_name=f"sdv_rosters_{season_start_year}.parquet",
    )

if __name__ == "__main__":
    season = 2025

    # Existing archive data
    pbp = load_pbp(season)
    matchups = load_matchups(season)
    shots = load_shots(season)

    # New SportsDataverse data
    schedule = load_schedule(season)
    inspect("SCHEDULE", schedule)

    player_box = load_player_boxscores(season)
    inspect("PLAYER BOXSCORES", player_box)

    rosters = load_rosters(season)
    inspect("ROSTERS", rosters)

    team_map = build_team_map(schedule, matchups)

    print("\nTEAM MAP")
    print(team_map["_merge"].value_counts())

    print("\nUNMATCHED TEAMS")
    print(
        team_map[
            team_map["_merge"] != "both"
        ].to_string(index=False)
    )


    game_map = build_game_map(schedule, shots)

    print("\nGAME MAP")
    print(game_map["_merge"].value_counts())

    matched = game_map["nba_game_id"].notna().sum()
    total = len(game_map)

    print(f"\nMatched games: {matched}/{total}")
    print(f"Coverage: {matched / total:.2%}")

    print("\nUNMATCHED REGULAR-SEASON GAMES")
    print(
        game_map[
            game_map["nba_game_id"].isna()
        ][
            [
                "espn_game_id",
                "game_date",
                "home_abbreviation",
                "away_abbreviation",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    unmatched_ids = game_map.loc[
    game_map["nba_game_id"].isna(),
    "espn_game_id",
]

unmatched_details = schedule[
    schedule["game_id"].isin(unmatched_ids)
][
    [
        "game_id",
        "game_date",
        "home_abbreviation",
        "away_abbreviation",
        "home_score",
        "away_score",
        "status_type_name",
        "status_type_state",
        "status_type_completed",
        "status_type_description",
        "status_type_detail",
        "notes_type",
        "notes_headline",
    ]
]

print(unmatched_details.to_string(index=False))