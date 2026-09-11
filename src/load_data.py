from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def season_to_sdv_year(start_year: int) -> int:
    """
    Projektkonvention: eine Saison wird ueber ihr Startjahr benannt.
    2025 bedeutet 2025/26.

    SportsDataverse benennt dieselbe Saison ueber das Endjahr,
    das cdechoch/shufinskiy-Archiv ueber das Startjahr.

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
    sdv_year = season_to_sdv_year(season_start_year)

    return load_sdv_parquet(
        release_tag="espn_nba_schedules",
        filename=f"nba_schedule_{sdv_year}.parquet",
        cache_name=f"sdv_schedule_{season_start_year}.parquet",
    )


def load_player_boxscores(season_start_year: int) -> pd.DataFrame:
    sdv_year = season_to_sdv_year(season_start_year)

    return load_sdv_parquet(
        release_tag="espn_nba_player_boxscores",
        filename=f"player_box_{sdv_year}.parquet",
        cache_name=f"sdv_player_box_{season_start_year}.parquet",
    )


def load_rosters(season_start_year: int) -> pd.DataFrame:
    sdv_year = season_to_sdv_year(season_start_year)

    return load_sdv_parquet(
        release_tag="espn_nba_rosters",
        filename=f"rosters_{sdv_year}.parquet",
        cache_name=f"sdv_rosters_{season_start_year}.parquet",
    )


def load_all(season_start_year: int) -> dict[str, pd.DataFrame]:
    """
    Laedt alle Rohquellen einer Saison und gibt sie als dict zurueck.
    """
    return {
        "pbp": load_pbp(season_start_year),
        "matchups": load_matchups(season_start_year),
        "shots": load_shots(season_start_year),
        "schedule": load_schedule(season_start_year),
        "player_box": load_player_boxscores(season_start_year),
        "rosters": load_rosters(season_start_year),
    }


if __name__ == "__main__":
    # 2025 bedeutet Saison 2025/26 im shufinskiy/cdechoch Archiv.
    raw = load_all(2025)

    for name, df in raw.items():
        inspect(name.upper(), df)
