"""
Rekonstruiert CourtStints aus einem konservativen Teil der Saison.

Das V3-PBP protokolliert nicht jede Aenderung an Viertelgrenzen. Deshalb
wird der Court zu Beginn jedes Viertels aus Spieler-Events vor dem ersten
Wechsel rekonstruiert. Ein Game wird nur behalten, wenn jede Aufstellung
eindeutig ist, alle Wechsel konsistent sind und die Stint-Minuten dem
Boxscore entsprechen.

Aufruf: python src/build_court_stints.py
"""

from pathlib import Path
import re

import pandas as pd


PROCESSED_DIR = Path("data/processed")
# ESPN speichert diese Saison Minuten nur ganzzahlig, PBP mit Zehntelsekunden.
MINUTE_TOLERANCE = 0.51


def elapsed_seconds(period: int, clock: str) -> float:
    """Wandelt eine NBA-Clock in Sekunden seit Spielbeginn um."""
    match = re.fullmatch(r"PT(\d+)M(\d+(?:\.\d+)?)S", clock)
    if match is None:
        raise ValueError(f"Unerwartete PBP-Clock: {clock}")

    minutes, seconds = int(match.group(1)), float(match.group(2))
    prior_seconds = 720 * min(period - 1, 4) + 300 * max(period - 5, 0)
    period_seconds = 720 if period <= 4 else 300
    return prior_seconds + period_seconds - (minutes * 60 + seconds)


def initial_lineup_for_period(
    game: pd.Series,
    period: int,
    pbp: pd.DataFrame,
    stats: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
) -> dict[int, set[int]]:
    """Leitet die fuenf Spieler jedes Teams zu Beginn eines Viertels ab."""
    game_id = int(game["game_id"])
    nba_game_id = int(game["nba_game_id"])
    team_ids = [int(game["home_team_id"]), int(game["away_team_id"])]

    active = (
        stats[stats["game_id"] == game_id]
        .merge(players[["player_id", "nba_player_id"]], on="player_id")
        .dropna(subset=["nba_player_id"])
    )
    active["nba_player_id"] = active["nba_player_id"].astype(int)

    nba_team_ids = teams.set_index("team_id")["nba_team_id"].to_dict()
    events = pbp[(pbp["gameId"] == nba_game_id) & (pbp["period"] == period)]
    substitutions = events[events["actionType"] == "Substitution"]
    first_substitution = substitutions.groupby("teamId")["actionNumber"].min().to_dict()

    lineup = {}
    for team_id in team_ids:
        nba_team_id = int(nba_team_ids[team_id])
        candidates = set(
            active.loc[active["team_id"] == team_id, "nba_player_id"]
        )
        before_first_sub = events["actionNumber"] < first_substitution.get(
            nba_team_id, float("inf")
        )
        observed = set(
            events.loc[
                before_first_sub
                & (events["teamId"] == nba_team_id)
                & events["personId"].isin(candidates),
                "personId",
            ].astype(int)
        )

        if len(observed) != 5:
            raise ValueError(
                f"Game {nba_game_id}, Q{period}, team={team_id}: "
                f"{len(observed)} Spieler vor erstem Wechsel beobachtet"
            )
        lineup[team_id] = observed

    return lineup


def add_stint(
    stints: list[dict],
    game_id: int,
    nba_game_id: int,
    start: tuple[int, str, float],
    end: tuple[int, str, float],
    lineup: dict[int, set[int]],
) -> None:
    """Fuegt nur Stints mit positiver Dauer hinzu."""
    duration = end[2] - start[2]
    if duration == 0:
        return
    if duration < 0:
        raise ValueError(f"Negative Stint-Dauer: {start} -> {end}")

    players_on_court = set().union(*lineup.values())
    if len(players_on_court) != 10 or {len(team) for team in lineup.values()} != {5}:
        raise ValueError(f"Ungültiges Lineup: {lineup}")

    stint = {
            "game_id": game_id,
            "nba_game_id": nba_game_id,
            "start_period": start[0],
            "start_clock": start[1],
            "start_elapsed_seconds": start[2],
            "end_period": end[0],
            "end_clock": end[1],
            "end_elapsed_seconds": end[2],
            "duration_seconds": duration,
            "lineup": {team_id: sorted(players) for team_id, players in lineup.items()},
        }

    # Falls an der Viertelgrenze dieselben zehn Spieler bleiben, ist es
    # laut Definition weiterhin derselbe maximale CourtStint.
    if (
        stints
        and stints[-1]["end_elapsed_seconds"] == start[2]
        and stints[-1]["lineup"] == stint["lineup"]
    ):
        stints[-1].update(
            {
                "end_period": end[0],
                "end_clock": end[1],
                "end_elapsed_seconds": end[2],
                "duration_seconds": stints[-1]["duration_seconds"] + duration,
            }
        )
    else:
        stint["stint_number"] = len(stints) + 1
        stints.append(stint)


def apply_substitution_group(
    group: pd.DataFrame,
    lineup: dict[int, set[int]],
) -> None:
    """Wendet alle Wechsel derselben Period/Clock gleichzeitig an."""
    for team_id, team_changes in group.groupby("team_id"):
        team_id = int(team_id)
        players_out = set(team_changes["player_out_nba_id"].astype(int))
        players_in = set(team_changes["player_in_nba_id"].astype(int))
        current = lineup[team_id]

        if not players_out.issubset(current):
            raise ValueError(
                f"Wechsel: Spieler nicht auf dem Feld, team={team_id}, "
                f"out={players_out}, lineup={current}"
            )
        if players_in & (current - players_out):
            raise ValueError(
                f"Wechsel: Spieler bereits auf dem Feld, team={team_id}, "
                f"in={players_in}, lineup={current}"
            )

        next_lineup = (current - players_out) | players_in
        if len(next_lineup) != 5:
            raise ValueError(
                f"Wechsel erzeugt nicht fuenf Spieler, team={team_id}, "
                f"lineup={next_lineup}"
            )
        lineup[team_id] = next_lineup


def reconstruct_game(
    game: pd.Series,
    substitutions: pd.DataFrame,
    stats: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Rekonstruiert CourtStints eines Games mit vollstaendig aufgeloesten Wechseln."""
    game_id = int(game["game_id"])
    nba_game_id = int(game["nba_game_id"])
    game_pbp = pbp[pbp["gameId"] == nba_game_id]

    changes = substitutions[substitutions["game_id"] == game_id].copy()
    if not (changes["resolution_status"] == "resolved").all():
        raise ValueError(f"Game {nba_game_id} hat ungeloeste Substitutionen")

    stints: list[dict] = []
    max_period = int(game_pbp["period"].max())

    for period in range(1, max_period + 1):
        lineup = initial_lineup_for_period(
            game, period, game_pbp, stats, players, teams
        )
        period_changes = changes[changes["period"] == period].copy()
        period_changes["elapsed_seconds"] = [
            elapsed_seconds(current_period, clock)
            for current_period, clock in zip(
                period_changes["period"], period_changes["clock"]
            )
        ]
        period_changes = period_changes.sort_values(["elapsed_seconds", "action_number"])

        start_clock = "PT12M00.00S" if period <= 4 else "PT05M00.00S"
        start = (period, start_clock, elapsed_seconds(period, start_clock))
        for (_, _), group in period_changes.groupby(["period", "clock"], sort=False):
            row = group.iloc[0]
            end = (period, row["clock"], float(row["elapsed_seconds"]))
            add_stint(stints, game_id, nba_game_id, start, end, lineup)
            apply_substitution_group(group, lineup)
            start = end

        final = (period, "PT00M00.00S", elapsed_seconds(period, "PT00M00.00S"))
        add_stint(stints, game_id, nba_game_id, start, final, lineup)

    return pd.DataFrame(stints)


def validate_minutes(
    stints: pd.DataFrame,
    game_id: int,
    stats: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Vergleicht rekonstruierte Minuten pro Spieler mit den ESPN-Boxscore-Minuten."""
    rows = []
    for stint in stints.itertuples():
        for team_id, player_ids in stint.lineup.items():
            for nba_player_id in player_ids:
                rows.append(
                    {
                        "nba_player_id": nba_player_id,
                        "stint_seconds": stint.duration_seconds,
                    }
                )

    reconstructed = (
        pd.DataFrame(rows)
        .groupby("nba_player_id", as_index=False)["stint_seconds"]
        .sum()
    )
    expected = (
        stats[stats["game_id"] == game_id]
        .merge(players[["player_id", "nba_player_id", "name"]], on="player_id")
        [["nba_player_id", "name", "minutes"]]
    )

    result = expected.merge(reconstructed, on="nba_player_id", how="outer")
    result["stint_minutes"] = result["stint_seconds"] / 60
    result["difference_minutes"] = result["stint_minutes"] - result["minutes"]

    return result.sort_values("difference_minutes", key=lambda col: col.abs(), ascending=False)


def games_with_observable_quarter_starts(
    pbp: pd.DataFrame,
    players: pd.DataFrame,
) -> set[int]:
    """Findet Games, deren PBP vor jedem ersten Wechsel genau fuenf Spieler zeigt."""
    known_player_ids = set(players["nba_player_id"].dropna().astype(int))
    player_events = pbp[
        pbp["teamId"].notna()
        & pbp["teamTricode"].notna()
        & pbp["personId"].isin(known_player_ids)
    ]
    substitutions = pbp[pbp["actionType"] == "Substitution"]
    first_substitution = (
        substitutions.groupby(["gameId", "period", "teamId"])["actionNumber"]
        .min()
        .rename("first_substitution")
        .reset_index()
    )
    visible = player_events.merge(
        first_substitution,
        on=["gameId", "period", "teamId"],
        how="left",
    )
    visible = visible[
        visible["first_substitution"].isna()
        | visible["actionNumber"].lt(visible["first_substitution"])
    ]

    observed_counts = visible.groupby(["gameId", "period", "teamId"])["personId"].nunique()
    all_team_periods = (
        pbp[pbp["teamId"].notna() & pbp["teamTricode"].notna()]
        .groupby(["gameId", "period", "teamId"])
        .size()
        .index
    )
    observed_counts = observed_counts.reindex(all_team_periods, fill_value=0)
    per_game = observed_counts.rename("players").reset_index().groupby("gameId")["players"]

    return set(per_game.agg(lambda counts: (counts == 5).all()).loc[lambda ok: ok].index)


def split_stint_players(
    stints: pd.DataFrame,
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trennt Stint-Properties und die zehn Player-Stint-Zugehoerigkeiten."""
    stints = stints.copy()
    stints["stint_id"] = (
        stints["game_id"].astype(str) + "_" + stints["stint_number"].astype(str)
    )

    player_ids = (
        players.dropna(subset=["nba_player_id"])
        .assign(nba_player_id=lambda df: df["nba_player_id"].astype(int))
        .set_index("nba_player_id")["player_id"]
        .to_dict()
    )
    membership_rows = []
    for stint in stints.itertuples():
        for team_id, nba_player_ids in stint.lineup.items():
            for nba_player_id in nba_player_ids:
                player_id = player_ids.get(nba_player_id)
                if player_id is None:
                    raise ValueError(f"NBA-Spieler ohne ESPN-Mapping: {nba_player_id}")
                membership_rows.append(
                    {
                        "stint_id": stint.stint_id,
                        "player_id": player_id,
                        "team_id": team_id,
                    }
                )

    memberships = pd.DataFrame(membership_rows)
    assert set(memberships.groupby("stint_id").size()) == {10}
    assert set(memberships.groupby(["stint_id", "team_id"]).size()) == {5}

    return stints.drop(columns="lineup"), memberships


def build_clean_stints(
    games: pd.DataFrame,
    substitutions: pd.DataFrame,
    stats: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baut nur die Games, die alle konservativen CourtStint-Pruefungen bestehen."""
    candidate_nba_ids = games_with_observable_quarter_starts(pbp, players)
    audit_rows = []
    stint_frames = []

    for game in games.sort_values("date").itertuples(index=False):
        game_dict = pd.Series(game._asdict())
        if game.nba_game_id not in candidate_nba_ids:
            audit_rows.append(
                {
                    "game_id": game.game_id,
                    "nba_game_id": game.nba_game_id,
                    "status": "excluded",
                    "reason": "quarter_start_not_fully_observed",
                    "max_minute_difference": None,
                }
            )
            continue

        try:
            game_pbp = pbp[pbp["gameId"] == game.nba_game_id]
            stints = reconstruct_game(
                game_dict, substitutions, stats, players, teams, game_pbp
            )
            minute_check = validate_minutes(stints, int(game.game_id), stats, players)
            max_difference = minute_check["difference_minutes"].abs().max()
            if max_difference > MINUTE_TOLERANCE:
                raise ValueError(
                    f"minute_difference_exceeds_{MINUTE_TOLERANCE:.2f}"
                )
        except ValueError as error:
            audit_rows.append(
                {
                    "game_id": game.game_id,
                    "nba_game_id": game.nba_game_id,
                    "status": "excluded",
                    "reason": str(error),
                    "max_minute_difference": None,
                }
            )
            continue

        stint_frames.append(stints)
        audit_rows.append(
            {
                "game_id": game.game_id,
                "nba_game_id": game.nba_game_id,
                "status": "included",
                "reason": None,
                "max_minute_difference": max_difference,
            }
        )

    return pd.concat(stint_frames, ignore_index=True), pd.DataFrame(audit_rows)


def main() -> None:
    games = pd.read_parquet("data/processed/games.parquet")
    substitutions = pd.read_parquet("data/processed/pbp_substitutions.parquet")
    stats = pd.read_parquet("data/processed/player_game_stats.parquet")
    players = pd.read_parquet("data/processed/players.parquet")
    teams = pd.read_parquet("data/processed/teams.parquet")
    pbp = pd.read_parquet("data/raw/nbastatsv3_2025.parquet")

    stints, audit = build_clean_stints(
        games, substitutions, stats, players, teams, pbp
    )
    court_stints, court_stint_players = split_stint_players(stints, players)

    court_stints.to_parquet(PROCESSED_DIR / "court_stints.parquet", index=False)
    court_stint_players.to_parquet(
        PROCESSED_DIR / "court_stint_players.parquet", index=False
    )
    audit.to_parquet(PROCESSED_DIR / "court_stint_audit.parquet", index=False)

    print(f"[saved] court_stints: {court_stints.shape}")
    print(f"[saved] court_stint_players: {court_stint_players.shape}")
    print(f"[saved] court_stint_audit: {audit.shape}")
    print("\nAudit:")
    print(audit.groupby(["status", "reason"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
