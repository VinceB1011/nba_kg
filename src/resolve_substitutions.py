"""
Loest PBP-Substitutionen auf kanonische Player-IDs auf.

Das NBA-V3-PBP nennt bei einer Substitution nur die NBA-ID des Spielers,
der das Feld verlaesst. Der eingewechselte Spieler steht als Kurzname in
`description`, z. B. ``SUB: Gilgeous-Alexander FOR Caruso``.

Die Aufloesung wird deshalb auf Spieler eingeschraenkt, die laut ESPN-
Boxscore fuer genau dieses Team in genau diesem Game eingesetzt wurden.
Nicht aufloesbare Wechsel bleiben als Audit-Zeilen erhalten. Sie werden
nicht als Statlines oder Abwesenheiten repariert.

Aufruf: python src/resolve_substitutions.py
"""

from pathlib import Path
import re

import pandas as pd

from normalize import normalize_player_name, normalize_team_abbreviation


PROCESSED_DIR = Path("data/processed")


def parse_player_in(description: str) -> str | None:
    """Extrahiert den PBP-Kurznamen des eingewechselten Spielers."""
    if not isinstance(description, str):
        return None

    match = re.fullmatch(r"SUB:\s*(.*?)\s+FOR\s+.+", description)
    return match.group(1) if match else None


def matches_pbp_name(pbp_name: str, full_name: str) -> bool:
    """
    Vergleicht einen PBP-Kurznamen mit einem vollstaendigen Namen.

    PBP laesst oft den Vornamen weg ("Gilgeous-Alexander") oder kuerzt ihn
    ab ("K. Williams"). Der Vergleich erfolgt daher zuerst ueber das Ende
    des vollstaendigen Namens und dann ueber Vorname-Praefix plus Nachname.
    """
    short = normalize_player_name(pbp_name)
    full = normalize_player_name(full_name)

    if short is None or full is None:
        return False

    short_parts = short.split()
    full_parts = full.split()

    if len(short_parts) <= len(full_parts) and full_parts[-len(short_parts):] == short_parts:
        return True

    return (
        len(short_parts) == 2
        and short_parts[-1] == full_parts[-1]
        and full_parts[0].startswith(short_parts[0])
    )


def build_active_players(
    stats: pd.DataFrame,
    game_map: pd.DataFrame,
    players: pd.DataFrame,
) -> dict[tuple[int, int], list[dict]]:
    """Bildet pro NBA-Game und Team den Pool der tatsaechlich eingesetzten Spieler."""
    active = (
        stats.merge(
            game_map[["espn_game_id", "nba_game_id"]],
            left_on="game_id",
            right_on="espn_game_id",
        )
        .merge(
            players[["player_id", "nba_player_id", "name"]],
            on="player_id",
        )
        .dropna(subset=["nba_player_id"])
    )

    active["nba_game_id"] = active["nba_game_id"].astype(int)
    active["player_id"] = active["player_id"].astype(int)
    active["nba_player_id"] = active["nba_player_id"].astype(int)

    return (
        active.groupby(["nba_game_id", "team_id"])[["player_id", "nba_player_id", "name"]]
        .apply(lambda group: group.to_dict("records"))
        .to_dict()
    )


def resolve_substitutions(
    pbp: pd.DataFrame,
    game_map: pd.DataFrame,
    stats: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Gibt eine Zeile pro PBP-Substitution mit Aufloesungsstatus zurueck."""
    team_ids = teams.set_index("abbreviation")["team_id"].to_dict()
    active_players = build_active_players(stats, game_map, players)

    game_ids = game_map.set_index("nba_game_id")["espn_game_id"].to_dict()
    player_ids = (
        players.dropna(subset=["nba_player_id"])
        .assign(nba_player_id=lambda df: df["nba_player_id"].astype(int))
        .set_index("nba_player_id")["player_id"]
        .to_dict()
    )

    substitutions = pbp[pbp["actionType"] == "Substitution"].copy()
    substitutions["nba_game_id"] = substitutions["gameId"].astype(int)
    substitutions["game_id"] = substitutions["nba_game_id"].map(game_ids)
    substitutions["team_abbreviation"] = substitutions["teamTricode"].apply(
        normalize_team_abbreviation
    )
    substitutions["team_id"] = substitutions["team_abbreviation"].map(team_ids)
    substitutions["player_in_name"] = substitutions["description"].apply(parse_player_in)
    substitutions["player_out_nba_id"] = substitutions["personId"].astype("Int64")
    substitutions["player_out_id"] = substitutions["player_out_nba_id"].map(player_ids)

    resolved_rows = []
    for row in substitutions.itertuples():
        candidates = active_players.get((row.nba_game_id, row.team_id), [])
        matches = [
            candidate
            for candidate in candidates
            if matches_pbp_name(row.player_in_name, candidate["name"])
        ]

        if len(matches) == 1:
            status = "resolved"
            player_in_id = matches[0]["player_id"]
            player_in_nba_id = matches[0]["nba_player_id"]
        elif len(matches) == 0:
            status = "unresolved_no_match"
            player_in_id = None
            player_in_nba_id = None
        else:
            status = "unresolved_ambiguous"
            player_in_id = None
            player_in_nba_id = None

        resolved_rows.append(
            {
                "resolution_status": status,
                "candidate_count": len(matches),
                "player_in_id": player_in_id,
                "player_in_nba_id": player_in_nba_id,
            }
        )

    resolved = pd.DataFrame(resolved_rows, index=substitutions.index)
    substitutions = pd.concat([substitutions, resolved], axis=1)

    output_columns = [
        "game_id",
        "nba_game_id",
        "actionNumber",
        "period",
        "clock",
        "team_id",
        "team_abbreviation",
        "player_out_id",
        "player_out_nba_id",
        "player_in_id",
        "player_in_nba_id",
        "player_in_name",
        "description",
        "resolution_status",
        "candidate_count",
    ]

    return substitutions[output_columns].rename(columns={"actionNumber": "action_number"})


def validate(substitutions: pd.DataFrame) -> None:
    """Prueft, dass aufgeloeste Wechsel wirklich vollstaendig sind."""
    assert substitutions["game_id"].notna().all(), "PBP-Game ohne ESPN-Mapping"
    assert substitutions["team_id"].notna().all(), "PBP-Team ohne Team-Mapping"

    resolved = substitutions[substitutions["resolution_status"] == "resolved"]
    assert resolved[["player_out_id", "player_in_id"]].notna().all().all()
    assert (resolved["candidate_count"] == 1).all()


def main() -> None:
    pbp = pd.read_parquet("data/raw/nbastatsv3_2025.parquet")
    game_map = pd.read_parquet(PROCESSED_DIR / "game_id_map.parquet")
    stats = pd.read_parquet(PROCESSED_DIR / "player_game_stats.parquet")
    players = pd.read_parquet(PROCESSED_DIR / "players.parquet")
    teams = pd.read_parquet(PROCESSED_DIR / "teams.parquet")

    substitutions = resolve_substitutions(pbp, game_map, stats, players, teams)
    validate(substitutions)

    path = PROCESSED_DIR / "pbp_substitutions.parquet"
    substitutions.to_parquet(path, index=False)

    print(f"[saved] {path}  {substitutions.shape}")
    print(substitutions["resolution_status"].value_counts().to_string())

    unresolved = substitutions[
        substitutions["resolution_status"] != "resolved"
    ]
    print(f"\nNicht aufgeloeste Wechsel: {len(unresolved)}")
    print(
        unresolved[[
            "nba_game_id",
            "team_abbreviation",
            "player_in_name",
            "description",
        ]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
