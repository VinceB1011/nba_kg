"""
Importiert den Basisgraphen und den validierten CourtStint-Subset nach Neo4j.

Erwartet eine lokale Neo4j-Instanz sowie die durch build_all.py,
resolve_substitutions.py und build_court_stints.py erzeugten Parquet-Tabellen.

Aufruf:
    NEO4J_PASSWORD=... python src/import_neo4j.py
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


PROCESSED_DIR = Path("data/processed")
BATCH_SIZE = 1_000


CONSTRAINTS = [
    "CREATE CONSTRAINT player_id IF NOT EXISTS FOR (p:Player) REQUIRE p.player_id IS UNIQUE",
    "CREATE CONSTRAINT team_id IF NOT EXISTS FOR (t:Team) REQUIRE t.team_id IS UNIQUE",
    "CREATE CONSTRAINT game_id IF NOT EXISTS FOR (g:Game) REQUIRE g.game_id IS UNIQUE",
    "CREATE CONSTRAINT stint_id IF NOT EXISTS FOR (s:CourtStint) REQUIRE s.stint_id IS UNIQUE",
]


def python_value(value: Any) -> Any:
    """Wandelt Pandas- und NumPy-Werte in Neo4j-kompatible Python-Werte um."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Erzeugt Records ohne fehlende Properties."""
    result = []
    for row in df.to_dict("records"):
        result.append(
            {
                key: converted
                for key, value in row.items()
                if (converted := python_value(value)) is not None
            }
        )
    return result


def run_in_batches(session, query: str, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        session.execute_write(lambda tx: tx.run(query, rows=batch).consume())


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {
        name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
        for name in [
            "teams",
            "players",
            "games",
            "player_game_stats",
            "player_game_absences",
            "court_stints",
            "court_stint_players",
        ]
    }

    # Neo4j soll plus_minus als Zahl und nicht als Text erhalten.
    tables["player_game_stats"] = tables["player_game_stats"].copy()
    tables["player_game_stats"]["plus_minus"] = pd.to_numeric(
        tables["player_game_stats"]["plus_minus"], errors="raise"
    ).astype(int)
    return tables


def import_graph(session, tables: dict[str, pd.DataFrame]) -> None:
    for statement in CONSTRAINTS:
        session.run(statement).consume()

    run_in_batches(
        session,
        "UNWIND $rows AS row MERGE (p:Player {player_id: row.player_id}) SET p += row",
        records(tables["players"]),
    )
    run_in_batches(
        session,
        "UNWIND $rows AS row MERGE (t:Team {team_id: row.team_id}) SET t += row",
        records(tables["teams"]),
    )
    run_in_batches(
        session,
        "UNWIND $rows AS row MERGE (g:Game {game_id: row.game_id}) SET g += row",
        records(tables["games"]),
    )

    games = tables["games"]
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (t:Team {team_id: row.team_id})
        MATCH (g:Game {game_id: row.game_id})
        MERGE (t)-[:HOME_TEAM]->(g)
        """,
        records(games.rename(columns={"home_team_id": "team_id"})[["game_id", "team_id"]]),
    )
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (t:Team {team_id: row.team_id})
        MATCH (g:Game {game_id: row.game_id})
        MERGE (t)-[:AWAY_TEAM]->(g)
        """,
        records(games.rename(columns={"away_team_id": "team_id"})[["game_id", "team_id"]]),
    )

    stats = tables["player_game_stats"].copy()
    stats["source"] = "espn_boxscore"
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (p:Player {player_id: row.player_id})
        MATCH (g:Game {game_id: row.game_id})
        MERGE (p)-[r:PLAYED_IN]->(g)
        SET r += row
        """,
        records(stats),
    )

    absences = tables["player_game_absences"].copy()
    absences["source"] = "espn_boxscore"
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (p:Player {player_id: row.player_id})
        MATCH (g:Game {game_id: row.game_id})
        MERGE (p)-[r:DID_NOT_PLAY]->(g)
        SET r += row
        """,
        records(absences),
    )

    listed_for = pd.concat(
        [stats[["player_id", "team_id"]], absences[["player_id", "team_id"]]],
        ignore_index=True,
    ).drop_duplicates()
    listed_for["season"] = 2025
    listed_for["source"] = "espn_game_sheet"
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (p:Player {player_id: row.player_id})
        MATCH (t:Team {team_id: row.team_id})
        MERGE (p)-[r:LISTED_FOR {season: row.season}]->(t)
        SET r.source = row.source
        """,
        records(listed_for),
    )

    stints = tables["court_stints"].copy()
    stints["source"] = "pbp_reconstruction"
    run_in_batches(
        session,
        "UNWIND $rows AS row MERGE (s:CourtStint {stint_id: row.stint_id}) SET s += row",
        records(stints),
    )
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (g:Game {game_id: row.game_id})
        MATCH (s:CourtStint {stint_id: row.stint_id})
        MERGE (g)-[:HAS_STINT]->(s)
        """,
        records(stints[["game_id", "stint_id"]]),
    )
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (p:Player {player_id: row.player_id})
        MATCH (s:CourtStint {stint_id: row.stint_id})
        MERGE (p)-[r:ON_COURT_IN]->(s)
        SET r.team_id = row.team_id
        """,
        records(tables["court_stint_players"]),
    )


def report(session) -> None:
    print("Nodes:")
    for row in session.run("MATCH (n) RETURN labels(n) AS labels, count(*) AS count"):
        print(f"  {row['labels']}: {row['count']}")

    print("Relationships:")
    for row in session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"):
        print(f"  {row['type']}: {row['count']}")


def main() -> None:
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("Bitte NEO4J_PASSWORD setzen.")

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    tables = load_tables()
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            import_graph(session, tables)
            report(session)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
