import csv
import sqlite3
from contextlib import closing
from pathlib import Path


def initialize_database(database_path, legacy_csv_path=None, output_dir=None):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                species TEXT NOT NULL,
                species_conf REAL NOT NULL,
                detect_conf REAL NOT NULL,
                image_path TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                name TEXT PRIMARY KEY
            )
            """
        )
        if legacy_csv_path is not None and output_dir is not None:
            _migrate_legacy_csv(
                connection, Path(legacy_csv_path), Path(output_dir)
            )


def _migrate_legacy_csv(connection, csv_path, output_dir):
    migration_name = "detections_csv_to_sqlite"
    migrated = connection.execute(
        "SELECT 1 FROM migrations WHERE name = ?", (migration_name,)
    ).fetchone()
    if migrated or not csv_path.is_file():
        return

    output_root = output_dir.resolve()
    with csv_path.open(newline="") as source:
        for row in csv.DictReader(source):
            image_path = None
            raw_path = row.get("file", "").strip()
            if raw_path:
                try:
                    image_path = Path(raw_path).resolve().relative_to(
                        output_root
                    ).as_posix()
                except ValueError:
                    pass
            connection.execute(
                """
                INSERT INTO detections
                    (timestamp, species, species_conf, detect_conf, image_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["timestamp"],
                    row["species"],
                    float(row["species_conf"]),
                    float(row["detect_conf"]),
                    image_path,
                ),
            )

    connection.execute(
        "INSERT INTO migrations (name) VALUES (?)", (migration_name,)
    )


def record_detection(database_path, timestamp, species, species_conf,
                     detect_conf, image_path):
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO detections
                (timestamp, species, species_conf, detect_conf, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                species,
                species_conf,
                detect_conf,
                Path(image_path).as_posix() if image_path is not None else None,
            ),
        )


def get_detections(database_path, name=None, date_from=None, date_to=None):
    conditions = []
    parameters = []
    if name:
        escaped_name = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions.append("species LIKE ? ESCAPE '\\' COLLATE NOCASE")
        parameters.append(f"%{escaped_name}%")
    if date_from:
        conditions.append("date(timestamp) >= date(?)")
        parameters.append(date_from)
    if date_to:
        conditions.append("date(timestamp) <= date(?)")
        parameters.append(date_to)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            f"""
            SELECT timestamp, species, species_conf, detect_conf, image_path
            FROM detections
            {where_clause}
            ORDER BY timestamp DESC, id DESC
            """,
            parameters,
        ).fetchall()
