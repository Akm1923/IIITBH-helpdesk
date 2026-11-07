import sqlite3

from app.core.config import get_settings

settings = get_settings()
database_path = (settings.data_dir / "ingestion.sqlite3").resolve()
conn = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
cursor = conn.cursor()

print("\n--- DATABASE TABLES ---")

tables = cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
""").fetchall()

for table in tables:
    print(table[0])

conn.close()