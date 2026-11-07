import sqlite3

conn = sqlite3.connect("ingestion.sqlite3")
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