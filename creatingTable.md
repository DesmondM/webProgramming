import sqlite3

conn = sqlite3.connect('flights.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    duration INTEGER NOT NULL
);
''')

conn.commit()
conn.close()
print("Flights table created!")