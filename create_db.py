import sqlite3

# Create or connect to the database
conn = sqlite3.connect('flights.db')
cursor = conn.cursor()

# Create flights table
cursor.execute('''
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    duration INTEGER NOT NULL
)
''')

# Save changes and close connection
conn.commit()
conn.close()

print("Flights table created successfully in flights.db")