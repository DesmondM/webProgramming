import sqlite3

conn = sqlite3.connect('flights.db')
cursor = conn.cursor()

# Multiple records as a list of tuples
flights_data = [
    ('Los Angeles', 'Tokyo', 600),
    ('Chicago', 'Paris', 480),
    ('Miami', 'Toronto', 180)
]

cursor.executemany('''
INSERT INTO flights (origin, destination, duration)
VALUES (?, ?, ?)
''', flights_data)

conn.commit()
conn.close()
print(f"{len(flights_data)} records inserted")