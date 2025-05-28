import sqlite3

conn = sqlite3.connect('flights.db')
cursor = conn.cursor()

# Select all records
cursor.execute("SELECT * FROM flights")
all_flights = cursor.fetchall()

print("\nCurrent flights in database:")
for flight in all_flights:
    print(f"ID: {flight[0]}, {flight[1]} to {flight[2]}, Duration: {flight[3]} mins")

conn.close()