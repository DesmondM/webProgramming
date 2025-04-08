class Flight():
    def __init__(self, capacity):
        self.capacity = capacity
        self.passengers= []

    def open_seats(self):
        return self.capacity-len(self.passengers)
    
    def add_passenger(self, name):
        if self.open_seats()==0:
            return False
        self.passengers.append(name)
        return True


flight = Flight(3)
people =['Desmond', 'Nokwanda', 'Sbonginkosi', 'Simbonge', 'Thandeka']
for person in people:
    if flight.add_passenger(person):
        print(f'Added {person} to flight successfully!')
    else:
        print(f'Sorry, {person} cannot be added to flight.')

