from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import render
from .models import Flight
from .models import Airport
from .models import Passenger


# Create your views here.
def index(request):
    return render(request, 'flights/index.html', {
        "flights":Flight.objects.all(),
        "airports":Airport.objects.all(),
    })

def flightFunc(request, flight_id):
    flight =Flight.objects.get(pk=flight_id)
    return render(request, "flights/flight.html", {
        "flight":flight,
        "passengers":flight.passengers.all(),
        "carriers":flight.carriers.all(),
        "non_passengers":Passenger.objects.exclude(flights=flight).all()

    })

def book(request, flight_id):
    if(request.method=='POST'):
        flight= Flight.objects.get(pk=flight_id)
        passenger=Passenger.objects.get(pk= int(request.POST['passenger']))
        passenger.flights.add(flight)
        return HttpResponseRedirect(reverse("flight", args=(flight.id,)))

def cancelBooking(request, flight_id, passenger_id):
    if request.method == 'POST':  # Good practice to require POST for actions that modify data
        try:
            passenger = Passenger.objects.get(pk=passenger_id)  # Fixed: use objects.get instead of just get
            flight = Flight.objects.get(pk=flight_id)
            passenger.flights.remove(flight)
            return HttpResponseRedirect(reverse("flight", args=(flight.id,)))
        except Passenger.DoesNotExist:
            # Handle case where passenger doesn't exist
            return HttpResponseRedirect(reverse("flight", args=(flight_id,)))
       