from django.shortcuts import render
from .models import Flight
from .models import Airport


# Create your views here.
def index(request):
    return render(request, 'flights/index.html', {
        "flights":Flight.objects.all(),
        "airports":Airport.objects.all(),
    })

def flightFunc(request, flight_id):
    flight =Flight.objects.get(pk=flight_id),
    return render(request, "flights/flight.html", {
        "flight":flight,
        "passenger":flight.passengers.all()

    })