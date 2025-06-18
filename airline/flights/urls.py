from django.contrib import admin
from django.urls import path


from . import views

urlpatterns = [
   path('', views.index, name='index'),
   path ('<int:flight_id>', views.flightFunc, name='flight'),
   path('<int:flight_id>/book', views.book, name='book' ),
   path('<int:flight_id>/<int:passenger_id>', views.cancelBooking, name='cancelBooking')
]