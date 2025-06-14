from django.db import models

# Create your models here.
class Airport(models.Model):
    code = models.CharField(max_length=3)
    city = models.CharField(max_length=64)

    def __str__(self):
        return f"{self.city} ({self.code})"
    
class Flight(models.Model):
    origin = models.ForeignKey(Airport, related_name='departures', on_delete=models.CASCADE)
    destination = models.ForeignKey(Airport, related_name='arrivals', on_delete=models.CASCADE)
    duration = models.IntegerField()

    def __str__(self):
        return f"{self.id}: {self.origin} to {self.destination}, Duration: {self.duration} mins"
    
class Passenger(models.Model):
    firstName =models.CharField(max_length=64)
    lastName=models.CharField(max_length=64)
    flights=models.ManyToManyField(Flight, blank=True, related_name="passengers")

    def __str__(self):
        return f"{self.firstName} {self.lastName}"
    
class Carrier(models.Model):
    name = models.CharField(max_length=64)
    manufacturer = models.CharField(max_length=64)
    planeModel = models.CharField(max_length=64)
    flights= models.ManyToManyField(Flight, blank=False, related_name="carriers")


    def __str__(self):
        return f"{self.name} {self.manufacturer} {self.planeModel}"
