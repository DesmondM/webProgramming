from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def greet(request,name):
    return render(request, "hello/greet.html", {"name": name})