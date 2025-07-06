from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import render

# Create your views here.
def index(request):
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login'))
    return render(request, 'users/users.html', {'user': request.user})
   
def login(request):
    if request.method=='POST':
        username= request.POST['username']
        password= request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return HttpResponseRedirect(reverse('index'))
        else:
            return render(request, 'users/login.html', {'loginError': 'Invalid credentials'})
    return render(request, 'users/login.html')

def logout(request):
    auth_logout(request)
    return render(request, 'users/login.html', {'loginError': 'You have been logged out.'})