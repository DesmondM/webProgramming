from django import forms
from django.shortcuts import render

class NewTaskForm(forms.Form):
    taskTitle =forms.CharField(label="New Task title")
    taskDescription =forms.CharField(label="Task description")

tasks = [
    {
     'id': 1,
     'title': 'Bath',
    'description': 'Take a bath',
    },
    {
     'id': 2,
     'title': 'Dress',
    'description': 'Get dressed',
    },
    {
     'id': 3,
     'title': 'Breakfast',
    'description': 'Eat breakfast',
    },
    {
     'id': 4,
     'title': 'Work',
    'description': 'Work on the project',
    }
]
# Create your views here.
def index(request):
    return render(request, 'tasks/index.html', {
        'tasks': tasks,
    })

def add(request):
    return render(request, 'tasks/add.html', {
        'form': NewTaskForm()
    })
