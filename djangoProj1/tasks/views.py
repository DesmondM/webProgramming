from django import forms
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse




class NewTaskForm(forms.Form):
    taskTitle =forms.CharField(label="New Task title")
    taskDescription =forms.CharField(label="Task description")
    

tasks1 = [
    {
     'id': '1',
     'title': 'Bath',
    'description': 'Take a bath',
    },
    {
     'id': '2',
     'title': 'Dress',
    'description': 'Get dressed',
    },
    {
      'id': '3',
     'title': 'Breakfast',
    'description': 'Eat breakfast',
    },
    {
     'id': '4',
     'title': 'Work',
    'description': 'Work on the project',
    }
]
# Create your views here.
def index(request):
    if 'tasks' not in request.session:
        request.session['tasks'] = [tasks1]
    return render(request, 'tasks/index.html', {
        'tasks':  request.session['tasks'],
    })

def add(request):
    if request.method=="POST":
        form = NewTaskForm(request.POST)
        if form.is_valid():
            task =  {
                    'title': form.cleaned_data["taskTitle"],
                    'description': form.cleaned_data["taskDescription"]
                    }
            tasks.append(task)
            return HttpResponseRedirect(reverse("tasks:index"))
        else:
            return render(request, "tasks/add.html",{
                'form':form
            })

    return render(request, 'tasks/add.html', {
        'form': NewTaskForm()
    })
