# car/views.py
from django.shortcuts import render

def index(request):
    car_manufacturers = ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi']  # Example manufacturers
    return render(request, 'car/index.html', {'car_manufacturers': car_manufacturers})

def result(request):
    manufacturer = request.GET.get('manufacturer')
    model = request.GET.get('model')
    return render(request, 'car/result.html', {'manufacturer': manufacturer, 'model': model})
