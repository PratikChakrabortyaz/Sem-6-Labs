# employeeform/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.employee_form, name='employee_form'),
]
