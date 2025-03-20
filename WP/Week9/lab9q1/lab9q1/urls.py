"""
URL configuration for lab9q1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

# Redirect root URL to the register page
def redirect_to_register(request):
    return redirect('register')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', redirect_to_register),  # This will redirect root URL to the register page
    path('register/', include('userapp.urls')),  # Include userapp URLs
]
