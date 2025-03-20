# magazine/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.magazine_cover_form, name='magazine_cover_form'),
    path('preview/', views.preview_cover, name='preview_cover'),
]
