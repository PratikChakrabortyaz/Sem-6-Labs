# books/urls.py

# books/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.view_books, name='view_books'),  # Default page for /books/
    path('add_author/', views.add_author, name='add_author'),
    path('add_publisher/', views.add_publisher, name='add_publisher'),
    path('add_book/', views.add_book, name='add_book'),
    path('view_books/', views.view_books, name='view_books'),
]
