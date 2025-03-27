from django.urls import path
from . import views



urlpatterns = [
    path('books/', views.index, name='index'),  # change '' to 'books/'
    path('books/update/<int:human_id>/', views.update_human, name='update_human'),
    path('books/delete/<int:human_id>/', views.delete_human, name='delete_human'),
]
