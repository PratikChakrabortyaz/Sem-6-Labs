# webapp/urls.py
# webapp/urls.py
# webapp/urls.py
# webapp/urls.py
from django.urls import path, re_path
from . import views

urlpatterns = [
    # Root path for webapp, will show current month and year if no parameters
    path('', views.index, name='index'),


]
