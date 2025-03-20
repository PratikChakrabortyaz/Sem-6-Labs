

# Create your views here.
# webapp/views.py
# webapp/views.py
from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'basic.html')
