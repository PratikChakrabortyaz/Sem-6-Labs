# products/views.py
from django.shortcuts import render, redirect
from .models import Product
from .forms import ProductForm

# View to list all products
def index(request):
    products = Product.objects.all()  # Get all products from the database
    return render(request, 'products/index.html', {'products': products})

# View to add a new product
def add_product(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            form.save()  # Save the new product to the database
            return redirect('index')  # Redirect to the index page
    else:
        form = ProductForm()
    return render(request, 'products/add_product.html', {'form': form})
