from django.shortcuts import render, redirect
from .models import Category, Page
from .forms import CategoryForm, PageForm

# View for displaying the form and the directory
def index(request):
    categories = Category.objects.all()
    pages = Page.objects.all()

    if request.method == "POST":
        # Handle Category Form Submission
        if 'category_form' in request.POST:
            category_form = CategoryForm(request.POST)
            if category_form.is_valid():
                category_form.save()
                return redirect('index')
        # Handle Page Form Submission
        elif 'page_form' in request.POST:
            page_form = PageForm(request.POST)
            if page_form.is_valid():
                page_form.save()
                return redirect('index')
    else:
        category_form = CategoryForm()
        page_form = PageForm()

    return render(request, 'pages/index.html', {
        'categories': categories,
        'pages': pages,
        'category_form': category_form,
        'page_form': page_form,
    })
