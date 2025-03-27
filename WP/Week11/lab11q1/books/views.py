# books/views.py

from django.shortcuts import render, redirect
from .forms import AuthorForm, PublisherForm, BookForm
from .models import Author, Publisher, Book

# Add an Author
def add_author(request):
    if request.method == 'POST':
        form = AuthorForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('add_author')  # Redirect after saving
    else:
        form = AuthorForm()
    return render(request, 'books/add_author.html', {'form': form})

# Add a Publisher
def add_publisher(request):
    if request.method == 'POST':
        form = PublisherForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('add_publisher')  # Redirect after saving
    else:
        form = PublisherForm()
    return render(request, 'books/add_publisher.html', {'form': form})

# Add a Book
def add_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('add_book')  # Redirect after saving
    else:
        form = BookForm()
    return render(request, 'books/add_book.html', {'form': form})

# View all books
def view_books(request):
    books = Book.objects.all()  # Retrieve all books
    return render(request, 'books/view_books.html', {'books': books})