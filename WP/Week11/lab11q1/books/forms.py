# books/forms.py

from django import forms
from .models import Author, Publisher, Book

class AuthorForm(forms.ModelForm):
    class Meta:
        model = Author
        fields = ['first_name', 'last_name', 'email']

class PublisherForm(forms.ModelForm):
    class Meta:
        model = Publisher
        fields = ['name', 'street_address', 'city', 'state_province', 'country', 'website']

class BookForm(forms.ModelForm):
    authors = forms.ModelMultipleChoiceField(queryset=Author.objects.all(), widget=forms.CheckboxSelectMultiple)
    publisher = forms.ModelChoiceField(queryset=Publisher.objects.all())

    class Meta:
        model = Book
        fields = ['title', 'publication_date', 'authors', 'publisher']
