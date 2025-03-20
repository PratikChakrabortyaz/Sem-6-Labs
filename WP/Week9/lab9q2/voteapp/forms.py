from django import forms
from .models import Vote

class VoteForm(forms.Form):
    vote_choice = forms.ChoiceField(choices=Vote.CHOICES, widget=forms.RadioSelect)
