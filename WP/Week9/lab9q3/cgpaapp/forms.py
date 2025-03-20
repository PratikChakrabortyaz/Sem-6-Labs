from django import forms

class CGPAForm(forms.Form):
    name = forms.CharField(max_length=100, required=True)
    total_marks = forms.IntegerField(required=True)
