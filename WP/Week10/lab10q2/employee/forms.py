from django import forms
from .models import Works, Lives

# Form for inserting data into WORKS table
class WorksForm(forms.ModelForm):
    class Meta:
        model = Works
        fields = ['person_name', 'company_name', 'salary']

# Form for inserting data into LIVES table
class LivesForm(forms.ModelForm):
    class Meta:
        model = Lives
        fields = ['person_name', 'street', 'city']

# Form to filter people by company
class CompanyFilterForm(forms.Form):
    company_name = forms.CharField(max_length=255)
