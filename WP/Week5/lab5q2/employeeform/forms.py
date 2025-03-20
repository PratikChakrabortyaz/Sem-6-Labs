# employeeform/forms.py

from django import forms

class EmployeeForm(forms.Form):
    employee_id = forms.ChoiceField(choices=[('1', 'E001'), ('2', 'E002'), ('3', 'E003'), ('4', 'E004')], label="Employee ID")
    date_of_joining = forms.DateField(widget=forms.TextInput(attrs={'type': 'date'}), label="Date of Joining")
