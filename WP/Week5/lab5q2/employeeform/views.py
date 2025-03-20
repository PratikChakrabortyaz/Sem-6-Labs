# employeeform/views.py

from django.shortcuts import render
from .forms import EmployeeForm
from datetime import date

def employee_form(request):
    promotion_status = ''
    
    if request.method == 'POST':
        form = EmployeeForm(request.POST)
        
        if form.is_valid():
            # Get the Date of Joining
            doj = form.cleaned_data['date_of_joining']
            
            # Calculate the experience in years
            experience = (date.today() - doj).days / 365.25  # years
            
            # Check eligibility for promotion
            if experience > 5:
                promotion_status = 'YES'
            else:
                promotion_status = 'NO'
    else:
        form = EmployeeForm()
    
    return render(request, 'employeeform/employee_form.html', {
        'form': form,
        'promotion_status': promotion_status
    })
