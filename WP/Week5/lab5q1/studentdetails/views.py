# studentdetails/views.py

from django.shortcuts import render
from .forms import StudentForm
from .models import Student

def student_details(request):
    if request.method == "POST":
        form = StudentForm(request.POST)
        if form.is_valid():
            # Save the student data
            student = form.save()
            # Calculate percentage
            percentage = student.percentage()
            # Display the student details and the percentage
            return render(request, 'studentdetails/display_details.html', {'student': student, 'percentage': percentage, 'form': form})
    else:
        form = StudentForm()
    
    return render(request, 'studentdetails/student_form.html', {'form': form})
