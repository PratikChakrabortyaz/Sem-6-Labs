# student/views.py
from django.shortcuts import render, redirect

def first_page(request):
    subjects = ['Math', 'Science', 'English', 'History', 'Geography']  # Example subjects

    if request.method == 'POST':
        name = request.POST.get('name')
        roll = request.POST.get('roll')
        subject = request.POST.get('subject')
        
        # Store the data in session
        request.session['name'] = name
        request.session['roll'] = roll
        request.session['subject'] = subject
        
        return redirect('second_page')

    return render(request, 'student/firstPage.html', {'subjects': subjects})

def second_page(request):
    # Retrieve session data
    name = request.session.get('name', 'Not available')
    roll = request.session.get('roll', 'Not available')
    subject = request.session.get('subject', 'Not available')

    return render(request, 'student/secondPage.html', {
        'name': name,
        'roll': roll,
        'subject': subject
    })
