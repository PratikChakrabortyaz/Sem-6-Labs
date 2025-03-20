from django.shortcuts import render, redirect
from .forms import CGPAForm


def index(request):
    if request.method == 'POST':
        form = CGPAForm(request.POST)
        if form.is_valid():

            request.session['name'] = form.cleaned_data['name']
            request.session['total_marks'] = form.cleaned_data['total_marks']
            return redirect('result')
    else:
        form = CGPAForm()
    
    return render(request, 'cgpaapp/index.html', {'form': form})


def result(request):
    name = request.session.get('name')
    total_marks = request.session.get('total_marks')


    if total_marks is not None:
        cgpa = total_marks / 50
    else:
        cgpa = 0

    return render(request, 'cgpaapp/result.html', {'name': name, 'cgpa': cgpa})
