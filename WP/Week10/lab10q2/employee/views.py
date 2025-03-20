from django.shortcuts import render, redirect
from .models import Works, Lives
from .forms import WorksForm, LivesForm, CompanyFilterForm

# View to insert data into WORKS and LIVES tables
def index(request):
    if request.method == "POST":
        # Handling form submissions for WORKS
        works_form = WorksForm(request.POST)
        lives_form = LivesForm(request.POST)
        company_filter_form = CompanyFilterForm(request.POST)
        
        if 'works_form' in request.POST:
            if works_form.is_valid():
                works_form.save()
                return redirect('index')
        
        # Handling form submissions for LIVES
        elif 'lives_form' in request.POST:
            if lives_form.is_valid():
                lives_form.save()
                return redirect('index')

        # Handling company filter form submission
        elif 'company_filter_form' in request.POST:
            if company_filter_form.is_valid():
                company_name = company_filter_form.cleaned_data['company_name']
                employees = Works.objects.filter(company_name=company_name)
                cities = {}
                
                # Retrieving city information for the employees
                for employee in employees:
                    city = Lives.objects.filter(person_name=employee.person_name).first()
                    if city:
                        cities[employee.person_name] = city.city
                
                return render(request, 'employee/index.html', {
                    'works_form': works_form,
                    'lives_form': lives_form,
                    'company_filter_form': company_filter_form,
                    'cities': cities,
                    'company_name': company_name,
                })
    else:
        works_form = WorksForm()
        lives_form = LivesForm()
        company_filter_form = CompanyFilterForm()

    return render(request, 'employee/index.html', {
        'works_form': works_form,
        'lives_form': lives_form,
        'company_filter_form': company_filter_form,
    })
