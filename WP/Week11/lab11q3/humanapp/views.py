from django.shortcuts import render, get_object_or_404, redirect
from .models import Human

def index(request):
    humans = Human.objects.all()
    selected_id = request.GET.get('human_id')
    selected = None

    # Handle form submission to create new Human
    if request.method == 'POST' and 'create' in request.POST:
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        phone = request.POST['phone']
        address = request.POST['address']
        city = request.POST['city']
        Human.objects.create(
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            address=address,
            city=city
        )
        return redirect('/books/')  # Refresh to show new entry

    # Handle selection
    if selected_id:
        selected = get_object_or_404(Human, id=selected_id)

    return render(request, 'index.html', {
        'humans': humans,
        'selected': selected
    })


def update_human(request, human_id):
    human = get_object_or_404(Human, id=human_id)
    if request.method == 'POST':
        human.last_name = request.POST['last_name']
        human.phone = request.POST['phone']
        human.address = request.POST['address']
        human.city = request.POST['city']
        human.save()
    return redirect('/')

def delete_human(request, human_id):
    human = get_object_or_404(Human, id=human_id)
    human.delete()
    return redirect('/')
