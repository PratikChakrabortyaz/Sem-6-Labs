from django.shortcuts import render, redirect
from .forms import VoteForm
from .models import Vote

def vote(request):
    if request.method == 'POST':
        form = VoteForm(request.POST)
        if form.is_valid():
            # Save the vote in the database
            vote_choice = form.cleaned_data['vote_choice']
            Vote.objects.create(vote_choice=vote_choice)
            return redirect('result')
    else:
        form = VoteForm()
    
    return render(request, 'voteapp/vote.html', {'form': form})

def result(request):
    # Get the total number of votes
    total_votes = Vote.objects.count()

    # Get the number of votes for each choice
    good_votes = Vote.objects.filter(vote_choice='good').count()
    satisfactory_votes = Vote.objects.filter(vote_choice='satisfactory').count()
    bad_votes = Vote.objects.filter(vote_choice='bad').count()

    # Calculate the percentages
    if total_votes > 0:
        good_percentage = (good_votes / total_votes) * 100
        satisfactory_percentage = (satisfactory_votes / total_votes) * 100
        bad_percentage = (bad_votes / total_votes) * 100
    else:
        good_percentage = satisfactory_percentage = bad_percentage = 0

    return render(request, 'voteapp/result.html', {
        'good_percentage': good_percentage,
        'satisfactory_percentage': satisfactory_percentage,
        'bad_percentage': bad_percentage,
        'good_votes': good_votes,
        'satisfactory_votes': satisfactory_votes,
        'bad_votes': bad_votes,
    })
