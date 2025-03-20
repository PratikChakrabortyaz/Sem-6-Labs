from django.db import models

class Vote(models.Model):
    GOOD = 'good'
    SATISFACTORY = 'satisfactory'
    BAD = 'bad'
    
    CHOICES = [
        (GOOD, 'Good'),
        (SATISFACTORY, 'Satisfactory'),
        (BAD, 'Bad')
    ]
    
    vote_choice = models.CharField(max_length=20, choices=CHOICES)
    
    def __str__(self):
        return self.vote_choice
