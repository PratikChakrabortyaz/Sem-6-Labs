from django.db import models

class Human(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    phone = models.CharField(max_length=20)
    address = models.CharField(max_length=200)
    city = models.CharField(max_length=50)

    def __str__(self):
        return self.first_name
