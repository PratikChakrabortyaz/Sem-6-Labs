from django.db import models

# Model for the Category
class Category(models.Model):
    name = models.CharField(max_length=255)
    num_visits = models.IntegerField(default=0)
    num_likes = models.IntegerField(default=0)

    def __str__(self):
        return self.name

# Model for the Page
class Page(models.Model):
    category = models.ForeignKey(Category, related_name='pages', on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    url = models.URLField()
    views = models.IntegerField(default=0)

    def __str__(self):
        return self.title
