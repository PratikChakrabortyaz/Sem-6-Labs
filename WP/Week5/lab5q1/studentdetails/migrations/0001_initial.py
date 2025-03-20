# Generated by Django 5.1.6 on 2025-02-13 04:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('dob', models.DateField()),
                ('address', models.TextField()),
                ('contact_number', models.CharField(max_length=15)),
                ('email', models.EmailField(max_length=254)),
                ('english_marks', models.IntegerField()),
                ('physics_marks', models.IntegerField()),
                ('chemistry_marks', models.IntegerField()),
            ],
        ),
    ]
