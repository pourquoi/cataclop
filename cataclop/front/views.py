from django.shortcuts import render
from django.utils import timezone
from django.views.generic.list import ListView

from cataclop.core import models

def dashboard(request):
    total_races = models.Race.objects.count()

    return render(request, 'dashboard.html', {})

