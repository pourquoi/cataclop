from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register([Player, RaceSession, Race, Owner, Herder, Jockey, Trainer, Horse, Hippodrome, Odds])