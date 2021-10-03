from django.contrib import admin
from .models import *


class BetAdmin(admin.ModelAdmin):
    raw_id_fields = ['player']


admin.site.register(Bet, BetAdmin)
