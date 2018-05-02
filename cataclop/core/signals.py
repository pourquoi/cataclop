from django.dispatch import receiver
from django.db import models

from .models import *

@receiver(models.signals.pre_save, sender=RaceSession)
def on_save_race_session(sender, instance, **kwargs):

    pass
