from django.dispatch import receiver
from . import signals
from cataclop.users.models import User
from .emails import send_verification_email
from django.db import models
from .models import RaceSession


@receiver(signals.user_registered)
def user_registered(sender, id, **kwargs):
    user = User.objects.get(pk=id)
    send_verification_email(user)


@receiver(models.signals.pre_save, sender=RaceSession)
def on_save_race_session(sender, instance, **kwargs):
    pass
