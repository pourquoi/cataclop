import django.dispatch
from django.dispatch import receiver
import requests
import os

from cataclop.pmu.settings import SMSAPI_USER, SMSAPI_SECRET, NEXT_RACE_LOG_FILE

next_race_queued = django.dispatch.Signal(providing_args=["race"])

bet_placed = django.dispatch.Signal(providing_args=["race", "horse", "amount"])


@receiver(bet_placed)
def bet_sms_notification(sender, **kwargs):
    if not SMSAPI_USER or not SMSAPI_SECRET:
        return
    try:
        msg = '{}: bet {}€ on {}'.format(kwargs.get('race'), kwargs.get('amount'), kwargs.get('horse'))
        requests.get('https://smsapi.free-mobile.fr/sendmsg',
                     params={'user': SMSAPI_USER, 'pass': SMSAPI_SECRET, 'msg': msg})
    except:
        pass


@receiver(next_race_queued)
def log_next_race(sender, **kwargs):
    try:
        with open(NEXT_RACE_LOG_FILE, 'w') as f:
            f.write(kwargs.get('race'))
    except:
        pass
