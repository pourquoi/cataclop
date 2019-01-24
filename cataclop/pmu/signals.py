import django.dispatch
from django.dispatch import receiver
import requests

from cataclop.pmu.settings import SMSAPI_USER, SMSAPI_SECRET

bet_placed = django.dispatch.Signal(providing_args=["race", "horse", "amount"])

@receiver(bet_placed)
def bet_sms_notification(sender, **kwargs):

    try:
        msg = '{}: bet {}€ on {}'.format(kwargs.get('race'), kwargs.get('amount'), kwargs.get('horse'))
        requests.get('https://smsapi.free-mobile.fr/sendmsg', params={'user': SMSAPI_USER, 'pass': SMSAPI_SECRET, 'msg': msg})
    except:
        pass