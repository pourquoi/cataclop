from django.conf import settings
import os

SCRAP_DIR = getattr(settings, 'PMU_SCRAP_DIR', os.path.join(settings.BASE_DIR, 'var/scrap'))

NODE_PATH = getattr(settings, 'NODE_PATH', 'node')

BET_SCRIPT_PATH = os.path.join(getattr(settings, 'BASE_DIR'), 'scripts', 'bet.js')

SMSAPI_USER = getattr(settings, 'SMSAPI_USER', '9999999')
SMSAPI_SECRET = getattr(settings, 'SMSAPI_SECRET', '9999999')