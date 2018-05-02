from django.conf import settings
import os

SCRAP_DIR = getattr(settings, 'PMU_SCRAP_DIR', os.path.join(settings.BASE_DIR, 'var/scrap'))