from django.conf import settings
import os

DATA_DIR = getattr(settings, 'ML_DATA_DIR', os.path.join(settings.BASE_DIR, 'var/data'))
PROGRAM_DIR = getattr(settings, 'ML_PROGRAM_DIR', os.path.join(settings.BASE_DIR, 'var/programs'))
MODEL_DIR = getattr(settings, 'ML_MODEL_DIR', os.path.join(settings.BASE_DIR, 'var/models'))