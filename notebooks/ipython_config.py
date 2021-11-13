import sys
import os

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_BASE_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))

sys.path.insert(1, PROJECT_BASE_PATH)

c.InteractiveShellApp.extensions = [
    'django_extensions.management.notebook_extension'
]