import os

from configurations import Configuration, values

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

class Common(Configuration):
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    SECRET_KEY = ''

    DEBUG = True

    ALLOWED_HOSTS = []

    DJANGO_APPS = [
        'django_extensions',
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'rest_framework'
    ]

    PROJECT_APPS = [
        'cataclop.users',
        'cataclop.core',
        'cataclop.api',
        'cataclop.pmu'
    ]

    INSTALLED_APPS = DJANGO_APPS + PROJECT_APPS

    MIDDLEWARE = [
        'django.middleware.security.SecurityMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'django.contrib.messages.middleware.MessageMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ]

    ROOT_URLCONF = 'cataclop.urls'

    TEMPLATES = [
        {
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        },
    ]

    REST_FRAMEWORK = {
        'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.LimitOffsetPagination',
        'PAGE_SIZE': 20
    }

    WSGI_APPLICATION = 'cataclop.wsgi.application'

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'OPTIONS': {
                'read_default_file': os.path.join(BASE_DIR, 'my.cnf'),
                'init_command': "SET sql_mode='STRICT_TRANS_TABLES'"
            }
        }
    }

    AUTH_USER_MODEL = 'users.User'

    AUTH_PASSWORD_VALIDATORS = [
        {
            'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
        },
    ]

    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
            }
        },
        'loggers': {
            #'django.db.backends': {
            #    'handlers': ['console'],
            #    'level': 'DEBUG',
            #},
            'cataclop': {
                'handlers': ['console'],
                'level': 'DEBUG'
            }
        }
    }

    LANGUAGE_CODE = 'en-us'

    TIME_ZONE = 'Europe/Paris'

    USE_I18N = True

    USE_L10N = True

    USE_TZ = False

    STATIC_URL = '/static/'

    PMU_SCRAP_DIR = '/path/to/scrap/dir'

    NOTEBOOK_ARGUMENTS = [
        '--notebook-dir', 'notebooks',
    ]

    IPYTHON_ARGUMENTS = [
        '--ext', 'django_extensions.management.notebook_extension',
        #'--debug',
    ]

class Development(Common):
    DEBUG = True

class Staging(Common):
    DEBUG = False

class Production(Staging):

    @classmethod
    def pre_setup(cls):
        super(Production, cls).pre_setup()
        
        sentry_sdk.init(
            dsn="https://<key>@sentry.io/<project>",
            integrations=[DjangoIntegration()]
        )