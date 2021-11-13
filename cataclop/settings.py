import os
from datetime import timedelta

from pathlib import Path
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    # set casting, default value
    DEBUG=(bool, False),
    SCHEME=(str, 'http'),
    PORT=(int, None),
    CORS_ALLOW_ALL_ORIGINS=(bool, False),
    EMAIL_RECIPIENTS=(tuple, None),
    PMU_CLIENT_ID=(str, None),
    PMU_CLIENT_PASSWRD=(str, None),
    PMU_CLIENT_DOB=(str, None),
    SMSAPI_USER=(str, None),
    SMSAPI_SECRET=(str, None)
)

environ.Env.read_env(os.path.join(BASE_DIR, '.env'))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = env('SECRET_KEY')

DEBUG = env('DEBUG')

ALLOWED_HOSTS = []

DJANGO_APPS = [
    'django_extensions',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_filters',
    'mjml',
    'corsheaders'
]

PROJECT_APPS = [
    'cataclop.users',
    'cataclop.core',
    'cataclop.api',
    'cataclop.pmu',
    'cataclop.front'
]

INSTALLED_APPS = DJANGO_APPS + PROJECT_APPS

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
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
    'DEFAULT_PAGINATION_CLASS': 'cataclop.api.pagination.ApiPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
    'DEFAULT_PERMISSION_CLASSES': ('rest_framework.permissions.IsAuthenticated',),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    )
}

WSGI_APPLICATION = 'cataclop.wsgi.application'

DATABASES = {
    'default': env.db()
}

AUTH_USER_MODEL = 'users.User'
AUTHENTICATION_BACKENDS = ['cataclop.users.auth.AppModelBackend']


SIMPLE_JWT = {
    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.SlidingToken',),

    'SLIDING_TOKEN_REFRESH_EXP_CLAIM': 'refresh_exp',
    'SLIDING_TOKEN_LIFETIME': timedelta(days=30),
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=30),
}


AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 6,
        }
    }
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
STATIC_ROOT = '/app/static'

PMU_SCRAP_DIR = env('SCRAP_DIR')

PMU_CLIENT_ID = env('PMU_CLIENT_ID')
PMU_CLIENT_PASSWORD = env('PMU_CLIENT_PASSWORD')
PMU_CLIENT_DOB = env('PMU_CLIENT_DOB')

SMSAPI_USER = env('SMSAPI_USER')
SMSAPI_SECRET = env('SMSAPI_SECRET')

NOTEBOOK_ARGUMENTS = [
    '--notebook-dir', 'notebooks',
    '--ip', '0.0.0.0',
    '--allow-root',
    '--no-browser'
]

IPYTHON_ARGUMENTS = [
    '--ext', 'django_extensions.management.notebook_extension',
    #'--debug',
]

MJML_BACKEND_MODE = 'cmd'
MJML_EXEC_CMD = os.path.join(BASE_DIR, 'node_modules/.bin/mjml')

os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

EMAIL_CONFIG = env.email(
    'EMAIL_URL'
)

vars().update(EMAIL_CONFIG)

CORS_ALLOW_ALL_ORIGINS = env('CORS_ALLOW_ALL_ORIGINS')
CORS_ALLOWED_ORIGIN_REGEXES = env('CORS_ALLOWED_ORIGIN')

DEFAULT_FROM_EMAIL = env('EMAIL_FROM')
EMAIL_RECIPIENTS = env('EMAIL_RECIPIENTS')

PROJECT_NAME = env('PROJECT_NAME')

HOST = env('HOST')
PORT = env('PORT')
SCHEME = env('SCHEME')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

'''
sentry_sdk.init(
    dsn="https://<key>@sentry.io/<project>",
    integrations=[DjangoIntegration()]
)
'''