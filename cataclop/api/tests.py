from django.test import TestCase
from django.test import Client
from rest_framework.test import APIClient

from django.test.utils import override_settings

TEST_CACHE_SETTING = {
   "default": {
        "BACKEND": "drjango.core.cache.backends.dummy.DummyCache",
    }
}

@override_settings(CACHES=TEST_CACHE_SETTING)
class RaceModelTests(TestCase):

    def test_get_sessions(self):
        client = APIClient()
        response = client.get('/api/sessions/')
        self.assertEqual(200, response.status_code)
        pass