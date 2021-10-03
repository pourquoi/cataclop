from django.test import TestCase
from rest_framework.test import APIClient
from django.test.utils import override_settings
from cataclop.core.tests.utils import factories

TEST_CACHE_SETTING = {
   "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}

@override_settings(CACHES=TEST_CACHE_SETTING)
class RaceModelTests(TestCase):

    def test_get_sessions(self):
        session = factories.RaceSessionFactory()
        session.save()
        race = factories.RaceFactory(session=session)
        race.save()

        client = APIClient()
        response = client.get('/api/sessions/')
        self.assertEqual(200, response.status_code)
        self.assertEqual(response.data['count'], 1)
        self.assertEqual(response.data['results'][0]['id'], session.id)
        self.assertEqual(len(response.data['results'][0]['race_set']), 1)

    def test_get_races(self):
        race = factories.RaceFactory()
        race.save()

        client = APIClient()
        response = client.get('/api/races/')
        self.assertEqual(200, response.status_code)
        self.assertEqual(response.data['count'], 1)
        self.assertEqual(response.data['results'][0]['id'], race.id)

    def test_get_race(self):
        race = factories.RaceFactory()
        race.save()

        client = APIClient()
        response = client.get('/api/races/' + str(race.id) + '/')
        self.assertEqual(200, response.status_code)
        self.assertEqual(response.data['id'], race.id)
