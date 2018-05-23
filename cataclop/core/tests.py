from django.test import TestCase

from .models import Race, Player, RaceSession

class RaceModelTests(TestCase):

    fixtures = ['2017-10-20']

    def setUp(self):
        pass
    
    def test_get_player(self):

        race = Race.objects.all().first()

        player = race.get_player(1)
        self.assertEqual(player.num, 1)

        player = race.get_player(30)
        self.assertIsNone(player)
