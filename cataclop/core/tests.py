from django.test import TestCase
from faker import Faker
from datetime import datetime

from .fixtures import factories

from .models import Race, Player, RaceSession, Horse

class RaceModelTests(TestCase):

    def setUp(self):
        pass
    
    def test_get_player(self):
        race = factories.RaceFactory()
        p1 = factories.PlayerFactory(race=race, num=1)
        p2 = factories.PlayerFactory(race=race, num=2)

        player = race.get_player(1)
        self.assertEqual(player.num, 1)

        player = race.get_player(30)
        self.assertIsNone(player)

    def test_get_labels(self):
        race = Race(category='SOME_CATEGORY', sub_category='SOME_SUB_CATEGORY')

        self.assertEqual(race.get_category_label(), 'Some Category')
        self.assertEqual(race.get_sub_category_label(), 'Some Sub Category')

        race = Race(category='PLAT', sub_category='HANDICAP', condition_age='DEUX_TROIS_QUATRE_ANS', condition_sex='FEMELLES_ET_MALES')
        self.assertEqual(race.get_category_label(), 'Plat')
        self.assertEqual(race.get_sub_category_label(), 'Handicap')
        self.assertEqual(race.get_condition_sex_label(), 'Femelles et mâles')
        self.assertEqual(race.get_condition_age_label(), '2, 3, 4 ans')
