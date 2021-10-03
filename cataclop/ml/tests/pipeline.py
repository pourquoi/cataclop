from django.test import TestCase

from cataclop.ml.pipeline import factories as pipeline
from cataclop.core.tests.utils import factories


class PipelineTest(TestCase):

    def test_dummy(self):
        session = factories.RaceSessionFactory()
        race = factories.RaceFactory(session=session)
        session.save()
        race.save()

        for i in range(10):
            player = factories.PlayerFactory(race=race, num=i+1)
            player.save()

        program = pipeline.Program.factory(name='dummy')
        program.run(model='predict')

        bets = program.bet()

        self.assertEqual(1, bets.iloc[0]['num'])
