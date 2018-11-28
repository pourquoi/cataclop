import datetime
import time

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count

from cataclop.pmu.settings import NODE_PATH, BET_SCRIPT_PATH
from cataclop.pmu.better import Better
from cataclop.core import models

from cataclop.ml.pipeline import factories

class Command(BaseCommand):
    help = '''
'''

    def add_arguments(self, parser):
        parser.add_argument('--simulation', action='store_true')
        parser.add_argument('--immediate', action='store_true')

    def handle(self, *args, **options):

        self.better = Better(NODE_PATH, BET_SCRIPT_PATH)
        self.simulation = options.get('simulation')
        self.immediate = options.get('immediate')

        print(options)

        self.bet()

    def bet(self):

        race = self.get_next_race()

        print(race)

        if race is None:
            return

        time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()
 
        while not self.immediate and time_remaining > 60*5:
            time.sleep(10)

        if time_remaining > 60:
            program = factories.Program.factory('default')
            program.predict(dataset_params = {
                'race_id': race.id
            })
            program.bet(max_odds=None)

            for row in program.bets.itertuples(index=True, name='Pandas'):
                self.better.bet(date=race.start_at, session_num=race.session.num, race_num=race.num, num=getattr(row, 'num'), amount=1.5, simulation=self.simulation)
                break

    def get_next_race(self):

        #races = models.Race.objects.filter(start_at__gt=datetime.datetime.now(), start_at__date=datetime.date.today())[:1]
        races = models.Race.objects.all().prefetch_related('player_set').filter(
                                player__id__gt=0, start_at__gt=datetime.datetime.now()
                            )[:1]
                            

        if len(races) == 0:
            return None

        return races[0]