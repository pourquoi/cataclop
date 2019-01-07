import datetime
import time

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count

from cataclop.pmu.settings import NODE_PATH, BET_SCRIPT_PATH, SCRAP_DIR
from cataclop.pmu.better import Better
from cataclop.pmu.scrapper import Scrapper
from cataclop.pmu.parser import Parser
from cataclop.core import models

from cataclop.ml.pipeline import factories

class Command(BaseCommand):
    help = '''
'''

    def add_arguments(self, parser):
        parser.add_argument('--simulation', action='store_true')
        parser.add_argument('--immediate', action='store_true')
        parser.add_argument('--loop', action='store_true')

    def handle(self, *args, **options):

        self.better = Better(NODE_PATH, BET_SCRIPT_PATH)
        self.scrapper = Scrapper(root_dir=SCRAP_DIR)
        self.parser = Parser(SCRAP_DIR)
        self.simulation = options.get('simulation')
        self.immediate = options.get('immediate')
        self.loop = options.get('loop')
        self.programs = []

        print(options)

        self.load_programs()
        self.bet()

    def load_programs(self):
        programs = ['2019-01-07']

        for p in programs:
            program = factories.Program.factory(p)
            self.programs.append(program)

    def bet(self):

        if self.loop:
            while( True ): 
                self._bet()
                time.sleep(10)
        else:
            self._bet()


    def _bet(self):

        programs = []

        max_race_check = 30
        checked_races = []

        while len(programs) == 0:

            if len(checked_races) > max_race_check:
                return

            race = self.get_next_race(exclude=checked_races)

            print(race)

            if race is None:
                return

            checked_races.append(race.id)

            programs = [p for p in self.programs if p.check_race(race)]

        time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()

        while not self.immediate and time_remaining > 60*5:
            time.sleep(10)

        # final scrap

        self.scrapper.scrap(force_scrap_races=True, force_scrap_players=True)
        self.parser.parse()

        bets = []

        for program in programs:

            if self.get_next_race() != race:
                return

            time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()

            if time_remaining > 60:
                program.predict(dataset_params = {
                    'race_id': race.id
                }, locked=True, dataset_reload=True)
                program.bet()

                for row in program.bets.itertuples(index=True, name='Pandas'):
                    num = getattr(row, 'num')
                    amount = getattr(row, 'bet')

                    bets.append({
                        'num': num,
                        'amount': amount,
                        'program': str(program)
                    })
                
        self.better.bet(date=race.start_at, session_num=race.session.num, race_num=race.num, bets=bets, simulation=self.simulation)


    def get_next_race(self, exclude=None):

        #races = models.Race.objects.filter(start_at__gt=datetime.datetime.now(), start_at__date=datetime.date.today())[:1]
        races = models.Race.objects.all().prefetch_related('player_set').filter(
                                player__id__gt=0, start_at__gt=datetime.datetime.now()
                            ).order_by('start_at')

        if exclude is not None:
            races = races.exclude(id__in=exclude)

        races = races[:1]

        if len(races) == 0:
            return None

        return races[0]