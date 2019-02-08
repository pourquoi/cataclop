import datetime
import time
import sys

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count

from cataclop.pmu.settings import NODE_PATH, BET_SCRIPT_PATH, SCRAP_DIR
from cataclop.pmu.better import Better
from cataclop.pmu.scrapper import Scrapper
from cataclop.pmu.parser import Parser
from cataclop.core import models

from cataclop.ml.pipeline import factories

from cataclop.pmu.signals import next_race_queued

import logging
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '''
'''

    def add_arguments(self, parser):
        parser.add_argument('--simulation', action='store_true')
        parser.add_argument('--immediate', action='store_true')
        parser.add_argument('--skip-scrap', action='store_true')
        parser.add_argument('--loop', action='store_true')
        parser.add_argument('--dummy', action='store_true')
        parser.add_argument('--provider', type=str)

    def handle(self, *args, **options):

        self.better = Better(NODE_PATH, BET_SCRIPT_PATH)
        self.scrapper = Scrapper(root_dir=SCRAP_DIR)
        self.parser = Parser(SCRAP_DIR)
        self.simulation = options.get('simulation')
        self.skip_scrap = options.get('skip_scrap')
        self.immediate = options.get('immediate')
        self.loop = options.get('loop')
        self.dummy = options.get('dummy')
        self.provider = options.get('provider', None)
        self.programs = []

        print(options)

        self.wait_until_minutes = 3

        self.load_programs()
        self.bet()

    def load_programs(self):
        if self.dummy:
            programs = ['dummy']
        else:
            programs = ['2019-01-07', '2019-01-24', '2019-02-04']

        for p in programs:
            program = factories.Program.factory(p)
            self.programs.append(program)

    def bet(self):

        if self.loop:
            while( True ): 
                self._bet()
                time.sleep(10)

                now = datetime.datetime.now()

                if now.hour == 23 and now.minute == 30:
                    self.scrapper.scrap(force_scrap_races=True, force_scrap_players=True)
                    self.parser.parse()

                    time.sleep(60)

                if now.hour == 5 and now.minute == 30:
                    self.scrapper.scrap(force_scrap_races=True, force_scrap_players=True)
                    self.parser.parse()

                    time.sleep(60)
        else:
            self._bet()


    def _bet(self):

        programs = []

        checked_races = []

        while len(programs) == 0:

            race = self.get_next_race(exclude=checked_races)

            logger.debug(str(race))

            if race is None:
                return

            checked_races.append(race.id)

            programs = [p for p in self.programs if p.check_race(race)]

        next_race_queued.send(sender=self.__class__, race=str(race))

        time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()

        while not self.immediate and time_remaining > 60*self.wait_until_minutes:
            time.sleep(10)
            race.refresh_from_db()
            time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()

        # final scrap

        if not self.skip_scrap:
            self.scrapper.scrap(force_scrap_races=True, force_scrap_players=True)
            self.parser.parse()

        # race might have been delayed
        race.refresh_from_db()
        time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()
        if not self.immediate and time_remaining > 60*self.wait_until_minutes:
            logger.info('race {} delayed'.format(str(race)))
            return

        bets = []

        for program in programs:

            time_remaining = (race.start_at - datetime.datetime.now()).total_seconds()

            if time_remaining > 60:
                
                try:
                    program.predict(dataset_params = {
                        'race_id': race.id
                    }, locked=True, dataset_reload=True)
                except:
                    logger.error('program prediction failed for race: {} {}'.format(race.id, sys.exc_info()[0]))
                    continue

                try:
                    program.bet()
                except:
                    logger.error('program bet failed for race: {}'.format(race.id))
                    continue

                for row in program.bets.itertuples(index=True, name='Pandas'):
                    num = getattr(row, 'num')
                    amount = getattr(row, 'bet')

                    odds_pmu = getattr(row, 'odds_final', None)
                    odds_unibet = getattr(row, 'odds_final_unibet', None)

                    provider = 'pmu'

                    if odds_pmu is not None and odds_unibet is not None and odds_unibet > odds_pmu:
                        provider = 'unibet'

                    if self.provider:
                        provider = self.provider

                    bets.append({
                        'provider': provider,
                        'num': num,
                        'amount': amount,
                        'program': str(program)
                    })

        if len(bets) > 0:
            try:
                self.better.bet(date=race.start_at, session_num=race.session.num, race_num=race.num, bets=bets, simulation=self.simulation)
            except:
                logger.error('bet failed: {}'.format(sys.exc_info()[0]))


    def get_next_race(self, exclude=None):

        #races = models.Race.objects.filter(start_at__gt=datetime.datetime.now(), start_at__date=datetime.date.today())[:1]
        races = models.Race.objects.all().prefetch_related('player_set').filter(
                                player__id__gt=0, start_at__gt=datetime.datetime.now() + datetime.timedelta(minutes=self.wait_until_minutes)
                            ).order_by('start_at')

        if exclude is not None:
            races = races.exclude(id__in=exclude)

        races = races[:1]

        if len(races) == 0:
            return None

        return races[0]