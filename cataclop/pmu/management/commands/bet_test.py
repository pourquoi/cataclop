import datetime

from django.core.management.base import BaseCommand, CommandError

from cataclop.pmu.settings import NODE_PATH, BET_SCRIPT_PATH
from cataclop.pmu.better import Better

class Command(BaseCommand):
    help = '''
'''

    def add_arguments(self, parser):
        parser.add_argument('date', nargs='?', type=str, default=datetime.date.today().isoformat())
        parser.add_argument('session_num', type=int)
        parser.add_argument('race_num', type=int)
        parser.add_argument('bets', type=str)
        parser.add_argument('provider', type=str, default='pmu')

    def handle(self, *args, **options):
        better = Better(NODE_PATH, BET_SCRIPT_PATH)

        bets = []

        s = options.get('bets').split(',')

        for ss in s:
            sss = ss.split(':')
            bets.append({
                'provider': options.get('provider'),
                'num': sss[0],
                'amount': sss[1],
                'program': 'test'
            })

        date = datetime.datetime.strptime(options.get('date', datetime.date.today().isoformat()), '%Y-%m-%d')

        better.bet(date=date, session_num=options.get('session_num'), race_num=options.get('race_num'), bets=bets, simulation=False)

    