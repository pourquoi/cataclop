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
        parser.add_argument('num', type=int)

    def handle(self, *args, **options):
        better = Better(NODE_PATH, BET_SCRIPT_PATH)

        date = datetime.datetime.strptime(options.get('date', datetime.date.today().isoformat()), '%Y-%m-%d')

        print("betting race {} R{}C{} num {}".format(date, options.get('session_num'), options.get('race_num'), options.get('num')))

        better.bet(date, options.get('session_num'), options.get('race_num'), options.get('num'), 1.5)

    