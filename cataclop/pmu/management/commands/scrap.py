import datetime

from django.core.management.base import BaseCommand, CommandError

from cataclop.pmu.settings import SCRAP_DIR
from cataclop.pmu.scrapper import Scrapper

class Command(BaseCommand):
    help = '''
'''

    def add_arguments(self, parser):
        parser.add_argument('start', nargs='?', type=str, default=None)
        parser.add_argument('end', nargs='?', type=str, default=None)

    def handle(self, *args, **options):
        scrapper = Scrapper(root_dir=SCRAP_DIR)

        start = options.get('start')
        end = options.get('end')

        if start == 'yesterday':
            start = (datetime.date.today() - datetime.timedelta(1)).isoformat()

        if end == 'yesterday':
            end = (datetime.date.today() - datetime.timedelta(1)).isoformat()

        scrapper.scrap(start, end, True, True)
