import datetime

from django.core.management.base import BaseCommand, CommandError

from cataclop.pmu.settings import SCRAP_DIR
from cataclop.pmu.scrapper import Scrapper

class Command(BaseCommand):
    help = '''
Scrap the web for races.
This will only save the JSON files.
Execute the parse command to import them in the database.

eg. scrap all January 2020:
python manage.py scrap 2020-01-01 2020-02-01
'''

    def add_arguments(self, parser):
        parser.add_argument('start', nargs='?', type=str, default=None, help='start date (YYYY-MM-DD)')
        parser.add_argument('end', nargs='?', type=str, default=None, help='end date (YYYY-MM-DD)')

    def handle(self, *args, **options):
        scrapper = Scrapper(root_dir=SCRAP_DIR)

        start = options.get('start')
        end = options.get('end')

        if start == 'yesterday':
            start = (datetime.date.today() - datetime.timedelta(1)).isoformat()

        if end == 'yesterday':
            end = (datetime.date.today() - datetime.timedelta(1)).isoformat()

        if start == 'tomorrow':
            start = (datetime.date.today() + datetime.timedelta(1)).isoformat()

        if end == 'tomorrow':
            end = (datetime.date.today() + datetime.timedelta(1)).isoformat()

        with_offline = True

        scrapper.scrap(start, end, True, True, with_offline=with_offline)
