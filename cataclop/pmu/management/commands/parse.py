import json
import os
import datetime
import glob

from django.core.management.base import BaseCommand, CommandError

from cataclop.pmu.settings import SCRAP_DIR
from cataclop.pmu.parser import Parser

class Command(BaseCommand):
    help = '''
Parse races json

eg. parse all January race from 2018:
parse "2018-01-*"
'''

    def add_arguments(self, parser):
        parser.add_argument('pattern', nargs='?', type=str, default=None)

    def handle(self, *args, **options):

        parser = Parser(SCRAP_DIR)

        pattern = options.get('pattern', datetime.date.today().isoformat())

        if pattern == 'today' or pattern is None:
            pattern = datetime.date.today().isoformat()
        elif pattern == 'yesterday':
            pattern = (datetime.date.today() - datetime.timedelta(1)).isoformat()

        patterns = pattern.split()

        for pattern in patterns:

            pattern = os.path.join(SCRAP_DIR, pattern)

            self.stdout.write('Parsing pattern {}'.format(pattern))

            dirs = []
            for dir in glob.glob(pattern):
                dirs.append(dir)

            dirs.sort()

            self.stdout.write('Found {} days'.format(len(dirs)))

            for dir in dirs:
                date = os.path.basename(os.path.normpath(dir))
                self.stdout.write('Parsing date {} ...'.format(date))
                parser.parse(date)
