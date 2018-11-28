import os
import subprocess

from datetime import datetime
from decimal import Decimal

from .models import Bet
from cataclop.core.models import Player

import logging
logger = logging.getLogger(__name__)


class Better:

    def __init__(self, node_path, script_path):
        self.node_path = node_path
        self.script_path = script_path

    def bet(self, date, session_num, race_num, num, amount, simulation=True):

        if date is None:
            date = datetime.now()

        url = 'https://www.pmu.fr/turf/{}/R{}/C{}'.format(date.strftime('%d%m%Y'), session_num, race_num)

        args = [
            self.node_path,
            self.script_path,
            url,
            "{}".format(num),
            "{}".format(amount),
            'gagnant',
            '1' if simulation else '0'
        ]

        logger.info('betting on {} in race R{}C{}'.format(num, session_num, race_num))

        try:
            player = Player.objects.get(num=num, race__start_at__date=date, race__num=race_num, race__session__num=session_num)
        except:
            logger.error('bet player not found')
            player = None

        bet = Bet(player=player, amount=Decimal('{}'.format(amount)), simulation=simulation)

        p = subprocess.run(args, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        bet.returncode = p.returncode
        bet.stderr = p.stderr
        bet.stdout = p.stdout
        bet.save()
        