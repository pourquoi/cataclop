import os
import subprocess

from datetime import datetime
from decimal import Decimal

from .models import Bet
from cataclop.core.models import Player

from .signals import bet_placed

import logging
logger = logging.getLogger(__name__)


class Better:

    def __init__(self, node_path, script_path):
        self.node_path = node_path
        self.script_path = script_path

    def bet(self, date, session_num, race_num, bets, simulation=True):
        """
        Execute a list of bets on a race.
        :param date: date or datetime of the race
        :param session_num: session number
        :param race_num: race number
        :param bets: list of bets. Each bet is a dict with the player number, the bet amount and the program name. e.g [('num': 3, 'amount': 1.5, 'program': test')]
        :param simulation: if True the process will skip the bet confirmation
        """

        if len(bets) == 0:
            return

        if date is None:
            date = datetime.datetime.now()

        url = 'https://www.pmu.fr/turf/{}/R{}/C{}'.format(date.strftime('%d%m%Y'), session_num, race_num)


        # group bets by player
        players_bet = []

        for b in bets:
            exists = False

            for bb in players_bet:
                if( bb['num'] == b['num'] ):
                    bb['amount'] += b['amount']
                    exists = True
            
            if not exists:
                players_bet.append({
                    'num': b['num'],
                    'amount': b['amount']
                })

        bet_cmd = ','.join( ['{}:{}'.format(b['num'], b['amount']) for b in players_bet] )

        args = [
            self.node_path,
            self.script_path,
            url,
            bet_cmd,
            'gagnant',
            '1' if simulation else '0'
        ]

        bet_logs = []

        for b in bets:

            logger.info('betting on {} in race R{}C{}'.format(b['num'], session_num, race_num))

            try:
                player = Player.objects.get(num=b['num'], race__start_at__date=date, race__num=race_num, race__session__num=session_num)
            except:
                logger.error('bet player not found')
                player = None

            bet = Bet(player=player, url=url, amount=Decimal('{}'.format(b['amount'])), simulation=simulation, program=b['program'])
            bet_logs.append(bet)

        p = subprocess.run(args, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(p)

        for bet in bet_logs:
            bet.returncode = p.returncode
            bet.stderr = p.stderr
            bet.stdout = p.stdout
            bet.save()

            bet_placed.send(sender=self.__class__, race="{} R{}C{}".format(date.strftime('%Y-%m-%d'), session_num, race_num), horse=str(bet.player) if bet.player is not None else '?', amount=bet.amount)
