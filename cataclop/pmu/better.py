import os
import subprocess

import datetime
from decimal import Decimal

from .models import Bet
from cataclop.core.models import Player

from .signals import bet_placed

import logging
logger = logging.getLogger(__name__)

class Better:

    def __init__(self, node_path, script_path):
        self.node_path = node_path
        self.script_path = os.path.join(script_path)

    def bet(self, date, session_num, race_num, bets, simulation=True):
        """
        Execute a list of bets on a race.
        :param date: date or datetime of the race
        :param session_num: session number
        :param race_num: race number
        :param bets: list of bets. Each bet is a dict with the provider, the player number, the bet amount and the program name. e.g [('provider': 'pmu', 'num': 3, 'amount': 1.5, 'program': test')]
        :param simulation: if True the process will skip the bet confirmation
        """

        if len(bets) == 0:
            return

        if date is None:
            date = datetime.datetime.now()

        urls = {
            'pmu': 'https://www.pmu.fr/turf/{}/R{}/C{}'.format(date.strftime('%d%m%Y'), session_num, race_num),
            'unibet': 'https://www.unibet.fr/turf/race/{}-R{}-C{}-t.html'.format(date.strftime('%d-%m-%Y'), session_num, race_num)
        }

        # group bets by player
        players_bet = {
            'pmu': [],
            'unibet': []
        }

        for b in bets:
            exists = False

            provider = b.get('provider', 'pmu')

            for bb in players_bet[provider]:
                if bb['num'] == b['num']:
                    bb['amount'] += b['amount']
                    exists = True
            
            if not exists:
                players_bet[provider].append(b)

        bet_logs = []

        for provider in players_bet.keys():

            url = urls[provider]

            if not len(players_bet[provider]):
                continue

            bet_cmd = ','.join( ['{}:{}'.format(b['num'], b['amount']) for b in players_bet[provider]] )

            args = [
                self.node_path,
                os.path.join(self.script_path,'bet.{}.js'.format(provider)),
                url,
                bet_cmd,
                'gagnant',
                '1' if simulation else '0'
            ]

            for b in players_bet[provider]:

                logger.info('betting on {} in race R{}C{}'.format(b['num'], session_num, race_num))

                try:
                    player = Player.objects.get(num=b['num'], race__start_at__date=date, race__num=race_num, race__session__num=session_num)
                except:
                    logger.error('bet player not found')
                    player = None

                bet = Bet(player=player, url=url, amount=Decimal('{}'.format(b['amount'])), simulation=simulation, program=b['program'])
                bet.stats_prediction_time = b.get('prediction_time', None)
                bet.stats_scrap_time = b.get('scrap_time', None)
                bet.provider = provider
                bet_logs.append(bet)

            t1 = datetime.datetime.now()
            p = subprocess.run(args, timeout=90, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            t2 = datetime.datetime.now()
            bet_time = (t2 - t1).total_seconds()

            for b in bet_logs:
                if b.provider == provider:
                    b.stats_bet_time = bet_time

        for bet in bet_logs:
            bet.returncode = p.returncode
            bet.stderr = p.stderr
            bet.stdout = p.stdout
            bet.save()

            bet_placed.send(sender=self.__class__, race="{} R{}C{}".format(date.strftime('%Y-%m-%d'), session_num, race_num), horse=str(bet.player) if bet.player is not None else '?', amount=bet.amount)
