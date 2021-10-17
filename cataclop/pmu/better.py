import os
import subprocess

import datetime
import random
import math
import requests
import json
from decimal import Decimal

from .models import Bet
from cataclop.core.models import Player
from cataclop.settings import PMU_CLIENT_ID, PMU_CLIENT_DOB, PMU_CLIENT_PASSWORD

from .signals import bet_placed

import logging

logger = logging.getLogger(__name__)


class PMUClient:

    def __init__(self):
        self.session = None
        self.response = None
        self.version = 61

    @staticmethod
    def gen_correlation():
        l = 10
        return ''.join([str(math.floor(random.random() * 10)) if random.random() > 0.5 else chr(97 + math.ceil(25 * random.random())) for i in range(l)])

    def login(self):
        s = requests.Session()

        s.headers.update({
          "accept": "application/json",
          "accept-language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
          "cache-control": "no-cache",
          "content-type": "application/json",
          "pragma": "no-cache",
          "sec-ch-ua": "\"Chromium\";v=\"94\", \"Google Chrome\";v=\"94\", \";Not A Brand\";v=\"99\"",
          "sec-ch-ua-mobile": "?0",
          "sec-ch-ua-platform": "\"macOS\"",
          "sec-fetch-dest": "empty",
          "sec-fetch-mode": "cors",
          "sec-fetch-site": "same-site",
          "sec-gpc": "1",
        })

        s.get("https://www.pmu.fr/")
        r = s.get("https://connect.awl.pmu.fr/auth/client/{}/session/pinpad".format(self.version))
        pinpad = r.json()
        password = ''.join([pinpad["pinPadCodes"][int(c) + 3][1] for c in PMU_CLIENT_PASSWORD])
        payload = {
            'codeConf': password,
            'dateNaissance': PMU_CLIENT_DOB,
            'numeroExterne': PMU_CLIENT_ID,
            'pinpadKey': pinpad["keyPinpad"]
        }
        r = s.post("https://connect.awl.pmu.fr/auth/client/{}/session".format(self.version), json=payload)

        s.headers.update({'pmu-session-id': s.cookies.get('pmusid')})

        self.session = s
        self.response = r
        return self.response.json()

    def bet(self, session_num, race_num, num, amount=None, date=None):
        if self.session is None:
            self.login()

        if date is None:
            date = datetime.datetime.now()
        if amount is None:
            amount = 100

        payload = [{
            'correlationId': self.gen_correlation(),
            'numeroReunion': session_num,
            'numeroCourse': race_num,
            'dateReunion': int(datetime.datetime.strptime(date.strftime('%Y-%m-%d'), '%Y-%m-%d').timestamp()*1000),
            'pari': 'E_SIMPLE_GAGNANT',
            'formule': 'UNITAIRE',
            'bases': [num],
            'associes': [],
            'selectionChampLibre': [],
            'complement': None,
            'spot': False,
            'dtlo': False,
            'valeur': int(amount)
        }]

        self.response = self.session.post("https://connect.awl.pmu.fr/turfPari/client/{}/parier".format(self.version), json=payload)
        return self.response.json()


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

        client = PMUClient()
        client.login()

        bets_processed = []

        for provider in players_bet.keys():

            if provider != 'pmu':
                logger.warning('provider {} not supported'.format(provider))
                continue

            for b in players_bet[provider]:

                logger.info('betting on {} in race R{}C{}'.format(b['num'], session_num, race_num))

                try:
                    player = Player.objects.get(num=b['num'], race__start_at__date=date, race__num=race_num,
                                                race__session__num=session_num)
                except:
                    logger.error('bet player not found')
                    player = None

                bet = Bet(player=player, amount=Decimal('{}'.format(b['amount'])), simulation=simulation,
                          program=b['program'])

                t1 = datetime.datetime.now()
                if not simulation:
                    try:
                        client.bet(session_num, race_num, b['num'], amount=b['amount'] * 100, date=date)
                        bet.stdout = client.response.text()
                    except Exception as err:
                        bet.stderr = str(err)
                        logger.error(err)
                t2 = datetime.datetime.now()

                bet.stats_prediction_time = b.get('prediction_time', None)
                bet.stats_scrap_time = b.get('scrap_time', None)
                bet.provider = provider
                bet.stats_bet_time = (t2 - t1).total_seconds()
                bet.save()

                bets_processed.append(bet)

                bet_placed.send(sender=self.__class__,
                                race="{} R{}C{}".format(date.strftime('%Y-%m-%d'), session_num, race_num),
                                horse=str(bet.player) if bet.player is not None else '?', amount=bet.amount)

        return bets_processed
