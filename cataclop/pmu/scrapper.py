import datetime
import requests
import json
import os
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Scrapper:

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.scrappers = []

        self.scrappers.append(PmuScrapper(root_dir))
        self.scrappers.append(UnibetScrapper(root_dir))

    def scrap(self, start_date=None, end_date=None, force_scrap_races=False, force_scrap_players=False, **kwargs):

        for scrapper in self.scrappers:

            scrapper.scrap(start_date=start_date, end_date=end_date, force_scrap_races=force_scrap_races, force_scrap_players=force_scrap_players, **kwargs)


class PmuScrapper:

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def scrap(self, start_date=None, end_date=None, force_scrap_races=False, force_scrap_players=False, with_offline=True, **kwargs):
        '''
        scrap races from start_date to end_date included
        '''

        if start_date is None:
            start_date = datetime.date.today()

        if end_date is None:
            end_date = datetime.date.today()

        logger.info('scrapping from {} to {}'.format(start_date, end_date))

        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        for date in dates:

            base_url = 'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/' + date.strftime('%d%m%Y')
            url = base_url

            logger.debug(url)

            base_dir = os.path.join(self.root_dir, date.strftime('%Y-%m-%d'))

            f = os.path.join(base_dir, 'programme.json')

            if os.path.isfile(f) and not force_scrap_races:
                with open(f, 'r') as data_file:
                    data = json.load(data_file)
            else:
                r = requests.get(url, params={'meteo': 'true', 'specialisation': 'INTERNET'})
                if r.status_code == requests.codes.ok:
                    data = r.json()
                else:
                    logger.error('request failed with status {}. {}'.format(r.status_code, url))
                    return

            if not os.path.exists(os.path.dirname(f)):
                os.makedirs(os.path.dirname(f))

            with open(f, 'w') as data_file:
                json.dump(data, data_file)

            for session in data['programme']['reunions']:

                for race in session['courses']:

                    race_name = 'R{}C{}'.format(race['numReunion'], race['numOrdre'])

                    race_url = '{}/R{}/C{}/'.format(base_url, race['numReunion'], race['numOrdre'])

                    odds_file = os.path.join(base_dir, race_name + '-odds.json')

                    race_time = datetime.datetime.fromtimestamp(race['heureDepart'] / 1000)

                    now = datetime.datetime.now()

                    time_remaining = (race_time - now).total_seconds()

                    # if race has ended, get the final odds
                    if( time_remaining < 0 and 'rapportsDefinitifsDisponibles' in race and race['rapportsDefinitifsDisponibles'] and not os.path.isfile(odds_file) ):
                        url = race_url + 'rapports-definitifs'
                        r = requests.get(url, {'combinaisonEnTableau': 'true', 'specialisation': 'INTERNET'})
                        if r.status_code == requests.codes.ok:
                            data = r.json()
                            if data:
                                with open(odds_file, 'w') as f:
                                    json.dump(data, f)
                        else:
                            logger.error('final odds request failed for race {} {}. status {}'.format(date.strftime('%Y-%m-%d'), race_name, r.status_code))


                    url = race_url + 'participants'

                    modes = ['INTERNET']

                    if with_offline:
                        modes.append('OFFLINE')

                    for mode in modes:
                        
                        # BC
                        if mode != 'INTERNET':
                            race_file = os.path.join(base_dir, race_name + '.' + mode.lower() + '.json')
                        else:
                            race_file = os.path.join(base_dir, race_name + '.json')

                        if not os.path.isfile(race_file) or force_scrap_players:

                            r = requests.get(url, {'specialisation': mode})

                            if r.status_code == requests.codes.ok:

                                with open(race_file, 'w') as f:
                                    json.dump(r.json(), f)

                            else:
                                logger.error('race request failed for race {} {} {}'.format(mode, date.strftime('%Y-%m-%d'), race_name))

                    logger.debug('R{} C{} in {}'.format(race['numReunion'], race['numOrdre'], (race_time - now)))


class UnibetScrapper:

    name = 'unibet'

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def scrap(self, start_date=None, end_date=None, force_scrap_races=False, force_scrap_players=False, **kwargs):

        if start_date is None:
            start_date = datetime.date.today()

        if end_date is None:
            end_date = datetime.date.today()

        logger.info('scrapping from {} to {}'.format(start_date, end_date))

        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        for date in dates:

            url = 'https://www.unibet.fr/zones/turf/program.json'

            base_dir = os.path.join(self.root_dir, date.strftime('%Y-%m-%d'))

            f = os.path.join(base_dir, 'programme.{}.json'.format(self.name))

            if os.path.isfile(f) and not force_scrap_races:
                with open(f, 'r') as data_file:
                    data = json.load(data_file)
            else:
                r = requests.get(url, params={'date': date.strftime('%Y-%m-%d')})
                if r.status_code == requests.codes.ok:
                    data = r.json()
                else:
                    logger.error('request failed with status {}. {}'.format(r.status_code, url))
                    return

            if not os.path.exists(os.path.dirname(f)):
                os.makedirs(os.path.dirname(f))

            with open(f, 'w') as data_file:
                json.dump(data, data_file)

            for session in data:
                for race in session['races']:

                    race_name = 'R{}C{}'.format(session['rank'], race['rank'])

                    url = 'https://www.unibet.fr/zones/turf/race.json'
                    
                    race_file = os.path.join(base_dir, race_name + '.' + self.name + '.json')
                    
                    if not os.path.isfile(race_file) or force_scrap_players:

                        r = requests.get(url, {'raceId': race['zeturfId']})

                        if r.status_code == requests.codes.ok:

                            with open(race_file, 'w') as f:
                                json.dump(r.json(), f)

                        else:
                            logger.error('race request failed for race {} {} {}'.format(mode, date.strftime('%Y-%m-%d'), race_name))
