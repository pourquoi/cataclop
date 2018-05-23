import os
import shutil

from cataclop.ml.pipeline import factories
from cataclop.core import models
from cataclop.core.managers import queryset_iterator

import pandas as pd

class Dataset(factories.Dataset):
    
    def __init__(self, name, params):
        super().__init__(name, params)

        self.races = None
        self.players = None
        self.hippos = None

    @property
    def defaults(self):
        return {
            'categories': ['PLAT', 'ATTELE'],
            'from': None,
            'to': None,
            'race': None
        }

    def load(self, force=False):

        if not force and os.path.exists(os.path.join(self.data_dir, 'races.h5')):
            self.races = pd.read_hdf(os.path.join(self.data_dir, 'races.h5'), infer_datetime_format=True)

        if force or self.races is None:

            races = models.Race.objects.all()
            sessions = models.RaceSession.objects.all()
            hippos = models.Hippodrome.objects.all()
            
            races = pd.DataFrame.from_records(races.values(), index='id')
            sessions = pd.DataFrame.from_records(sessions.values(), index='id')
            hippos = pd.DataFrame.from_records(hippos.values(), index='id')

            hippos.index.name = "hippodrome_id"
            sessions.index.name = "session_id"
            races.index.name = "race_id"


            for c in ['category', 'condition_age', 'condition_sex', 'sub_category']:
                races[c] = races[c].astype('category')

            for c in ['country']:
                hippos[c] = hippos[c].astype('category')

            sessions = sessions.join(hippos, on="hippodrome_id", lsuffix="_session", rsuffix="_hippo")
            races = races.join(sessions, on="session_id", lsuffix="_race", rsuffix="_session")

            self.races = races

        if not force and os.path.exists(os.path.join(self.data_dir, 'players.h5')):
            self.players = pd.read_hdf(os.path.join(self.data_dir, 'players.h5'))

        if force or self.players is None:
            players = models.Player.objects.filter(is_racing=True)

            players = pd.DataFrame.from_records(players.values(), index='id')

            players = players.join(self.races, on="race_id", lsuffix="_player", rsuffix="_race")

            players.reset_index(inplace=True)
            #players['start_at_date'] = players['start_at'].dt.date
            players.set_index(['race_id', 'id'], inplace=True)

            players[['winner_dividend', 'placed_dividend']] = players[['winner_dividend', 'placed_dividend']].fillna(0.0)

            self.players = players

    def save(self, clear=False):

        d = self.data_dir

        if not os.path.isdir(d):
            os.makedirs(d)
        elif clear:
            shutil.rmtree(d)
            os.makedirs(d)

        self.races.to_hdf(os.path.join(d, 'races.h5'), 'races')
        self.players.to_hdf(os.path.join(d, 'players.h5'), 'players')
