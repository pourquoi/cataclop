import os
import shutil
import logging

import pandas as pd

from cataclop.ml.pipeline import factories
from cataclop.core import models

from cataclop.ml.preprocessing import append_hist, model_to_dict


class Dataset(factories.Dataset):
    
    def __init__(self, name, params, version=None):
        super().__init__(name=name, params=params, version=version)

        self.players = None

        self.agg_features = []
        self.agg_features_funcs = []

        self.logger = logging.getLogger(__name__)

    @property
    def defaults(self):
        return {
            'categories': None,
            'sub_categories': None,
            'from': None,
            'to': None,
            'race': None,
            'nan_flag': +100000
        }

    def load(self, force=False):

        if not force and os.path.exists(os.path.join(self.data_dir, 'players.gzip')):
            self.logger.debug('loading data {} from cache'.format(self.hash))
            self.players = pd.read_parquet(os.path.join(self.data_dir, 'players.gzip'))

        if force or self.players is None:
            self.create_dataframe()

    def save(self, clear=False):

        d = self.data_dir

        if not os.path.isdir(d):
            os.makedirs(d)
        elif clear:
            shutil.rmtree(d)
            os.makedirs(d)

        path = os.path.join(d, 'players.gzip')
        if os.path.isfile(path):
            os.remove(path)

        self.players.to_parquet(path, compression='gzip')

    def create_dataframe(self):
        races = models.Race.objects.all().prefetch_related('player_set', 'session')
        if self.params.get('from') is not None:
            races = races.filter(start_at__gte=self.params.get('from'))
        if self.params.get('to') is not None:
            races = races.filter(start_at__lte=self.params.get('to'))
        if self.params.get('race_id') is not None:
            races = races.filter(id=self.params.get('race_id'))
        if self.params.get('categories') is not None:
            races = races.filter(category__in=self.params.get('categories'))
        if self.params.get('sub_categories') is not None:
            races = races.filter(sub_category__in=self.params.get('sub_categories'))

        hippos = models.Hippodrome.objects.all()

        sessions = [race.session for race in races]
        sessions = list(map(lambda s: model_to_dict(s), set(sessions)))

        hippos = [model_to_dict(hippo) for hippo in hippos]

        players = [model_to_dict(p) for race in races for p in race.player_set.all()]

        races = [model_to_dict(race) for race in races]

        races_df = pd.DataFrame.from_records(races, index='id')
        sessions_df = pd.DataFrame.from_records(sessions, index='id')
        hippos_df = pd.DataFrame.from_records(hippos, index='id')

        hippos_df.index.name = "hippodrome_id"
        sessions_df.index.name = "session_id"
        races_df.index.name = "race_id"

        for c in ['category', 'condition_age', 'condition_sex', 'sub_category']:
            races_df[c] = races_df[c].astype('category')

        for c in ['country']:
            hippos_df[c] = hippos_df[c].astype('category')

        sessions_df = sessions_df.join(hippos_df, on="hippodrome_id", lsuffix="_session", rsuffix="_hippo")
        races_df = races_df.join(sessions_df, on="session_id", lsuffix="_race", rsuffix="_session")
        races_df.date = races_df.date.astype('str')

        df = pd.DataFrame.from_records(players, index='id')

        df = df.join(races_df, on="race_id", lsuffix="_player", rsuffix="_race")

        df.reset_index(inplace=True)
        df.set_index(['id'], inplace=True)

        df['winner_dividend'].fillna(0., inplace=True)
        df['winner_dividend'] = df.apply(lambda p: p['final_odds'] * 100. if p['winner_dividend'] == 0 and p['position'] == 1 else p['winner_dividend'], axis=1)

        df = append_hist(df, 6)

        df.date = df.date.astype('str')

        self.players = df
