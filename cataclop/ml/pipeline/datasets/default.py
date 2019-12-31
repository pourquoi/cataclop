import os
import shutil
import logging

import pandas as pd
import numpy as np

from cataclop.ml.pipeline import factories
from cataclop.core import models
from cataclop.core.managers import queryset_iterator

from cataclop.ml.preprocessing import append_hist, model_to_dict

class Dataset(factories.Dataset):
    
    def __init__(self, name, params, version=None):
        super().__init__(name=name, params=params, version=version)

        self.players = None

        self.agg_features = ['race_count', 
                        'victory_count', 
                        'placed_2_count', 
                        'placed_3_count',
                        'victory_earnings',
                        'placed_earnings',
                        'prev_year_earnings',
                        'handicap_distance',
                        'handicap_weight',
                        'final_odds_ref',
                        'final_odds_ref_offline'
                    ]
        self.agg_features_funcs = [
            ('mean', np.mean), 
            ('std', np.std), 
            ('amin', np.min), 
            ('amax', np.max)
        ]

        self.logger = logging.getLogger(__name__)

    @property
    def defaults(self):
        return {
            'categories': None, #['PLAT', 'ATTELE'],
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
        NAN_FLAG = self.params['nan_flag']

        races = models.Race.objects.all().prefetch_related('player_set', 'session', 'session__hippodrome')
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
        if self.params.get('countries') is not None:
            races = races.filter(session__hippodrome__country__in=self.params.get('countries'))

        hippos = models.Hippodrome.objects.all()

        sessions = [race.session for race in races]
        sessions = list( map(lambda s: model_to_dict(s), set(sessions)) )

        hippos = [ model_to_dict(hippo) for hippo in hippos ]

        players = [ model_to_dict(p) for race in races for p in race.player_set.all() ]

        races = [ model_to_dict(race) for race in races ]

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
        df['placed_dividend'].fillna(0., inplace=True)

        df['winner_dividend'] = df.apply(lambda p: p['final_odds']*100. if p['winner_dividend'] == 0 and p['position'] == 1 else p['winner_dividend'], axis=1)

        df = append_hist(df, 6)

        df.date = df.date.astype('str')

        # append average speed, might be used as a target
        df['speed'] = (df['distance'] / df['time']).fillna(0)

        df['win'] = (df['position'] == 1).astype(np.float)

        df['victory_earnings'] = np.log(1+df['victory_earnings'].fillna(0))
        df['placed_earnings'] = np.log(1+df['placed_earnings'].fillna(0))
        df['prev_year_earnings'] = np.log(1+df['prev_year_earnings'].fillna(0))
        df['year_earnings'] = np.log(1+df['year_earnings'].fillna(0))
        df['handicap_distance'] = df['handicap_distance'].fillna(0.0)
        df['handicap_weight'] = df['handicap_weight'].fillna(0.0)
        df['prize'] = np.log(df['prize'].fillna(0)+1)

        # append last odds inverse, equivalent to the estimated probability of winning 
        df['final_odds_ref_inv'] = (1. / df['final_odds_ref']).fillna(0.)

        # features we want to append stats relative to the race (mean, std ...)

        for f in self.agg_features:
            df[f] = df[f].fillna(NAN_FLAG)

        races = df.groupby('race_id')
        stats = races[self.agg_features].agg([ f[1] for f in self.agg_features_funcs])
        # transform the 2 level columns index in 1 (victory_count, (mean, std)) -> (victory_count_mean, victory_count_std) 
        stats.columns = ['_'.join(col) for col in stats.columns.values]

        df = df.join(stats, how='left', on='race_id')

        for f in self.agg_features:
            df['{}_r'.format(f)] = (df[f] - df['{}_mean'.format(f)]) / df['{}_std'.format(f)]
            
        relative_features = ['{}_r'.format(f) for f in self.agg_features]

        df[relative_features] = df[relative_features].replace([np.inf, -np.inf], np.nan)
        df[relative_features] = df[relative_features].fillna(NAN_FLAG)

        # append sorted odds of the race
        odds = pd.DataFrame(columns=['odds_{:d}'.format(i) for i in range(20)], index=df.index)

        df['final_odds_ref'] = df['final_odds_ref'].fillna(NAN_FLAG)

        races = df.groupby('race_id')
        for (id, race) in races:
            odds_sorted = sorted(race['final_odds_ref'].values)[0:20]
            odds.loc[race.index, ['odds_{:d}'.format(i) for i, v in enumerate(odds_sorted)]] = odds_sorted

        df = pd.concat([df,odds], axis=1)

        df[['odds_{:d}'.format(i) for i in range(20)]] = df[['odds_{:d}'.format(i) for i in range(20)]].fillna(NAN_FLAG)

        self.players = df