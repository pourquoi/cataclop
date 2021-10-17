import pandas as pd
import numpy as np

import datetime
import os
import shutil

from cataclop.ml.pipeline import factories

class Program(factories.Program):

    def __init__(self, name, params=None, version=None):
        super().__init__(name=name, params=params, version=version)

        self.dataset = None
        self.bets = None
        self.model = None
        self.df = None

    @property
    def defaults(self):
        return {}

    def lock(self, key=None):
        """Make a copy of the program, model and dataset.
        """

        if self.dataset is None:
            raise ValueError('The program\'s dataset is required before locking')

        if self.model is None:
            raise ValueError('The program\'s model is required before locking')

        if key is None:
            key = datetime.datetime.now().strftime('%Y-%m-%d')

        pipeline_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        shutil.copy( os.path.join(pipeline_path, 'datasets', self.dataset.name + '.py'), os.path.join(pipeline_path, 'datasets', key + '.py') )
        shutil.copy( os.path.join(pipeline_path, 'models', self.model.name + '.py'), os.path.join(pipeline_path, 'models', key + '.py') )
        shutil.copy( os.path.join(pipeline_path, 'programs', self.name + '.py'), os.path.join(pipeline_path, 'programs', key + '.py') )

        self.dataset.name = key
        self.model.name = key

        self.dataset.save()
        self.model.save()

    def run(self, mode='predict', **kwargs):

        dataset_params = {}

        if kwargs.get('dataset_params') is not None:
            dataset_params.update(kwargs.get('dataset_params'))

        dataset = factories.Dataset.factory(self.name if kwargs.get('locked') else kwargs.get('dataset', 'default'), params=dataset_params, version='1.4')
        dataset.load(force=kwargs.get('dataset_reload', False))

        model_params = {
            'seed': 123456,
            'kfolds': 2,
            'nan_flag': -10000,
            'n_targets': 1,
        }
        if kwargs.get('model_params') is not None:
            model_params.update(kwargs.get('model_params'))

        self.model = factories.Model.factory(self.name if kwargs.get('locked') else kwargs.get('model', 'default'), params=model_params, dataset=dataset, version='1.0')

        if mode == 'train':
            self.df = self.model.train(dataset)
        elif mode == 'predict':
            self.model.load()
            self.df = self.model.predict(dataset)
        else:
            pass

        self.dataset = dataset

    def train(self, **kwargs):
        return self.run('train', **kwargs)

    def predict(self, **kwargs):
        return self.run('predict', **kwargs)

    def bet(self):

        if self.df is None:
            raise RuntimeError("Call the predict method before betting")

        df = self.df
        races = df.sort_values('start_at').groupby('race_id')

        bets = []

        for (race_id, race) in races:

            r = race.sort_values(by='pred', ascending=True)

            player = r.iloc[0]

            bet = 1

            profit = player['winner_dividend'] / 100.0 * bet - bet
            bets += [{
                "id": id,
                "date": player['start_at'],
                "num": player['num'],
                "position": player['position'],
                "odds": player['final_odds'],
                "bet": bet,
                "profit": profit
            }]

        bets = pd.DataFrame(bets)
        bets['date'] = pd.to_datetime(bets['date'])

        bets = bets.set_index(bets['date'])
        bets = bets.sort_index()

        bets['bets'] = bets['bet'].cumsum()
        bets['stash'] = bets['profit'].cumsum()

        self.bets = bets.copy()

        return self.bets




