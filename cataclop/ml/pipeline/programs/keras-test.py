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

        dataset = factories.Dataset.factory(self.name if kwargs.get('locked') else kwargs.get('dataset', 'default'), params=dataset_params, version='1.5')
        dataset.load(force=kwargs.get('dataset_reload', False))

        if self.model is None or mode != 'predict':
            model_params = {}
            if kwargs.get('model_params') is not None:
                model_params.update(kwargs.get('model_params'))

            self.model = factories.Model.factory(self.name if kwargs.get('locked') else kwargs.get('model', 'keras'), params=model_params, dataset=dataset, version='1.0')

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

        def fast_bet(r):
            p = 'score'

            s = r.sort_values(by=p, ascending=False)
            o = s.index.sort_values(ascending=True, return_indexer=True)
                
            s2 = r.sort_values(by='final_odds_ref')
            o2 = s2.index.sort_values(ascending=True, return_indexer=True)
            
            s3 = r.sort_values(by='final_odds')
            o3 = s3.index.sort_values(ascending=True, return_indexer=True)
                
            r['pn'] = o[1]
            r['oddsrn'] = o2[1]
            r['oddsn'] = o3[1]
            r['pstd'] = r[p].std()

            return r

        df = self.df

        df['pn'] = 0

        df = df.groupby('race_id').apply(fast_bet)

        df['bet'] = np.clip(df['oddsn'] + 1, 1, 3)
        df['profit'] = df['bet'] * (df['winner_dividend'].fillna(0.)/100.-1.0) + df['bet'] * (df['placed_dividend'].fillna(0.)/100.-1.0)
        
        bets = df[(df['score'] > 0) & (df['prize'] > 0.3) & (df['sub_category'] == 'HANDICAP')][['race_id', 'start_at', 'category', 'sub_category', 'country', 'profit', 'num', 'position', 'final_odds_ref', 'final_odds', 'score']].copy()
        bets['date'] = pd.to_datetime(bets['start_at'])

        self.bets = bets
        return self.bets

