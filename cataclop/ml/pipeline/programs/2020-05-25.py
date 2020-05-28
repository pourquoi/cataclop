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

        dataset_params = {
            #'from': '2017-01-01',
            #'to': '2020-01-01'
        }

        if kwargs.get('dataset_params') is not None:
            dataset_params.update(kwargs.get('dataset_params'))

        dataset = factories.Dataset.factory(self.name if kwargs.get('locked') else kwargs.get('dataset', 'default'), params=dataset_params, version='1.4')
        dataset.load(force=kwargs.get('dataset_reload', False))

        model_params = {
            'seed': 12345,
            'kfolds': 5,
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

    def check_race(self, race):
        return race.category == 'PLAT' and race.sub_category in ['HANDICAP', 'HANDICAP_DIVISE']

    def bet(self):

        df = self.df

        models = [{"name":'mlp_30'}]
        
        def fast_bet(r):
            for model in models:
                p = 'pred_{}_1'.format(model['name'])
                #print(model['name'], df['pred_{}_1'.format(model['name'])].mean())
                s = r.sort_values(by=p)
                o = s.index.sort_values(ascending=True, return_indexer=True)
                s2 = r.sort_values(by='final_odds_ref')
                o2 = s2.index.sort_values(ascending=True, return_indexer=True)


                idx = (r[p] == r[p].max())
            #idx = (r['pred_knn_5_1'] > 0) & (r['final_odds_ref'] > 5)
            #idx = (r['pred_knn_5_1'] > 0.) & (r['final_odds_ref'] > 5) & (r['final_odds_ref'] < 30)
                #if r[p].std() == 0:
                #    r['bet'] = 0
                #    return r
                r['bet_{}'.format(model['name'])] = np.clip(r[p], a_min=0., a_max=1.) #((idx).astype('float'))


                r['n_{}'.format(model['name'])] = o[1]
                r['n_odds_{}'.format(model['name'])] = o2[1]
            return r
        
        df = df[(df['country']=='FRA') & (df['sub_category'].isin(['HANDICAP', 'HANDICAP_DIVISE']))].copy()
        
        if len(df) == 0:
            self.bets = None
            return None

        df = df.groupby('race_id').apply(fast_bet)

        for model in models:
            m = model['name']
            #dd['profit_{}'.format(m)] = np.clip(dd['pred_{}_1'.format(m)], a_min=0., a_max=10.) * 1.0 * (dd['target_returns']-1.0)
            #dd['profit_{}'.format(m)] = 1.0 * (dd['target_returns']-1.0)
            df['bet_{}'.format(m)] = np.ceil(0.1 * np.clip((df['pred_{}_1'.format(m)]), a_min=0., a_max=10.) * np.log(df['n_odds_{}'.format(m)]+1.) )
            df['profit_{}'.format(m)] = df['bet_{}'.format(m)] * 1.0 * (df['target_returns']-1.0)

        df['bet'] = df[['bet_{}'.format(model['name']) for model in models]].sum(axis=1)
        df['profit'] = df[['profit_{}'.format(model['name']) for model in models]].sum(axis=1)
        df['target'] = 'mlp_30_1'
        
        bets = df[(df['pred_mlp_30_1'] >= 13.6) & (df['final_odds_ref_offline']>df['final_odds_offline']) & (df['final_odds_ref'] > 20) & (df['final_odds_ref']<50)][['race_id', 'start_at', 'bet', 'category', 'sub_category', 'country', 'profit', 'num', 'position', 'final_odds_ref', 'final_odds', 'profit_{}'.format(m), 'bet_{}'.format(m), 'pred_{}_1'.format(m)]].copy()
        bets['date'] = pd.to_datetime(bets['start_at'])

        self.bets = bets
        return self.bets




