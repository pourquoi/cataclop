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
            'categories': ['PLAT']
        }

        if kwargs.get('dataset_params') is not None:
            dataset_params.update(kwargs.get('dataset_params'))

        dataset = factories.Dataset.factory(self.name if kwargs.get('locked') else 'default', params=dataset_params, version='1.4')
        dataset.load(force=kwargs.get('dataset_reload', False))

        model_params = {
            'kfolds': 3,
            'nan_flag': 100000
        }

        if kwargs.get('model_params') is not None:
            model_params.update(kwargs.get('model_params'))

        self.model = factories.Model.factory(self.name if kwargs.get('locked') else 'stacked', params=model_params, dataset=dataset, version='1.0')

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
        return race.category == 'PLAT' and race.sub_category != 'COURSE_A_CONDITIONS' and race.session.hippodrome.country == 'FRA'

    def bet(self):

        N = 3
        max_odds = 30
        break_on_bet = False
        break_on_odds = False

        features = self.model.features
        categorical_features = self.model.categorical_features

        df = self.df

        df['pred_rnd'] = np.random.rand(df.shape[0])
        df['pred_sum'] = df[['pred_{}_1'.format(model['name']) for model in self.model.models ]].sum(axis=1)

        for model in self.model.models:
            df['pred_minus_ref_{}'.format(model['name'])] = df['pred_{}_1'.format(model['name'])] - df['final_odds_ref']
            
        for model in self.model.stacked_models:
            df['pred_minus_ref_stacked_{}'.format(model['name'])] = df['pred_stacked_{}_1'.format(model['name'])] - df['final_odds_ref']

        targets = ['pred_rnd', 'final_odds_ref', 'pred_sum']
        targets += ['pred_{}_1'.format(model['name']) for model in self.model.models]
        targets += ['pred_stacked_{}_1'.format(model['name']) for model in self.model.stacked_models]
        targets += ['pred_minus_ref_stacked_{}'.format(model['name']) for model in self.model.stacked_models]
        targets += ['pred_minus_ref_{}'.format(model['name']) for model in self.model.models]

        races = df.sort_values('start_at').groupby('race_id')

        bets = []

        for (id, race) in races:

            candidate_bets = []

            nums = []

            for target in targets:

                r = race.sort_values(by=target, ascending=False)

                if len(r) <= N:
                    break
                    
                NN = N

                for n in range(NN):

                    player = r.iloc[n]

                    odds = player['final_odds_ref']

                    if max_odds is not None and odds > max_odds:
                        if break_on_odds:
                            break
                        else:
                            continue

                    #nth = (r['final_odds_ref']<odds).sum()+1
                    
                    if player[target] < 0:
                        break

                    bet = np.clip(player[target]/100.0, 0, 10)
                    
                    bet = np.round(1+bet) * 1.5
                    
                    if bet <= 0:
                        break
                    
                    if n+1 < len(r) and r.iloc[n+1][target] == player[target]:
                        NN = NN+1
                        
                    bet = 1

                    profit = player['winner_dividend']/100.0 * bet - bet

                    row = [id, player['start_at'], player['num'], player['position'], n, odds, player['final_odds'], target, player[target], r[target].std(), bet, profit]

                    for nn in range(1,4):
                        if n+nn < len(r):
                            row.append(r.iloc[n+nn][target])
                        else:
                            row.append(np.nan)

                    for f in features:
                        row.append(player[f])
                    for f in categorical_features:
                        row.append(player[f])

                    candidate_bets.append( row )

                    nums.append(player['num'])

                    if break_on_bet:
                        break

            #if len(candidate_bets) == 1:
            #    bets += candidate_bets
            bets += candidate_bets

        cols = ['id', 'date', 'num', 'pos', 'nb', 'odds_ref', 'odds_final', 'target', 'pred', 'pred_std', 'bet', 'profit']

        for nn in range(1,4):
            cols.append('next_pred_{}'.format(nn))

        cols = cols + features + categorical_features

        bets = pd.DataFrame(bets, columns=cols)
        bets['date'] = pd.to_datetime(bets['date'])

        bets = bets.set_index(bets['date'])

        bets = bets.sort_index()

        bets['bets'] = bets['bet'].cumsum()
        bets['stash'] = bets['profit'].cumsum()

        bb = bets[ (bets['country']=='FRA') & (bets['sub_category']!='COURSE_A_CONDITIONS') & (bets['nb']==0) & ( (bets['target']=='pred_knn_10_1') | (bets['target']=='pred_minus_ref_lasso_1') ) & (bets['odds_ref']>10) & (bets['odds_ref']<20) & (bets['pred']>20)].copy()        

        self.bets = bb

        return self.bets





