import pandas as pd
import numpy as np

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

    def run(self, mode='predict', **kwargs):

        dataset_params = {}

        if mode == 'train':
            dataset_params = {
                'from': '2016-01-01',
                'to': '2018-11-23',
                'categories': ['PLAT']
            }

        if kwargs.get('dataset_params') is not None:
            dataset_params.update(kwargs.get('dataset_params'))

        dataset = factories.Dataset.factory('2018-12-25_sum', params=dataset_params, version='1.0')
        dataset.load(force=kwargs.get('dataset_reload', False))

        model_params = {}
        if kwargs.get('model_params') is not None:
            model_params.update(kwargs.get('model_params'))

        self.model = factories.Model.factory('2018-12-25_sum', params=model_params, dataset=dataset, version='1.0')

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
        self._bet()

        b = self.bets
        b = b[ (b['target'] == 'pred_sum') & (b['pred'] > 313.) & (b['sub_category'] == 'COURSE_A_CONDITIONS') & (b['pred_std'] > 0) & (b['odds_ref'] < 20) & (b['declared_player_count'] > 1) ].copy()

        self.bets = b

    def _bet(self, targets=None, N=1, max_odds=30, break_on_bet=True, break_on_odds=True):

        features = self.model.features
        categorical_features = self.model.categorical_features

        if targets is None:
            targets = ['pred_sum']

        self.df['pred_sum'] = self.df.loc[:, ['pred_{}_1'.format(model['name']) for model in self.model.models]].sum(axis=1)

        races = self.df.sort_values('start_at').groupby('race_id')

        bets = []

        for (id, race) in races:
            
            candidate_bets = []
            
            nums = []
            
            for target in targets:

                r = race.sort_values(by=target, ascending=False)

                if len(r) <= N:
                    break

                for n in range(N):

                    player = r.iloc[n]

                    odds = player['final_odds_ref']

                    if max_odds is not None and odds > max_odds:
                        if break_on_odds:
                            break
                        else:
                            continue

                    if player[target] < 0:
                        break

                    bet = np.clip(player[target]/100.0, 0, 10)
                    
                    bet = np.round(1+bet) * 1.5
                    
                    if bet <= 0:
                        break

                    profit = player['winner_dividend']/100.0 * bet - bet

                    row = [id, player['date'], player['num'], odds, player['final_odds'], target, player[target], r[target].std(), bet, profit]

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

        cols = ['id', 'date', 'num', 'odds_ref', 'odds_final', 'target', 'pred', 'pred_std', 'bet', 'profit']

        for nn in range(1,4):
            cols.append('next_pred_{}'.format(nn))
        
        cols = cols + features + categorical_features

        bets = pd.DataFrame(bets, columns=cols)

        bets.index = bets['date']

        bets = bets.sort_index()

        bets['bets'] = bets['bet'].cumsum()
        bets['stash'] = bets['profit'].cumsum()

        self.bets = bets




