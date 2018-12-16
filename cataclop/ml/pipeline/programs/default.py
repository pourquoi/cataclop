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

        if kwargs.get('dataset_params') is not None:
            dataset_params.update(kwargs.get('dataset_params'))

        dataset = factories.Dataset.factory('default', params=dataset_params, version='1.4')
        dataset.load(force=kwargs.get('dataset_reload', False))

        model_params = {}
        if kwargs.get('model_params') is not None:
            model_params.update(kwargs.get('model_params'))

        self.model = factories.Model.factory('default', params=model_params, dataset=dataset, version='1.9')

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

    def bet(self, targets=None, N=1, max_odds=20, break_on_bet=True, break_on_odds=False):

        features = self.model.features
        categorical_features = self.model.categorical_features

        if targets is None:
            targets = ['pred_{}_1'.format(model['name']) for model in self.model.models] + ['pred_sum']

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

                    #nth = (r['final_odds_ref']<odds).sum()+1

                    bet = 1.

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




