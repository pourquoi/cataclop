import pandas as pd

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
        pass

    def run(self, mode='predict', **kwargs):

        dataset_params = {}

        if kwargs.get('dataset_params') is not None:
            dataset_params.update(kwargs.get('dataset_params'))

        dataset = factories.Dataset.factory('dummy',
                                            params=dataset_params,
                                            version='1.0')
        dataset.load(force=kwargs.get('dataset_reload', False))

        model_params = {}

        if kwargs.get('model_params') is not None:
            model_params.update(kwargs.get('model_params'))

        self.model = factories.Model.factory('dummy',
                                             params=model_params,
                                             dataset=dataset,
                                             version='1.0')

        if mode == 'predict':
            self.df = self.model.train(dataset)
        else:
            raise ValueError("unknown mode [{}]".format(mode))

        self.dataset = dataset

    def train(self, **kwargs):
        return self.run('train', **kwargs)

    def predict(self, **kwargs):
        return self.run('predict', **kwargs)

    def check_race(self, race):
        return True

    def bet(self):

        if self.df is None:
            raise RuntimeError("Call the predict method before betting")

        df = self.df
        races = df.sort_values('start_at').groupby('race_id')

        bets = []

        for (race_id, race) in races:

            r = race.sort_values(by='pred_dummy_1', ascending=True)

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
