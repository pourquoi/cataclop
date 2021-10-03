import logging

from cataclop.ml.pipeline import factories


class Model(factories.Model):

    def __init__(self, name, params=None, version=None, dataset=None):
        super().__init__(name=name, params=params, version=version)

        if dataset is None:
            raise ValueError('this model requires a dataset to initialise the features')

        self.dataset = dataset
        self.params['features'] = self.features
        self.params['categorical_features'] = self.categorical_features

        # this will be filled in train or load methods
        self.models = [{
            'name': 'dummy',
            'desc': 'always predict the horse number'
        }]

        self.logger = logging.getLogger(__name__)

    @property
    def defaults(self):
        return {
            'seed': 1234,
            'nan_flag': 0
        }

    @property
    def features(self):
        return ['num']

    @property
    def categorical_features(self):
        return []

    def load(self):
        pass

    def save(self, clear=False):
        pass

    def prepare_data(self, dataset, train=True):

        df = dataset.players

        features = self.features

        df = df.reset_index()

        df.loc[:, features] = df.loc[:, features].fillna(self.params['nan_flag'])

        df['position'].fillna(self.params['nan_flag'])
        df['pred_dummy_1'] = 0.0

        return df

    def train(self, dataset):

        self.models = []

        df = self.prepare_data(dataset, train=True)

        df['pred_num_1'] = df['num'].fillna(30)
        
        return df

    def predict(self, dataset, train=False):
        
        df = self.prepare_data(dataset, train=train)

        if len(df) == 0:
            return df

        df['pred_num_1'] = df['num'].fillna(30)

        return df



