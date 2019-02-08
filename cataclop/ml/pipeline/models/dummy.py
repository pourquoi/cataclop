import os
import shutil
import logging

import pandas as pd
import numpy as np

from joblib import dump, load

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, MinMaxScaler, RobustScaler

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone as clone
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBRegressor, XGBClassifier

from cataclop.ml.exploration import random_race
from cataclop.ml import preprocessing
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
            'name': 'num'
        }]

        self.logger = logging.getLogger(__name__)

    @property
    def defaults(self):
        return {
            'seed': 1234,
            'kfolds': 2,
            'n_targets': 1,
            'n_targets_stacked': 1,
            'nan_flag': +10000
        }

    @property
    def features(self):
        features = ['prize', 'declared_player_count', 'num']

        features += ['odds_{:d}'.format(i) for i in range(10)]

        features += ['hist_{}_pos'.format(i+1) for i in range(6)]

        agg_features = self.dataset.agg_features
        #agg_features = [ f for f in self.dataset.agg_features if f not in ['final_odds_ref', 'final_odds_ref_offline'] ]

        features += agg_features

        for f in agg_features:
            features.append('{}_r'.format(f))
            for s in self.dataset.agg_features_funcs:
                features.append('{}_{}'.format(f, s[0]))

        return sorted(list(set(features)))

    @property
    def categorical_features(self):
        return ['category', 'sub_category', 'country']

    def load(self):
        pass

    def save(self, clear=False):
        pass

    def prepare_data(self, dataset, train=True):
        self.logger.debug('preparing model data')
        df = dataset.players

        features = self.features

        df = df.reset_index()

        df.loc[:, features] = df.loc[:, features].fillna(self.params['nan_flag'])

        df['position'].fillna(self.params['nan_flag'])

        for model in self.models:
            for i in range(self.params['n_targets']):
                df['pred_{}_{}'.format(model['name'], i+1)] = 0.0

        return df

    def train(self, dataset):

        features = self.features
        categorical_features = self.categorical_features

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



