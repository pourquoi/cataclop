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
from xgboost import XGBRegressor

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
        self.models = []

        self.logger = logging.getLogger(__name__)

    @property
    def defaults(self):
        return {
            'seed': 1234,
            'kfolds': 2,
            'n_targets': 1,
            'nan_flag': +10000
        }

    @property
    def features(self):
        features = ['prize', 'declared_player_count', 'final_odds_ref', 'final_odds_ref_unibet']

        features += ['odds_{:d}'.format(i) for i in range(10)]

        features += ['hist_{}_pos'.format(i+1) for i in range(6)]

        features += self.dataset.agg_features

        for f in self.dataset.agg_features:
            features.append('{}_r'.format(f))
            for s in self.dataset.agg_features_funcs:
                features.append('{}_{}'.format(f, s[0]))

        return sorted(list(set(features)))

    @property
    def categorical_features(self):
        return ['category', 'sub_category', 'country']

    def load(self):
        self.models = load(os.path.join(self.data_dir, 'models.joblib'))

    def save(self, clear=False):

        d = self.data_dir

        if not os.path.isdir(d):
            os.makedirs(d)
        elif clear:
            shutil.rmtree(d)
            os.makedirs(d)

        path = os.path.join(d, 'models.joblib')

        if os.path.isfile(path):
            os.remove(path)

        dump(self.models, path)

    def prepare_data(self, dataset, train=True):
        self.logger.debug('preparing model data')
        df = dataset.players

        features = self.features

        # keep only interesting races: at least one winner, with reference odds
        if train:
            df = df.groupby('race_id').filter(lambda r: r['position'].min() == 1 and r['winner_dividend'].max() > 0 and r['odds_0'].min() != dataset.params['nan_flag'] and r['odds_1'].min() != dataset.params['nan_flag'] )

        df = df.reset_index()

        df.loc[:, features] = df.loc[:, features].fillna(self.params['nan_flag'])

        df['position'].fillna(self.params['nan_flag'])

        #df['target'] = df['speed'].astype('float').replace(1000, self.params['nan_flag'])

        df['target_returns'] = df['winner_dividend'] / 100.
        df['target_returns'].fillna(0, inplace=True)

        df['target'] = df['position'].fillna(self.params['nan_flag'])

        df['target'] = df['target_returns']

        for model in self.models:
            for i in range(self.params['n_targets']):
                df['pred_{}_{}'.format(model['name'], i+1)] = 0.0

        return df

    def train(self, dataset):

        features = self.features
        categorical_features = self.categorical_features

        self.models = []

        for n in [30]:
            self.models.append(
                {
                    'name': 'xgb_{}'.format(n),
                    'steps': [XGBRegressor(n_estimators=n, missing=self.params['nan_flag'], random_state=self.params['seed'])],
                    'estimators': []
                }
            )
        
        '''
        for a in [1]:
            self.models.append(
                {
                    'name': 'ridge_{}'.format(a),
                    'steps': [RobustScaler(), Ridge(alpha=a)],
                    'estimators': []
                }
            )

        for a in [1]:
            self.models.append(
                {
                    'name': 'lasso_{}'.format(a),
                    'steps': [RobustScaler(), Lasso(alpha=a)],
                    'estimators': []
                }
            )

        self.models.append({
            'name': 'svr',
            'steps': [RobustScaler(), svm.LinearSVR()],
            'estimators': []
        })
        '''

        for n in [1, 2, 5, 10]:

            self.models.append(
                {
                    'name': 'knn_{}'.format(n),
                    'steps': [RobustScaler(), KNeighborsRegressor(n_neighbors=n)],
                    'estimators': []
                }
            )

        

        for n in [1, 10, 30, 100]:
            self.models.append(
                {
                    'name': 'mlp_{}'.format(n),
                    'steps': [RobustScaler(), MLPRegressor(activation='relu', hidden_layer_sizes=(n,), random_state=self.params['seed'])],
                    'estimators': []
                }
            )

        '''
        for n in [10, 20, 30, 40, 100]:
            self.models.append(
                {
                    'name': 'gbr_{}'.format(n),
                    'steps': [GradientBoostingRegressor(n_estimators=n)],
                    'estimators': []
                }
            )

        for n in [100]:
            self.models.append(
                {
                    'name': 'rf_{}'.format(n),
                    'steps': [RandomForestRegressor(n_estimators=n, random_state=self.params['seed'])],
                    'estimators': []
                }
            )
        '''

        df = self.prepare_data(dataset)

        groups = df['race_id'].values

        group_kfold = GroupKFold(n_splits=self.params['kfolds'])

        splits = list(group_kfold.split(df.values, df['position'].values, groups))

        for train_index, test_index in splits:
            
            for model in self.models:

                self.logger.debug('training {}'.format(model['name']))

                X_train = df[features].iloc[train_index].copy()
                y_train = df['target'].iloc[train_index]

                dummies = preprocessing.get_dummies(df.iloc[train_index], categorical_features)
                X_train = pd.concat([X_train, preprocessing.get_dummy_values(df.iloc[train_index], dummies)], axis=1)

                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) & (df.iloc[train_index]['category'] != 'CfOURSE_A_CONDITIONS') & (df.iloc[train_index]['final_odds_ref'] < 20) & ((df.iloc[train_index]['position'] == 1) | (df.iloc[train_index]['position'] == 3) | (df.iloc[train_index]['position'] == self.params['nan_flag'])) 
                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) & (df.iloc[train_index]['final_odds_ref'] < 30) & ((df.iloc[train_index]['position'] == 1) | (df.iloc[train_index]['position'] == 2))
                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) & (df.iloc[train_index]['final_odds_ref'] < 20) & ((df.iloc[train_index]['position'] == 1) | (df.iloc[train_index]['position'] > 3) | (df.iloc[train_index]['position'] == self.params['nan_flag']) )

                idx = (df.iloc[train_index]['target'] != self.params['nan_flag'])
                X_train = X_train[ idx ]
                y_train = df['target'].iloc[train_index][ idx ]

                X_test = df[features].iloc[test_index].copy()
                y_test = df['target'].iloc[test_index]

                X_test = pd.concat([X_test, preprocessing.get_dummy_values(df.iloc[test_index], dummies)], axis=1)
            
                X_train = X_train.values
                X_test = X_test.values
            
                idx = df.iloc[test_index].index

                steps = [ clone(step) for step in model['steps']]

                pipeline = make_pipeline(*steps)

                pipeline.fit(X_train, y_train.values)

                if self.params['n_targets'] > 1:

                    clf = pipeline.steps[-1][1]
                    
                    p = pipeline.predict_proba(X_test)

                    for i in range(self.params['n_targets']):
                        df.loc[idx, 'pred_{}_{}'.format(model['name'], i+1)] = p[:, list(clf.classes_).index(i+1)]
                    
                else:

                    p = pipeline.predict(X_test)

                    df.loc[idx, 'pred_{}_1'.format(model['name'])] = p

                    self.logger.debug( 'mea: {}'.format(mean_absolute_error(y_test.values, p)) )

                model['estimators'].append({
                    'pipeline': pipeline,
                    'dummies': dummies
                })

        return df

    def predict(self, dataset):
        
        df = self.prepare_data(dataset, train=False)

        if len(df) == 0:
            return df

        for model in self.models:

            for estimator in model['estimators']:

                X = df[self.features].copy()
                X = pd.concat([X, preprocessing.get_dummy_values(df, estimator['dummies'])], axis=1)
                X = X.values
                y = df['target'].values

                if self.params['n_targets'] > 1:
                    p = estimator['pipeline'].predict_proba(X)

                    clf = estimator['pipeline'].steps[-1][1]

                    for i in self.params['n_targets']:
                        df['pred_{}_{}'.format(model['name'], i+1)] += p[:, list(clf.classes_).index(i+1)]

                else:

                    p = estimator['pipeline'].predict(X)

                    df['pred_{}_1'.format(model['name'])] += p

            n_estimators = len(model['estimators'])

            if n_estimators:

                if self.params['n_targets'] > 1:
                    for i in self.params['n_targets']:
                        df['pred_{}_{}'.format(model['name'], i+1)] /= n_estimators
                else:
                    df['pred_{}_1'.format(model['name'])] /= n_estimators
        
        return df



