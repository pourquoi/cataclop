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
from .default import Model as DefaultModel

class Model(DefaultModel):

    def __init__(self, name, params=None, version=None, dataset=None):
        super().__init__(name=name, params=params, version=version, dataset=dataset)

        self.stacked_models = []


    @property
    def defaults(self):
        return {**(super().defaults), 'n_targets_stacked': 1}

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

    @property
    def stacked_features(self):
        stacked_features = self.features + ['final_odds_ref', 'final_odds_ref_offline']
        if self.params['n_targets'] > 1:
            for n in range(self.params['n_targets']):
                stacked_features = stacked_features + ['pred_{}_{}'.format(model['name'], n+1) for model in self.models]
                stacked_features = stacked_features + ['pred_{}_{}_std'.format(model['name'], n+1) for model in self.models]
                stacked_features = stacked_features + ['pred_{}_{}_min'.format(model['name'], n+1) for model in self.models]
                stacked_features = stacked_features + ['pred_{}_{}_max'.format(model['name'], n+1) for model in self.models]
        else:
            stacked_features = stacked_features + ['pred_{}_1'.format(model['name']) for model in self.models]
            stacked_features = stacked_features + ['pred_{}_std'.format(model['name']) for model in self.models]
            stacked_features = stacked_features + ['pred_{}_min'.format(model['name']) for model in self.models]
            stacked_features = stacked_features + ['pred_{}_max'.format(model['name']) for model in self.models]

        return stacked_features

    def load(self):
        self.models = load(os.path.join(self.data_dir, 'models.joblib'))
        self.stacked_models = load(os.path.join(self.data_dir, 'stacked_models.joblib'))

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

        path = os.path.join(d, 'stacked_models.joblib')
        if os.path.isfile(path):
            os.remove(path)
        dump(self.stacked_models, path)

    def prepare_data(self, dataset, train=True):
        self.logger.debug('preparing model data')
        df = dataset.players

        features = self.features

        # keep only interesting races: at least one winner, with reference odds
        if train:
            df = df.groupby('race_id').filter(lambda r: r['position'].min() == 1 and r['winner_dividend'].max() > 0 and r['odds_0'].min() != dataset.params['nan_flag'] and r['odds_1'].min() != dataset.params['nan_flag'] )

        df = df.reset_index()

        races = df.groupby('race_id')

        '''
        strats = [
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]

        strat_df = {
            'race_id': [],
            'strat': [],
        }

        for n in range(len(strats)):
            strat_df['strat_{}'.format(n)] = []

        for (id, race) in races:

            r = race.sort_values(by='final_odds_ref', ascending=True)

            strat_df['race_id'].append(id)

            max_s = (0, 0)

            for (si, strat) in enumerate(strats):

                s = 0

                for n in range(len(strat)):
                    if len(r) <= n:
                        break
                    s += r.iloc[n]['winner_dividend']/100.0 * strat[n] - strat[n]

                if max_s[0] < s:
                    max_s = (s, si+1)

                strat_df['strat_{}'.format(si)].append(s)

            strat_df['strat'].append(max_s[1])

        strat_df = pd.DataFrame(strat_df)

        df = pd.merge(left=df, right=strat_df, on='race_id', how='left')
        '''

        df.loc[:, features] = df.loc[:, features].fillna(self.params['nan_flag'])

        df['position'].fillna(self.params['nan_flag'])

        #df['target'] = df['speed'].astype('float').replace(1000, self.params['nan_flag'])

        df['target_returns'] = df['winner_dividend'] / 100.
        df['target_returns'].fillna(0, inplace=True)
        df['target_returns'] = df['target_returns']

        #df['target_pos'] = np.clip(df['position'].fillna(df['declared_player_count']), 1, 2)

        df['target_pos'] = df.apply(lambda p: 1 if p['position'] >= 1 and p['position'] <= 2 else 2 if p['position'] > 2 and p['position'] <= 4 else 3, axis=1)

        #df['target_pos'] = df['position'].fillna(df['declared_player_count']) / df['declared_player_count']

        df['target_odds'] = df['final_odds'].fillna(self.params['nan_flag'])

        df['target'] = np.log(1.+df['target_returns'])

        for model in self.models:
            for i in range(self.params['n_targets']):
                df['pred_{}_{}'.format(model['name'], i+1)] = 0.0

        for model in self.stacked_models:
            for i in range(self.params['n_targets']):
                df['pred_stacked_{}_{}'.format(model['name'], i+1)] = 0.0

        return df

    def prepared_stacked_data(self, df):
        self.logger.debug('preparing stacked model data')
        races = df.groupby('race_id')
        for (id, race) in races:
            for model in self.models:
                if self.params['n_targets'] > 1:
                    for n in range(self.params['n_targets']):
                        df.loc[race.index, 'pred_{}_{}_std'.format(model['name'], n+1)] = race['pred_{}_{}'.format(model['name'], n+1)].std()
                        df.loc[race.index, 'pred_{}_{}_min'.format(model['name'], n+1)] = race['pred_{}_{}'.format(model['name'], n+1)].min()
                        df.loc[race.index, 'pred_{}_{}_max'.format(model['name'], n+1)] = race['pred_{}_{}'.format(model['name'], n+1)].max()
                else:
                    df.loc[race.index, 'pred_{}_std'.format(model['name'])] = race['pred_{}_1'.format(model['name'])].std()
                    df.loc[race.index, 'pred_{}_min'.format(model['name'])] = race['pred_{}_1'.format(model['name'])].min()
                    df.loc[race.index, 'pred_{}_max'.format(model['name'])] = race['pred_{}_1'.format(model['name'])].max()

        return df

    def train(self, dataset):

        features = self.features
        categorical_features = self.categorical_features

        self.models = []

        
        for n in [10, 30, 100]:
            if self.params['n_targets'] > 1:
                self.models.append(
                    {
                        'name': 'xgb_{}'.format(n),
                        'steps': [XGBClassifier(max_depth=3, n_estimators=n, missing=self.params['nan_flag'], random_state=self.params['seed'])],
                        'estimators': []
                    }
                )
            else:
                self.models.append(
                    {
                        'name': 'xgb_{}'.format(n),
                        'steps': [XGBRegressor(max_depth=3, n_estimators=n, missing=self.params['nan_flag'], random_state=self.params['seed'])],
                        'estimators': []
                    }
                )
        

        '''
        for a in [0.1, 1]:
            self.models.append(
                {
                    'name': 'ridge_{}'.format(a),
                    'steps': [RobustScaler(), Ridge(alpha=a)],
                    'estimators': []
                }
            )

        for a in [0.1, 1]:
            self.models.append(
                {
                    'name': 'lasso_{}'.format(a),
                    'steps': [RobustScaler(), Lasso(alpha=a)],
                    'estimators': []
                }
            )
        '''

        '''
        self.models.append({
            'name': 'svr',
            'steps': [RobustScaler(), svm.LinearSVR()],
            'estimators': []
        })
        '''

        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]:

            if self.params['n_targets'] > 1:
                self.models.append(
                    {
                        'name': 'knn_{}'.format(n),
                        'steps': [RobustScaler(), KNeighborsClassifier(n_neighbors=n, weights='uniform')],
                        'estimators': []
                    }
                )
            else:
                self.models.append(
                    {
                        'name': 'knn_{}'.format(n),
                        'steps': [RobustScaler(), KNeighborsRegressor(n_neighbors=n, weights='uniform')],
                        'estimators': []
                    }
                )

        '''
        for n in [1, 20, 30, 100]:
            self.models.append(
                {
                    'name': 'mlp_{}'.format(n),
                    'steps': [RobustScaler(), MLPRegressor(activation='relu', hidden_layer_sizes=(n,), random_state=self.params['seed'])],
                    'estimators': []
                }
            )
        

        for n in [10, 30, 100]:
            self.models.append(
                {
                    'name': 'gbr_{}'.format(n),
                    'steps': [GradientBoostingRegressor(n_estimators=n)],
                    'estimators': []
                }
            )

        for n in [10, 30, 100]:
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

                dummies = preprocessing.get_dummies(df.iloc[train_index], categorical_features)
                X_train = pd.concat([X_train, preprocessing.get_dummy_values(df.iloc[train_index], dummies)], axis=1)

                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) & (df.iloc[train_index]['category'] != 'CfOURSE_A_CONDITIONS') & (df.iloc[train_index]['final_odds_ref'] < 20) & ((df.iloc[train_index]['position'] == 1) | (df.iloc[train_index]['position'] == 3) | (df.iloc[train_index]['position'] == self.params['nan_flag'])) 
                #idx = (df.iloc[train_index]['target_pos'] != self.params['nan_flag']) & (df.iloc[train_index]['final_odds_ref'] < 30) 
                idx = (df.iloc[train_index]['target_returns'] != self.params['nan_flag'])
                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) & (df.iloc[train_index]['final_odds_ref'] < 20) & ((df.iloc[train_index]['position'] == 1) | (df.iloc[train_index]['position'] > 3) | (df.iloc[train_index]['position'] == self.params['nan_flag']) )
                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) 
                X_train = X_train[ idx ]
                y_train = df['target_returns'].iloc[train_index][ idx ]

                X_test = df[features].iloc[test_index].copy()
                y_test = df['target_returns'].iloc[test_index]

                X_test = pd.concat([X_test, preprocessing.get_dummy_values(df.iloc[test_index], dummies)], axis=1)
            
                X_train = X_train.values
                X_test = X_test.values
            
                idx = df.iloc[test_index].index

                steps = [ clone(step) for step in model['steps']]

                pipeline = make_pipeline(*steps)

                pipeline.fit(X_train, y_train.values)

                if self.params['n_targets'] > 1:

                    clf = pipeline.steps[-1][1]

                    p = pipeline.predict(X_test)

                    df.loc[idx, 'pred_{}'.format(model['name'])] = p

                    p = pipeline.predict_proba(X_test)

                    for i in range(self.params['n_targets']):
                        if (i+1) in clf.classes_:
                            df.loc[idx, 'pred_{}_{}'.format(model['name'], i+1)] = p[:, list(clf.classes_).index(i+1)]
                        else:
                            df.loc[idx, 'pred_{}_{}'.format(model['name'], i+1)] = np.zeros(len(idx))
                else:

                    p = pipeline.predict(X_test)

                    df.loc[idx, 'pred_{}_1'.format(model['name'])] = p

                    self.logger.debug( 'mea: {}'.format(mean_absolute_error(y_test.values, p)) )

                model['estimators'].append({
                    'pipeline': pipeline,
                    'dummies': dummies
                })

        df = self.prepared_stacked_data(df)

        self.stacked_models = []

        '''
        self.stacked_models.append(
            {
                'name': 'knn',
                'steps': [RobustScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance')],
                'estimators': []
            }
        )
        '''

        self.stacked_models.append({
            'name': 'xgb',
            'steps': [XGBRegressor(missing=self.params['nan_flag'], random_state=self.params['seed'])],
            'estimators': []
        })

        '''
        self.stacked_models.append({
            'name': 'ridge',
            'steps': [RobustScaler(), Ridge()],
            'estimators': []
        })

        self.stacked_models.append({
            'name': 'lasso',
            'steps': [RobustScaler(), Lasso()],
            'estimators': []
        })
        '''

        self.stacked_models.append({
            'name': 'mlp_sigmoid',
            'steps': [RobustScaler(), MLPRegressor(activation='logistic', hidden_layer_sizes=(10,), random_state=self.params['seed'])],
            'estimators': []
        })

        self.stacked_models.append({
            'name': 'mlp_sigmoid_100',
            'steps': [RobustScaler(), MLPRegressor(activation='logistic', hidden_layer_sizes=(100,), random_state=self.params['seed'])],
            'estimators': []
        })

        self.stacked_models.append({
            'name': 'mlp_relu_100',
            'steps': [RobustScaler(), MLPRegressor(activation='relu', hidden_layer_sizes=(100,), random_state=self.params['seed'])],
            'estimators': []
        })

        for train_index, test_index in splits:

            for model in self.stacked_models:

                self.logger.debug('training {}'.format(model['name']))

                X_train = df[self.stacked_features].iloc[train_index].copy()

                #idx = (df.iloc[train_index]['target'] != self.params['nan_flag']) & (df.iloc[train_index]['final_odds_ref'] < 30)
                #idx = (df.iloc[train_index]['target_returns'] != self.params['nan_flag']) & (df.iloc[train_index]['country'] == 'FRA') & (df.iloc[train_index]['sub_category'] == 'HANDICAP') & (df.iloc[train_index]['final_odds_ref'] > 0 ) & ((df.iloc[train_index]['position'] <= 10) )
                idx = (df.iloc[train_index]['target_returns'] != self.params['nan_flag']) 
                y_train = df['target_returns'].iloc[train_index]

                dummies = preprocessing.get_dummies(df.iloc[train_index], self.categorical_features)
                X_train = pd.concat([X_train, preprocessing.get_dummy_values(df.iloc[train_index], dummies)], axis=1)

                X_train = X_train[idx]
                y_train = y_train[idx]

                X_test = df[self.stacked_features].iloc[test_index].copy()
                y_test = df['target_returns'].iloc[test_index]

                X_test = pd.concat([X_test, preprocessing.get_dummy_values(df.iloc[test_index], dummies)], axis=1)

                X_train = X_train.values
                X_test = X_test.values

                steps = [ clone(step) for step in model['steps']]

                pipeline = make_pipeline(*steps)

                pipeline.fit(X_train, y_train.values)

                idx = df.iloc[test_index].index

                if self.params['n_targets_stacked'] > 1:

                    clf = pipeline.steps[-1][1]

                    p = pipeline.predict(X_test)

                    df.loc[idx, 'pred_stacked_{}'.format(model['name'])] = p
                    
                    p = pipeline.predict_proba(X_test)

                    for i in range(self.params['n_targets_stacked']):
                        if i in clf.classes_:
                            df.loc[idx, 'pred_stacked_{}_{}'.format(model['name'], i+1)] = p[:, list(clf.classes_).index(i)]
                        else:
                            df.loc[idx, 'pred_stacked_{}_{}'.format(model['name'], i+1)] = np.zeros(len(idx))
                    
                else:

                    p = pipeline.predict(X_test)

                    df.loc[idx, 'pred_stacked_{}_1'.format(model['name'])] = p

                    self.logger.debug( 'mea: {}'.format(mean_absolute_error(y_test.values, p)) )

                model['estimators'].append({
                    'pipeline': pipeline,
                    'dummies': dummies
                })

        
        
        '''
        for model in self.models:
            for i in range(self.params['n_targets']):
                df['pred_{}_{}'.format(model['name'], i+1)] = -df['pred_{}_{}'.format(model['name'], i+1)]
        '''
        

        return df

    def predict(self, dataset, train=False):
        
        df = self.prepare_data(dataset, train=train)

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

                    for i in range(self.params['n_targets']):
                        if (i+1) in clf.classes_:
                            df['pred_{}_{}'.format(model['name'], i+1)] = p[:, list(clf.classes_).index(i+1)]
                        else:
                            df['pred_{}_{}'.format(model['name'], i+1)] = np.zeros(len(df))

                else:

                    p = estimator['pipeline'].predict(X)

                    df['pred_{}_1'.format(model['name'])] += p

            n_estimators = len(model['estimators'])

            if n_estimators:

                if self.params['n_targets'] > 1:
                    for i in range(self.params['n_targets']):
                        df['pred_{}_{}'.format(model['name'], i+1)] /= n_estimators
                else:
                    df['pred_{}_1'.format(model['name'])] /= n_estimators

        df = self.prepared_stacked_data(df)
        

        for model in self.stacked_models:

            for estimator in model['estimators']:

                X = df[self.stacked_features].copy()
                X = pd.concat([X, preprocessing.get_dummy_values(df, estimator['dummies'])], axis=1)
                X.fillna(self.params['nan_flag'], inplace=True)

                X = X.values
                y = df['target'].values

                if self.params['n_targets_stacked'] > 1:
                    p = estimator['pipeline'].predict_proba(X)

                    clf = estimator['pipeline'].steps[-1][1]

                    for i in range(self.params['n_targets']):
                        if (i+1) in clf.classes_:
                            df['pred_stacked_{}_{}'.format(model['name'], i+1)] = p[:, list(clf.classes_).index(i+1)]
                        else:
                            df['pred_stacked_{}_{}'.format(model['name'], i+1)] = np.zeros(len(df))

                else:

                    p = estimator['pipeline'].predict(X)

                    df['pred_stacked_{}_1'.format(model['name'])] += p

            n_estimators = len(model['estimators'])

            if n_estimators:

                if self.params['n_targets'] > 1:
                    for i in range(self.params['n_targets']):
                        df['pred_stacked_{}_{}'.format(model['name'], i+1)] /= n_estimators
                else:
                    df['pred_stacked_{}_1'.format(model['name'])] /= n_estimators

        '''
        for model in self.models:
            for i in range(self.params['n_targets']):
                df['pred_{}_{}'.format(model['name'], i+1)] = -df['pred_{}_{}'.format(model['name'], i+1)]
        '''

        return df



