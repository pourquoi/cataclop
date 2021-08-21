import os
import shutil
import logging

import pandas as pd
import numpy as np

from dill import dump, load

from tqdm.notebook import tqdm

import keras

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, MinMaxScaler, RobustScaler

from cataclop.ml.exploration import random_race
from cataclop.ml import preprocessing as cataclop_preprocessing
from cataclop.ml.pipeline import factories


class Model(factories.Model):

    def __init__(self, name, params=None, version=None, dataset=None):
        super().__init__(name=name, params=params, version=version)

        if dataset is None:
            raise ValueError('this model requires a dataset to initialise the features')

        self.dataset = dataset
        self.model = None
        self.horse_dummies = None
        self.race_dummies = None
        self.scaler = None
        self.loaded = False
      
        self.logger = logging.getLogger(__name__)

    @property
    def defaults(self):
        return {
            'seed': 1234
        }

    def load(self, force=False):
        if not self.loaded or force:
            self.horse_dummies = load(open(os.path.join(self.data_dir, 'horse_dummies.pkl'), 'rb'))
            self.race_dummies = load(open(os.path.join(self.data_dir, 'race_dummies.pkl'), 'rb'))
            self.model = keras.models.load_model(os.path.join(self.data_dir, 'model.h5'))
            self.scaler = load(open(os.path.join(self.data_dir, 'scaler.pkl'), 'rb'))

        self.loaded = True

    def save(self, clear=False):

        d = self.data_dir

        if not os.path.isdir(d):
            os.makedirs(d)
        elif clear:
            shutil.rmtree(d)
            os.makedirs(d)

        path = os.path.join(d, 'model.h5')

        if os.path.isfile(path):
            os.remove(path)

        self.model.save(path)

        dump(self.horse_dummies, open(os.path.join(self.data_dir, 'horse_dummies.pkl'), 'wb'))
        dump(self.race_dummies, open(os.path.join(self.data_dir, 'race_dummies.pkl'), 'wb'))
        dump(self.scaler, open(os.path.join(self.data_dir, 'scaler.pkl'), 'wb'))

        # @todo save scikit version, numpy version

    def prepare_data(self, dataset, train=True):
        self.logger.debug('preparing model data')

        race_features = ['prize', 'declared_player_count']
        race_features += ['odds_{:d}'.format(i) for i in range(10)]

        horse_features = ['age'] + ['hist_{}_pos'.format(i+1) for i in range(6)]

        for f in dataset.agg_features:
                if f.startswith('final_odds'):
                    continue
                horse_features.append(f)
                horse_features.append('{}_r'.format(f))
                #for s in dataset.agg_features_funcs:
                    #features.append('{}_{}'.format(f, s[0]))

        horse_cat_features = ['horse_sex', 'horse_breed']
        race_cat_features = ['category', 'sub_category']

        race_features = sorted(list(set(race_features)))
        horse_features = sorted(list(set(horse_features)))
        horse_cat_features = sorted(list(set(horse_cat_features)))
        race_cat_features = sorted(list(set(race_cat_features)))

        features = race_features + horse_features
        cat_features = race_cat_features + horse_cat_features

        NAN_FLAG = 0

        df = dataset.players.copy()
        if train:
            df = df.groupby('race_id').filter(lambda r: r['position'].min() == 1 and r['winner_dividend'].max() > 0 and r['odds_0'].min() != dataset.params['nan_flag'] and r['odds_1'].min() != dataset.params['nan_flag'] )
        else:
            df = df.groupby('race_id').filter(lambda r: r['odds_0'].min() != dataset.params['nan_flag'] and r['odds_1'].min() != dataset.params['nan_flag'] )
        df.reset_index(inplace=True)
        df.loc[:, features] = df.loc[:, features].fillna(NAN_FLAG)

        df['position'] = df['position'].fillna(20)

        if train:
            self.scaler = RobustScaler() 
        
        scaled = self.scaler.fit_transform(df.loc[:, features].values)

        df.loc[:, features] = scaled

        if train:
            self.horse_dummies = cataclop_preprocessing.get_dummies(df, horse_cat_features, limit=5)

        df_horse_dummies = cataclop_preprocessing.get_dummy_values(df, self.horse_dummies)

        df = pd.concat([df, df_horse_dummies], axis=1)

        if train:
            self.race_dummies = cataclop_preprocessing.get_dummies(df, race_cat_features, limit=5)

        df_race_dummies = cataclop_preprocessing.get_dummy_values(df, self.race_dummies)

        df = pd.concat([df, df_race_dummies], axis=1)


        self.all_horse_features = list(set(horse_features + list(set(df_horse_dummies.columns))))

        self.all_race_features = list(set(race_features + list(set(df_race_dummies.columns))))

        self.features = features
        self.cat_features = cat_features


        return df


    def make_Xy(self, df, train=True):

        races = df.groupby('race_id')
        n_races = len(races)

        X = []
        y = []

        # keep track of the raw data position in the dataset 
        back_idx = []

        for race_id, race in tqdm(races, total=n_races):
            n_players = len(race)


            for i in range(n_players):
                player1 = race.iloc[i]
                x1 = player1[self.all_horse_features].values

                # only train on 4 first
                if train and (player1['position'] == 0 or player1['final_odds'] >= 40 ):
                    continue

                for j in range(n_players):
                    if j == i:
                        continue
                    player2 = race.iloc[j]

                    # only train on 4 first
                    if train and (player2['position'] == 0 or player2['final_odds'] >= 40):
                        continue

                    x2 = player2[self.all_horse_features].values

                    row_x = np.concatenate((x1, x2, player1[self.all_race_features]))

                    row_y = np.log(1+player2['position']) / np.log(1+player1['position'])

                    X.append(row_x)
                    y.append(row_y)

                    back_idx.append((race.index[i], race.index[j]))

        X = np.array(X)
        X = X.astype(np.float32)

        y = np.array(y)
        y = y.astype(np.float32)
                    
        return (X, y, back_idx)
        

    def train(self, dataset):

        df = self.prepare_data(dataset, train=True)

        race_ids = df['race_id'].unique()
        test_portion = int(len(race_ids) * 0.2)
        test_race_ids = race_ids[0:test_portion]
        train_race_ids = race_ids[test_portion+1:]

        df_train = df[df['race_id'].isin(train_race_ids)].copy()
        df_test = df[df['race_id'].isin(test_race_ids)].copy()

        X, y, _ = self.make_Xy(df_train, train=True)

        model = Sequential()

        model.add(Dense(50, input_dim=X.shape[1]))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('relu'))

        model.compile(loss='mean_squared_error',
            optimizer='RMSprop',
            metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath='/tmp/weights.best.hdf5', 
                        verbose=1, save_best_only=True)

        model.fit(X, y, batch_size=16, epochs=10, validation_split=0.1, callbacks=[checkpointer])

        self.model = model
        self.df = df

        return self._predict(df_test)

    def predict(self, dataset):
        
        df = self.prepare_data(dataset, train=False)

        return self._predict(df)

    def _predict(self, df, train=False):

        X, y, back_idx = self.make_Xy(df, train=False)

        predictions = self.model.predict(X)

        df['score'] = 0

        for i in tqdm(range(len(predictions))):

            df.loc[ back_idx[i][0], 'score' ] = df.loc[ back_idx[i][0], 'score' ] + predictions[i]

        return df



