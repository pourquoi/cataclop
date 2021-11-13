import pandas as pd
import numpy as np
import tensorflow
import os
import shutil
import json
import logging
from tqdm import tqdm
from dill import dump, load

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.callbacks import EarlyStopping

from cataclop.settings import BASE_DIR
from cataclop.core import models
from cataclop.ml.keras import KerasRegressor
from cataclop.ml.preprocessing import append_hist, model_to_dict
from cataclop.ml.preprocessing import get_dummies, get_dummy_values

logger = logging.getLogger(__name__)


class Balphagore:

    def __init__(self):
        self.df = None
        self.df_train = None
        self.df_validation = None
        self.models = []
        self.bet_models = []
        self.bets = None
        self.strategies = None

        self.name = 'balphagore'
        self.version = '1.0'
        self.verbose = True

        self.TARGET = 'position_log'

        self.NUM_HIST = 6

        self.AGGREG_FEATURES = ['race_count',
                                'hist_1_pos',
                                'hist_2_pos',
                                'hist_3_pos',
                                'victory_count',
                                'placed_2_count',
                                'placed_3_count',
                                'victory_earnings',
                                'placed_earnings',
                                'prev_year_earnings',
                                'handicap_distance',
                                'handicap_weight'
                                ]

        self.AGGREG_FUNCS = [
            ('mean', np.mean),
            ('std', np.std),
            ('amin', np.min),
            ('amax', np.max)
        ]

        self.NAN_FLAG = 0

        self.FEATURES = [
                            'race_count',
                            'victory_count',
                            'placed_count',
                            'placed_2_count',
                            'placed_3_count',
                            'hist_1_days',
                            'hist_2_days',
                            'prize',
                            'declared_player_count',
                            'age',
                            'earnings',
                            'prev_year_earnings',
                            'victory_earnings',
                            'placed_earnings',
                            'post_position',
                            'handicap_weight',
                            'handicap_distance'
                        ] + \
                        ['hist_{}_pos'.format(h + 1) for h in range(self.NUM_HIST)] + \
                        ['odds_{:d}'.format(i) for i in range(10)]

        for aggreg_feature in self.AGGREG_FEATURES:
            self.FEATURES.append('{}_r'.format(aggreg_feature))
            for func in self.AGGREG_FUNCS:
                self.FEATURES.append('{}_{}'.format(aggreg_feature, func[0]))

        self.CATEGORICAL_FEATURES = [
            'country',
            'category',
            'sub_category',
            'horse_breed',
            'horse_sex'
        ]

        self.BET_FEATURES = [
            'prize',
            'pred'
        ] + ['odds_{:d}'.format(i) for i in range(10)]

        for aggreg_feature in self.AGGREG_FEATURES:
            for func in self.AGGREG_FUNCS:
                self.FEATURES.append('{}_{}'.format(aggreg_feature, func[0]))

        self.BET_CATEGORICAL_FEATURES = [
            'category',
            'sub_category',
            'country',
        ]

        self.BET_TARGET = 'bet_target'

    def load_race(self, id):
        return self.load_dataset(filters={'pk': id})

    def load_dataset(self, filters=None):
        races = models.Race.objects.all().prefetch_related('player_set', 'session', 'session__hippodrome')

        if filters is None:
            races = races.filter(
                start_at__gte='2019-01-01',
                start_at__lt='2019-12-31'
            )
        else:
            races = races.filter(
                **filters
            )

        hippodromes = models.Hippodrome.objects.all()
        hippodromes = [model_to_dict(hippo) for hippo in hippodromes]

        sessions = [race.session for race in races]
        sessions = list(map(lambda s: model_to_dict(s), set(sessions)))

        def player_dict(p):
            d = model_to_dict(p)
            d.update({
                "horse_name": p.horse.name,
                "horse_breed": p.horse.breed,
                "horse_sex": p.horse.sex
            })
            return d

        players = [player_dict(p) for race in races for p in race.player_set.all()]

        races = [model_to_dict(race) for race in races]

        logger.debug('{} races'.format(len(races)))

        races_df = pd.DataFrame.from_records(races, index='id')
        sessions_df = pd.DataFrame.from_records(sessions, index='id')
        hippodromes_df = pd.DataFrame.from_records(hippodromes, index='id')
        players_df = pd.DataFrame.from_records(players, index='id')

        hippodromes_df.index.name = "hippodrome_id"
        sessions_df.index.name = "session_id"
        races_df.index.name = "race_id"

        # optimize a bit the dataframe
        for c in ['horse_breed', 'horse_sex']:
            players_df[c] = players_df[c].astype('category')

        for c in ['category', 'condition_age', 'condition_sex', 'sub_category']:
            races_df[c] = races_df[c].astype('category')

        for c in ['country']:
            hippodromes_df[c] = hippodromes_df[c].astype('category')

        # join all the dataframes into one
        sessions_df = sessions_df.join(hippodromes_df, on="hippodrome_id", lsuffix="_session", rsuffix="_hippo")
        races_df = races_df.join(sessions_df, on="session_id", lsuffix="_race", rsuffix="_session")

        df = players_df.join(races_df, on="race_id", lsuffix="_player", rsuffix="_race")

        df.reset_index(inplace=True)
        df.set_index(['id'], inplace=True)

        df['winner_dividend'].fillna(0., inplace=True)
        df['placed_dividend'].fillna(0., inplace=True)

        # append 6 columns of position history (hist_1_pos, hist_2_pos...)
        # parse the horse 'musique' into those columns
        # eg. '1a4a7d0' : hist_1_pos=1, hist_2_pos=4, hist_3_pos=7, hist_4_pos=10
        df = append_hist(df, self.NUM_HIST)

        df['victory_earnings'] = np.log(1 + df['victory_earnings'].fillna(0))
        df['placed_earnings'] = np.log(1 + df['placed_earnings'].fillna(0))
        df['prev_year_earnings'] = np.log(1 + df['prev_year_earnings'].fillna(0))
        df['year_earnings'] = np.log(1 + df['year_earnings'].fillna(0))
        df['prize'] = np.log(df['prize'].fillna(0) + 1)

        df['handicap_distance'] = df['handicap_distance'].fillna(0.0)
        df['handicap_weight'] = df['handicap_weight'].fillna(0.0)

        df['win'] = (df['position'] == 1).astype(np.float)
        df['placed'] = ((df['position'] >= 1) & (df['position'] <= 3)).astype(np.float)
        df['position_log'] = np.log(1 + df['position'])

        races = df.groupby('race_id')

        stats = races[self.AGGREG_FEATURES].agg([f[1] for f in self.AGGREG_FUNCS])
        # transform the 2 level columns index in 1 (victory_count, (mean, std)) -> (victory_count_mean, victory_count_std)
        stats.columns = ['_'.join(col) for col in stats.columns.values]

        df = df.join(stats, how='left', on='race_id')

        for f in self.AGGREG_FEATURES:
            df['{}_r'.format(f)] = (df[f] - df['{}_mean'.format(f)]) / df['{}_std'.format(f)]

        relative_features = ['{}_r'.format(f) for f in self.AGGREG_FEATURES]

        df[relative_features] = df[relative_features].replace([np.inf, -np.inf], np.nan)
        df[relative_features] = df[relative_features].fillna(self.NAN_FLAG)

        # append sorted odds of the race
        odds = pd.DataFrame(columns=['odds_{:d}'.format(i) for i in range(20)], index=df.index)

        for (id, race) in races:
            df.loc[race.index, 'race_winner_dividend'] = race['winner_dividend'].max()
            odds_sorted = sorted(race['final_odds_ref'].values)[0:20]
            odds.loc[race.index, ['odds_{:d}'.format(i) for i, v in enumerate(odds_sorted)]] = odds_sorted

        df = pd.concat([df, odds], axis=1)

        df[['odds_{:d}'.format(i) for i in range(20)]] = df[['odds_{:d}'.format(i) for i in range(20)]].fillna(self.NAN_FLAG)

        for feature in self.FEATURES:
            df[feature] = df[feature].fillna(self.NAN_FLAG)

        self.df = df

    def train_bet(self):
        df_train = self.bets
        df_train['pred_bet'] = 0

        def build_fn_factory(input_dim):
            def baseline_regressor():
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout

                model = Sequential()
                model.add(Dense(20, input_dim=input_dim, activation='relu'))
                model.add(Dense(1, activation='relu'))
                model.compile(loss='mean_squared_error', optimizer='adam')
                return model

            return baseline_regressor

        K_FOLD = 2

        kfold = KFold(n_splits=K_FOLD)

        splits = list(kfold.split(df_train.values))

        models = []

        for train_index, test_index in splits:
            dummies = get_dummies(df_train.iloc[train_index], self.BET_CATEGORICAL_FEATURES)

            X_train = df_train[self.BET_FEATURES].iloc[train_index].copy()
            y_train = df_train[self.BET_TARGET].iloc[train_index]

            df_dummies = get_dummy_values(df_train.iloc[train_index], dummies)
            X_train = pd.concat([X_train, df_dummies], axis=1)

            X_test = df_train[self.BET_FEATURES].iloc[test_index].copy()
            y_test = df_train[self.BET_TARGET].iloc[test_index]
            df_dummies = get_dummy_values(df_train.iloc[test_index], dummies)
            X_test = pd.concat([X_test, df_dummies], axis=1)

            train_filtered_idx = (df_train[self.TARGET] != 0)

            X_train = X_train.loc[train_filtered_idx]
            y_train = y_train.loc[train_filtered_idx]

            steps = [QuantileTransformer(), KerasRegressor(
                build_fn=build_fn_factory(X_train.shape[1]),
                # 400
                epochs=5,
                batch_size=32,
                # callbacks = callbacks
                verbose=True)]

            X_train = X_train.values
            X_test = X_test.values

            pipeline = make_pipeline(*steps)
            pipeline.fit(X_train, y_train.values)

            p = pipeline.predict(X_test)

            idx = df_train.iloc[test_index].index

            df_train.loc[idx, 'pred_bet'] = p

            models.append({
                'dummies': dummies,
                'pipeline': pipeline
            })

        self.bet_models = models

    def train(self):
        df = self.df
        df['pred'] = 0
        df['profit'] = 0.

        race_ids = df['race_id'].unique()

        # keep 1/5 of the dataset for validation
        VALIDATION_NUM_SAMPLES = int(len(race_ids) / 5)
        race_ids_validation = race_ids[:VALIDATION_NUM_SAMPLES]
        race_ids_train = race_ids[VALIDATION_NUM_SAMPLES:]

        df_validation = df[df['race_id'].isin(race_ids_validation)].copy()

        df_train = df[(df['race_id'].isin(race_ids_train))].copy()
        df_train.reset_index(inplace=True)
        df_validation.reset_index(inplace=True)

        logger.debug('{} train samples, {} validation samples'.format(len(df_train), len(df_validation)))

        def build_fn_factory(input_dim):
            def baseline_regressor():
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout

                model = Sequential()
                model.add(Dense(100, input_dim=input_dim, activation='relu'))
                model.add(Dropout(.2))
                model.add(Dense(5, activation='relu'))
                model.add(Dense(1, activation='relu'))
                model.compile(loss='mean_squared_error', optimizer='adam')
                return model

            return baseline_regressor

        K_FOLD = 2

        groups = df_train['race_id'].values
        group_kfold = GroupKFold(n_splits=K_FOLD)

        splits = list(group_kfold.split(df_train.values, df_train[self.TARGET].values, groups))

        models = []

        for train_index, test_index in splits:
            dummies = get_dummies(df_train.iloc[train_index], self.CATEGORICAL_FEATURES)

            X_train = df_train[self.FEATURES].iloc[train_index].copy()
            y_train = df_train[self.TARGET].iloc[train_index]

            df_dummies = get_dummy_values(df_train.iloc[train_index], dummies)
            X_train = pd.concat([X_train, df_dummies], axis=1)

            X_test = df_train[self.FEATURES].iloc[test_index].copy()
            y_test = df_train[self.TARGET].iloc[test_index]
            df_dummies = get_dummy_values(df_train.iloc[test_index], dummies)
            X_test = pd.concat([X_test, df_dummies], axis=1)

            # filter only the training set
            # train_filtered_idx = ((df_train.iloc[train_index]['position'] == 1) | (df_train.iloc[train_index]['position'] == 2))

            train_filtered_idx = ((~df_train[self.TARGET].isna()) & (~df_train['final_odds_ref'].isna()) & (~df_train['final_odds'].isna()))

            X_train = X_train.loc[train_filtered_idx]
            y_train = y_train.loc[train_filtered_idx]

            callbacks = [
                EarlyStopping(patience=20, monitor='loss')
            ]

            steps = [StandardScaler(), KerasRegressor(
                build_fn=build_fn_factory(X_train.shape[1]),
                # 400
                epochs=5,
                batch_size=32,
                # callbacks = callbacks
                verbose=True)]

            X_train = X_train.values
            X_test = X_test.values

            pipeline = make_pipeline(*steps)
            pipeline.fit(X_train, y_train.values)

            p = pipeline.predict(X_test)

            idx = df_train.iloc[test_index].index

            df_train.loc[idx, 'pred'] = p

            models.append({
                'dummies': dummies,
                'pipeline': pipeline
            })

        self.models = models
        self.df_train = df_train
        self.df_validation = df_validation

    def predict(self, df):
        dummies = self.models[0]['dummies']
        pipeline = self.models[0]['pipeline']

        X_val = df[self.FEATURES]
        y_test = df[self.TARGET]
        df_dummies = get_dummy_values(df, dummies)
        X_val = pd.concat([X_val, df_dummies], axis=1)

        df['pred'] = pipeline.predict(X_val)

    def save_model(self):
        model_dir = os.path.join(BASE_DIR, 'var/models/' + self.name + '-' + self.version)

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        else:
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)

        path = os.path.join(model_dir, 'models.dill')
        dump(self.models, open(path, 'wb+'))

    def load_model(self):
        model_dir = os.path.join(BASE_DIR, 'var/models/' + self.name + '-' + self.version)

        self.models = load(open(os.path.join(model_dir, 'models.dill'), 'rb'))

        if self.verbose:
            print(self.models)

    def bet(self, df, max_races=None):
        bets = []

        df['strategy'] = 0

        df['combo'] = ''
        df['odds'] = ''
        df['top1_correct'] = 0
        df['top1_in_3_correct'] = 0
        df['top2_correct'] = 0
        df['top2_correct_disorder'] = 0
        df['top3_correct'] = 0
        df['top3_correct_disorder'] = 0
        df['top4_correct'] = 0
        df['top4_correct_disorder'] = 0
        df['top2_in_4_correct'] = 0

        races = df.groupby('race_id')

        if max_races is None:
            max_races = len(races)

        def one_fav_two_underdogs(r):
            mid = np.floor(len(r)/2)
            r = r.sort_values('pred', ascending=True)
            if len(r) < 4:
                return None
            return r.iloc[[0, mid, mid-1, mid+1]]

        def three_odds_favs(r):
            r = r.sort_values('final_odds', ascending=True)
            if len(r) < 4:
                return None
            return r.iloc[[0, 1, 2, 3]]

        def one_underdog_two_favs(r):
            mid = np.floor(len(r) / 2)
            r = r.sort_values('pred', ascending=True)
            if len(r) < 4:
                return None
            return r.iloc[[mid, 0, 1, mid + 1]]

        def three_underdogs(r):
            mid = np.floor(len(r) / 2)
            r = r.sort_values('pred', ascending=True)
            if len(r) < 4:
                return None
            return r.iloc[[mid-1, mid, mid+1, 0]]

        def two_odds_fav_one_fav(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 2:
                return None
            favs = r.sort_values('pred', ascending=True)
            favs = favs[(favs['num'] != odds_favs.iloc[0]['num']) & (favs['num'] != odds_favs.iloc[1]['num'])]
            if len(favs) < 2:
                return None

            return pd.DataFrame([odds_favs.iloc[0], odds_favs.iloc[1], favs.iloc[0], favs.iloc[1]])

        def one_odds_fav_two_favs(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 2:
                return None
            favs = r.sort_values('pred', ascending=True)
            favs = favs[(favs['num'] != odds_favs.iloc[0]['num']) & (favs['num'] != odds_favs.iloc[1]['num'])]
            if len(favs) < 2:
                return None

            return pd.DataFrame([odds_favs.iloc[0], favs.iloc[0], favs.iloc[1], odds_favs.iloc[1]])

        def three_favs(r):
            r = r.sort_values('pred', ascending=True)
            if len(r) < 4:
                return None
            return r.iloc[[0, 1, 2, 3]]

        def tocards(r):
            return r.sort_values('pred', ascending=True)

        def tocards1(r):
            return r.sort_values('pred', ascending=True)[1:]

        def tocards2(r):
            return r.sort_values('pred', ascending=True)[2:]

        def tocards3(r):
            return r.sort_values('pred', ascending=True)[3:]

        def tocards4(r):
            return r.sort_values('pred', ascending=True)[4:]

        def two_tocards_one_fav(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 1:
                return None
            tocards = r.sort_values('pred', ascending=True)
            tocards = tocards[tocards['num'] != odds_favs.iloc[0]['num']]
            if len(tocards) < 3:
                return None

            return pd.DataFrame([tocards.iloc[0], tocards.iloc[1], odds_favs.iloc[0], tocards.iloc[2]])

        def two_tocards1_one_fav(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 1:
                return None
            tocards = r.sort_values('pred', ascending=True)[1:]
            tocards = tocards[tocards['num'] != odds_favs.iloc[0]['num']]
            if len(tocards) < 3:
                return None

            return pd.DataFrame([tocards.iloc[0], tocards.iloc[1], odds_favs.iloc[0], tocards.iloc[2]])

        def two_tocards2_one_fav(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 1:
                return None
            tocards = r.sort_values('pred', ascending=True)[2:]
            tocards = tocards[tocards['num'] != odds_favs.iloc[0]['num']]
            if len(tocards) < 3:
                return None

            return pd.DataFrame([tocards.iloc[0], tocards.iloc[1], odds_favs.iloc[0], tocards.iloc[2]])


        def two_tocards3_one_fav(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 1:
                return None
            tocards = r.sort_values('pred', ascending=True)[3:]
            tocards = tocards[tocards['num'] != odds_favs.iloc[0]['num']]
            if len(tocards) < 3:
                return None

            return pd.DataFrame([tocards.iloc[0], tocards.iloc[1], odds_favs.iloc[0], tocards.iloc[2]])

        def two_tocards4_one_fav(r):
            odds_favs = r.sort_values('final_odds', ascending=True)
            if len(odds_favs) < 1:
                return None
            tocards = r.sort_values('pred', ascending=True)[4:]
            tocards = tocards[tocards['num'] != odds_favs.iloc[0]['num']]
            if len(tocards) < 3:
                return None

            return pd.DataFrame([tocards.iloc[0], tocards.iloc[1], odds_favs.iloc[0], tocards.iloc[2]])

        strategies = [
            {
                "name": "three_favs",
                "players": three_favs
            },
            {
                "name": "three_underdogs",
                "players": three_underdogs
            },
            {
                "name": "one_underdog_two_favs",
                "players": one_underdog_two_favs
            },
            {
                "name": "one_fav_two_underdogs",
                "players": one_fav_two_underdogs
            },
            {
                "name": "three_odds_favs",
                "players": three_odds_favs
            },
            {
                "name": "two_odds_fav_one_fav",
                "players": two_odds_fav_one_fav
            },
            {
                "name": "one_odds_fav_two_favs",
                "players": one_odds_fav_two_favs
            },
            {
                "name": "tocards1",
                "players": tocards1
            },
            {
                "name": "tocards1",
                "players": tocards1
            },
            {
                "name": "tocards2",
                "players": tocards2
            },
            {
                "name": "tocards3",
                "players": tocards3
            },
            {
                "name": "tocards4",
                "players": tocards4
            },
            {
                "name": "two_tocards_one_fav",
                "players": two_tocards_one_fav
            },
            {
                "name": "two_tocards1_one_fav",
                "players": two_tocards1_one_fav
            },
            {
                "name": "two_tocards2_one_fav",
                "players": two_tocards2_one_fav
            },
            {
                "name": "two_tocards3_one_fav",
                "players": two_tocards3_one_fav
            },
            {
                "name": "two_tocards4_one_fav",
                "players": two_tocards4_one_fav
            }
        ]

        i_race = 0

        for name, group in tqdm(races, total=max_races, disable=not self.verbose):
            if i_race >= max_races:
                break

            if group['win'].max() != 1:
                continue

            if len(group) <= 6:
                continue

            for strategy in strategies:

                top_pred = strategy["players"](group)
                if top_pred is None:
                    continue
                top_pred.reset_index(inplace=True)

                l = len(top_pred)

                if l == 0:
                    continue

                top1_correct = 0
                top1_in_3_correct = 0
                top2_correct = 0
                top2_correct_disorder = 0
                top3_correct = 0
                top3_correct_disorder = 0
                top4_correct = 0
                top4_correct_disorder = 0
                top2_in_4_correct = 0

                combo = []
                odds = []

                for n, player in top_pred.iterrows():
                    if n >= 5:
                        break
                    if n > l:
                        break
                    pos = 20 if np.isnan(player['position']) else player['position']
                    combo.append(int(player['num']))
                    odds.append(player['final_odds_ref'])
                    if pos == n + 1:
                        if n < 1:
                            top1_correct = 1
                        if n < 2:
                            top2_correct = top2_correct + 1
                        if n < 3:
                            top3_correct = top3_correct + 1
                        if n < 4:
                            top4_correct = top4_correct + 1
                    if pos <= 2 and n < 2:
                        top2_correct_disorder = top2_correct_disorder + 1
                    if pos <= 2 and n < 4:
                        top2_in_4_correct = top2_in_4_correct + 1
                    if pos <= 3 and n < 3:
                        top3_correct_disorder = top3_correct_disorder + 1
                    if pos <= 4 and n < 4:
                        top4_correct_disorder = top4_correct_disorder + 1
                    if pos <= 3 and n == 0:
                        top1_in_3_correct = 1

                row = top_pred.iloc[0].copy()
                row['strategy'] = strategy["name"]

                row['combo'] = json.dumps(combo)
                row['odds'] = json.dumps(odds)
                row['top1_correct'] = top1_correct
                row['top1_in_3_correct'] = top1_in_3_correct
                row['top2_correct'] = int(top2_correct == 2)
                row['top2_correct_disorder'] = int(top2_correct_disorder == 2)
                row['top3_correct'] = int(top3_correct == 3)
                row['top3_correct_disorder'] = int(top3_correct_disorder == 3)
                row['top4_correct'] = int(top4_correct == 4)
                row['top4_correct_disorder'] = int(top4_correct_disorder == 4)
                row['top2_in_4_correct'] = int(top2_in_4_correct >= 2)

                bets.append(row)

            i_race = i_race + 1

        bets = pd.DataFrame(bets)
        bets['strategy'] = bets['strategy'].astype('category')
        bets.reset_index(inplace=True)

        def to_b_odds_n(n):
            def to_b_odds(r):
                odds = json.loads(r)
                return odds[n] if len(odds) > n else np.nan

            return to_b_odds

        bets['b_odds_1'] = bets['odds'].apply(to_b_odds_n(0))
        bets['b_odds_2'] = bets['odds'].apply(to_b_odds_n(1))
        bets['b_odds_3'] = bets['odds'].apply(to_b_odds_n(2))
        bets['b_odds_mean2'] = bets[['b_odds_1', 'b_odds_2']].mean(axis=1)
        bets['b_odds_std2'] = bets[['b_odds_1', 'b_odds_2']].std(axis=1)
        bets['b_odds_mean3'] = bets[['b_odds_1', 'b_odds_2', 'b_odds_3']].mean(axis=1)
        bets['b_odds_std3'] = bets[['b_odds_1', 'b_odds_2', 'b_odds_3']].std(axis=1)

        self.bets = bets
        self.strategies = strategies

    def compute_profit(self):
        bets = self.bets

        bets['combo_real'] = ''
        bets['profit_win'] = 0
        bets['profit_placed'] = 0
        bets['profit_top2_correct'] = 0
        bets['profit_top2_correct_disorder'] = 0
        bets['profit_top3_correct'] = 0
        bets['profit_top3_correct_disorder'] = 0
        bets['profit_top4_correct'] = 0
        bets['profit_top4_correct_disorder'] = 0
        bets['profit_top2_in_4_correct'] = 0

        for index, row in tqdm(bets.iterrows(), total=bets.shape[0], disable=not self.verbose):
            race = models.Race.objects.get(pk=row['race_id'])
            combo = json.loads(row['combo'])

            has_1_in_3 = False
            has_2_sur_4 = False

            for r in race.betresult_set.order_by('type', '-dividend'):

                won = int(r.combo == combo[0:len(r.combo)])

                if r.type == 'E_SIMPLE_GAGNANT':
                    won = row['top1_correct']
                    bets.loc[index, 'profit_win'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_SIMPLE_PLACE' and bets.loc[index, 'profit_placed'] == 0:
                    if not isinstance(r.combo, list):
                        print(r.combo, combo[0:len(r.combo)])
                        continue
                    won = int(np.array_equal(np.sort(r.combo), np.sort(combo[0:len(r.combo)])))
                    has_1_in_3 = True
                    if won:
                        bets.loc[index, 'profit_placed'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_COUPLE_GAGNANT' and bets.loc[index, 'profit_top2_correct_disorder'] == 0:
                    won = row['top2_correct_disorder']
                    bets.loc[index, 'profit_top2_correct_disorder'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_TRIO' and bets.loc[index, 'profit_top3_correct_disorder'] == 0:
                    bets.loc[index, 'combo_real'] = json.dumps(r.combo)
                    won = row['top3_correct_disorder']
                    bets.loc[index, 'profit_top3_correct_disorder'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_TRIO_ORDRE' and bets.loc[index, 'profit_top3_correct'] == 0:
                    won = row['top3_correct']
                    bets.loc[index, 'profit_top3_correct'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_MULTI' and bets.loc[index, 'profit_top4_correct_disorder'] == 0:
                    won = row['top4_correct_disorder']
                    bets.loc[index, 'profit_top4_correct_disorder'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_DEUX_SUR_QUATRE' and bets.loc[index, 'profit_top2_in_4_correct'] == 0:
                    if not isinstance(r.combo, list):
                        print(r.combo, combo[0:len(r.combo)])
                        continue
                    won = int(np.array_equal(np.sort(r.combo), np.sort(combo[0:len(r.combo)])))
                    has_2_sur_4 = True
                    if won:
                        bets.loc[index, 'profit_top2_in_4_correct'] = -1 + won * (r.dividend / 100)

                elif r.type == 'E_QUARTE_PLUS' and bets.loc[index, 'profit_top4_correct'] == 0:
                    bets.loc[index, 'profit_top4_correct'] = -1 + won * (r.dividend / 100)

            if has_2_sur_4 and bets.loc[index, 'profit_top2_in_4_correct'] == 0:
                bets.loc[index, 'profit_top2_in_4_correct'] = -1
            if has_1_in_3 and bets.loc[index, 'profit_placed'] == 0:
                bets.loc[index, 'profit_placed'] = -1

    def debug_race(self, df=None, race_id=None, n=10):
        if df is None:
            df = self.df
        cols = ['position', 'sub_category', 'num', 'music', 'final_odds', 'final_odds_ref', 'pred']

        return df.reset_index(drop=True) \
            .set_index(['race_id', df.index]) \
            .loc[np.random.permutation(df['race_id'].unique())[0:n]][cols] \
            .sort_values(by='race_id')
