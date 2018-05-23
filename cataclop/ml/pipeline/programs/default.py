from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier

from cataclop.ml.pipeline import factories

class Program(factories.Program):

    def __init__(self, name, params):
        super().__init__(name, params)

    @property
    def defaults(self):
        return {

        }

    def run(self, **kwarg):

        d = factories.Dataset.factory('default')
        d.load()

        # todo prepare data (remove races etc..)

        groups = d.players['race'].values

        group_kfold = GroupKFold(n_splits=3)

        features = ['race_count', 'victory_count', 'placed_2_count', 'placed_3_count']

        d.players['pred'] = 0.0
            
        for train_index, test_index in group_kfold.split(d.players.values, d.players['pos'].values, groups):
            X_train = d.players[features].iloc[train_index].values
            y_train = (d.players['pos'] == 1).iloc[train_index].astype('int32').values
            
            X_test = d.players[features].iloc[test_index].values
            y_test = (d.players['pos'] == 1).iloc[test_index].astype('int32').values
            
            clf = RandomForestClassifier(n_estimators=10)
            clf = clf.fit(X_train, y_train)
            
            p = clf.predict_proba(X_test)
            
            idx = d.players.iloc[test_index].index
            
            d.players.loc[idx, 'pred'] = p[:, list(clf.classes_).index(1)]
            
            
            