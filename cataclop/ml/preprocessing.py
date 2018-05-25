from collections import defaultdict, Counter
import pandas as pd

def get_dummies(dataset, features, limit=10):
  dummies = {}
  for feature in features:
      values = [
          value
          for (value, _) in Counter(dataset[feature]).most_common(limit)
      ]
      dummies[feature] = values

  return dummies


def get_dummy_values(dataset, dummies):
  df = {}
  
  for (feature, values) in dummies.items():
      for val in values:
          name = u'%s_value_%s' % (feature, val)
          df[name] = (dataset[feature] == val).astype('double')

  return pd.DataFrame(df)


def get_dummy_features(dummies, subset=None):
  features = []

  for (feature, values) in dummies.items():
      if subset != None and feature not in subset: continue

      for val in values:
          name = u'%s_value_%s' % (feature, val)
          features.append(name)

  return features
