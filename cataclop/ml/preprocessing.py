from collections import defaultdict, Counter
import pandas as pd
import numpy as np

def model_to_dict(model):
    d = {}
    for item in model.__dict__.items():
        if not item[0].startswith('_'):
            d.update({item[0]:item[1]})

    return d


def parse_music(music, length):

    positions = np.zeros(length)

    pos = None
    cat = None
    is_year = False
    i = 0
    for c in music:
        if i+1 > length:
            break
            
        if c == '(':
            is_year = True
            continue
            
        if c == ')':
            is_year = False
            continue
            
        if is_year: continue
            
        if pos is None:
            pos = c
            cat = None
            positions[i] = pos if pos.isdigit() else 0
            if positions[i] == 0: positions[i] = 10
            continue
        
        if cat is None:
            cat = c
            pos = None
            i = i+1
            continue
            
    return pd.Series([p for p in positions[:length]], index=['hist_{:d}_pos'.format(i+1) for i in range(length)])


def append_hist(dataset, length=6):
    for i in range(length):
        col = 'hist_{}_pos'.format(i+1)
        if col in dataset.columns:
            dataset.drop(col, axis=1, inplace=True)
    
    df = dataset.apply( lambda p: parse_music(p['music'],length), axis=1 )

    return pd.concat([dataset, df], axis=1)


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
