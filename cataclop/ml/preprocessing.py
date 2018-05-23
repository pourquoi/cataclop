from collections import defaultdict, Counter

def select_dummy_values(dataset, features, limit=10):
  dummy_values = {}
  for feature in features:
      values = [
          value
          for (value, _) in Counter(dataset[feature]).most_common(limit)
      ]
      dummy_values[feature] = values

  return dummy_values


def dummy_encode_dataframe(dataset, dummies):
  for (feature, dummy_values) in dummies.items():
      for dummy_value in dummy_values:
          dummy_name = u'%s_value_%s' % (feature, unicode(dummy_value))
          dataset[dummy_name] = (dataset[feature] == dummy_value).astype('double')
      del dataset[feature]

  return dataset


def get_dummy_features(DUMMY_VALUES, subset=None):
  features = []
  for (feature, dummy_values) in DUMMY_VALUES.items():
      if subset != None and feature not in subset: continue
      for dummy_value in dummy_values:
          dummy_name = u'%s_value_%s' % (feature, unicode(dummy_value))
          features.append(dummy_name)
  return features
