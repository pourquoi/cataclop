from keras.models import Sequential
from keras.layers import Dense


def baseline_regressor(input_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model