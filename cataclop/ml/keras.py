import copy
import io
import h5py
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor as BaseKerasRegressor


def baseline_regressor(input_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class KerasRegressor(BaseKerasRegressor):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = keras.models.load_model(file)
        self.__dict__ = state