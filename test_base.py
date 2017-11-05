import pytest

from keras.models import Sequential
from keras.layers import Dense, Activation

def test_simple():
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])