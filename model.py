import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Flatten, merge, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import matplotlib.pyplot as plt

# Evidently this model breaks Python's default recursion limit
# This is a theano issue
import sys
sys.setrecursionlimit(10000)

def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    model.add(Convolution2D(24, 5, 5, strides=(2,2)))
    model.add(Convolution2D(35, 5, 5, strides=(2,2)))
    model.add(Convolution2D(48, 5, 5, strides=(2,2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# this inspired a lot by neggert/inception_v3.py
# https://gist.github.com/neggert/f8b86d001a367aa7dde1ab6b587246b5

def BNConv(nb_filter, nb_row, nb_col, w_decay, subsample=(1, 1), border_mode="same"):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode, activation="relu",
                      W_regularizer=l2(w_decay) if w_decay else None, init="he_normal")(input)
        return BatchNormalization(mode=0, axis=1)(conv)
    return f

def inception_v3_model():
    pass