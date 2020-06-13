
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, BatchNormalization
from keras.layers import LSTM
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from debugging_tools import *
from pkl_processing_tools import pklToNumpy
from keras.callbacks import LambdaCallback
from keras.models import load_model

"""
This file creates a keras classification model from hand dataset:
"""


def loadKerasModel(filename='LSTM_hand_model.h5'):
    """
    Load a saved keras model from file.
    Path of file is same with this function.
    filename: filename to load.
    """
    model = load_model(filename)
    print("Keras model from file {} is loaded:".format(filename))
    model.summary()
    return model


def createKerasModel(timestep, feature, output):
    """
    Create a keras model.
    timestep: trainX.shape[1]
    feature: trainX.shape[2]
    output: trainy.shape[1]
    """
    model = Sequential()
    model.add(LSTM(200, input_shape=(timestep, feature)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output, activation='softmax'))
    print("Keras model is created:")
    model.summary()
    return model



def evaluate_model(trainX, trainy, testX, testy, load_model=False,
                   filename='LSTM_model_hand_model.h5', save_model=False):
    """
    Evaluate a keras model for given train and test parameters.
    The model is either loaded from file with loadKerasModel or created with
    createKerasModel().
    trainX, trainy, testX, testy: Model inputs in Numpy
    load_model: If True load the model from file name.
    filename: File of the model.
    """

    verbose, epochs, batch_size = 1, 50, 42
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]  # 128, 9, 6
    if load_model==True:
        model = loadKerasModel(filename)
    else:
        model = createKerasModel(n_timesteps, n_features, n_outputs)

    # print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[0].get_weights()))
    print_epoch_nr = LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch))

    opt = keras.optimizers.Adam(learning_rate=0.0001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-07,
                                 amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size,
                                        verbose=verbose,
                                        validation_data=(testX, testy))#, callbacks = [print_weights])
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks = epoch_ending(trainX, trainy))
    print("loss: ", history.history['loss'])
    print("accuracy: ",history.history['accuracy'])
    # evaluate model
    _, train_accuracy = model.evaluate(trainX, trainy, batch_size=batch_size,
                                                       verbose=0)
    _, test_accuracy = model.evaluate(testX, testy, batch_size=batch_size,
                                                    verbose=0)
    if save_model==True:
        model.save("{}".format(filename))
    return train_accuracy, test_accuracy


def main():

    x,y = pklToNumpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.20,
                                                        random_state=42)

    repeats = 1
    scores = list()
    for r in range(repeats):
        train_test_scores = evaluate_model(x_train, y_train, x_test, y_test,
                                           load_model=False,
                                           filename='LSTM_hand_model.h5',
                                           save_model=True)
        train_score = train_test_scores[0]
        test_score = train_test_scores[1]
        train_score = train_score * 100.0
        test_score = test_score * 100.0

        print('train score: >#%d: %.3f' % (r + 1, train_score))
        print('test score: >#%d: %.3f' % (r + 1, test_score))
        scores.append(train_score)
        scores.append(test_score)


main()
