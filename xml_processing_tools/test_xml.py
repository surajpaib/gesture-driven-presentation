import pandas as pd
import xmltodict
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical


"""
XML file 1. layer:
keys: 'data'
XML file 2. layer:
keys: 'Label', 'FPS', 'Frame'
(Label is string, FPS is number, Frame has 3. layer)
XML file 3. layer:
array of lentgh: 36 (Depends on time)
XML file 4. layer:
keys: 'Avg_x', 'Avg_y', 'Avg_dist', 'Keypoint'
(Avg_x, Avg_y, Avg_dist are numbers, Keypoint has 5.layer)
XML file 5. layer:
array of lentgh: 18 (ID, X, Y, Confidence) (18: Count of keypoints
"""


'''
Convert data into TrainX (nr_of_example, nr_timestep, 12). 
'''


def load_data_file(dic_filename):
    '''
    dataX is ndarray (25,12)
    dataY is ndarray (1, 4)
    '''
    with open(dic_filename) as fd:
        doc = xmltodict.parse(fd.read())
    #############DataX##############
    dataX = []
    nr_timestep = 25  # Video is 30fps and at least 1 second. #TODO: A fixed value is not a wise choice.
    for idx in np.linspace(3, len(doc['data']['Frame']) - 3, nr_timestep,
                           dtype=int):  # Remove the noise in the beginning and ending
        data_X = []
        for j in [2, 3, 4, 5, 6,
                  7]:  # The order must be consistent: left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist
            data_X.append(float(doc['data']['Frame'][idx]['Keypoint'][j - 1]['X']))
            data_X.append(float(doc['data']['Frame'][idx]['Keypoint'][j - 1]['Y']))
        dataX.append(data_X)
    dataX = np.vstack(dataX)

    #############DataY##############
    dataY = []
    if 'Iprev' in dic_filename:
        dataY = [1, 0, 0, 0]
    elif 'reset' in dic_filename:
        dataY = [0, 1, 0, 0]
    elif 'rnext' in dic_filename:
        dataY = [0, 0, 1, 0]
    elif 'startstop' in dic_filename:
        dataY = [0, 0, 0, 1]
    else:
        dataY = [0, 0, 0, 0]

    return dataX, dataY


# dic_filename = 'rnext.xml'
# dataX, dataY = read_data(dic_filename)

def load_data_dic(file_dic):
    '''
    X is ndarray (nr_files, 25, 12)
    Y is ndarray (nr_files, 4)
    '''
    trainX = []
    trainY = []
    # testX = []
    for filename in os.listdir(file_dic):
        dataX, dataY = load_data_file(file_dic + filename)
        trainX.append(dataX)
        trainY.append(dataY)
    X = np.stack(trainX)
    Y = np.stack(trainY)
    return X, Y


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]  # 128, 9, 6
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy



################## Main code #################

file_dic = '/Users/lizhaolin/Downloads/preprocessed_video_data/xml_files/test_dic/'
X, Y = load_data_dic(file_dic)
repeats = 10
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
scores = list()
for r in range(repeats):
    score = evaluate_model(trainX, trainy, testX, testy)
    score = score * 100.0
    print('>#%d: %.3f' % (r + 1, score))
    scores.append(score)

summarize_results(scores)
