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
import matplotlib.pyplot as plt
from debugging_tools import *




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

length=[]
def load_data_file(dic_filename):
    '''
    dataX is ndarray (25,12)
    dataY is ndarray (1, 4)
    '''
    with open(dic_filename) as fd:
        print(dic_filename)
        doc = xmltodict.parse(fd.read())
    #############DataX##############
    dataX = []
    nr_timestep = 25  # Video is 30fps and at least 1 second. #TODO: A fixed value is not a wise choice.
    length.append(len(doc['data']['Frame']))
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
        if (file_dic + filename).endswith(".xml"):
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



def getDataPath():
    """
    Relative path is used to store the data.
    Please store the data in the same folder with the git repo:
    ../gesture-driven-presentation       (git repo)
    ../preprocessed_video_data           (xml data)
    """
    return "../../preprocessed_video_data/xml_files/"


def pickleChecker():
    """
    Checks if X, Y pickles exist. If Yes returns X, y.
    If not returns IOError.
    """
    PICKLE_FOLDER = "../../pickles/"
    file_name_x = 'X.npy'
    file_name_y = 'Y.npy'

    PICKLE_PATH_X = PICKLE_FOLDER + file_name_x
    PICKLE_PATH_Y = PICKLE_FOLDER + file_name_y

    try:
        X = np.load(PICKLE_PATH_X)
        Y = np.load(PICKLE_PATH_Y)
        return X,Y
    except IOError as ioe:
        print(ioe)
        return ioe

def pickleSaver(array, file_name):
    """
    Saves a numpy array as .npy pickle to '../../pickles/file_name'
    array: Numpy array
    file_name: filename as string without extension.
    """
    directory = '../../pickles/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    SAVE_PATH = '../../pickles/' + file_name + '.npy'
    np.save(SAVE_PATH, array)
    return None

################## Main code #################
"""
Important:
The X, Y pickles changes depending on the "folders" variable.
Delete pickles folder if "folders" variable changes.
"""
folders =['LPrev', 'Reset', 'RNext', 'StartStop']

folders = ['StartStop']

loaded_pickle = pickleChecker()

if type(loaded_pickle)==FileNotFoundError:
    X = []
    Y = []
    for folder in folders:
        dic = getDataPath()
        file_dic = dic + folder + '/'
        dataX, dataY = load_data_dic(file_dic)
        X.append(dataX)
        Y.append(dataY)

    X = np.vstack(X)
    Y = np.vstack(Y)
    pickleSaver(X, 'X')
    pickleSaver(Y, 'Y')

else:
    X = loaded_pickle[0]
    Y = loaded_pickle[1]

repeats = 2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scores = list()
for r in range(repeats):
    score = evaluate_model(X_train, y_train, X_test, y_test)

    score = score * 100.0
    print('>#%d: %.3f' % (r + 1, score))
    scores.append(score)

debug(scores)
# summarize_results(scores)
