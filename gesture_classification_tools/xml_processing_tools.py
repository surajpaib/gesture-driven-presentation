import xmltodict
import numpy as np
import os
from preprocessing_tools import *
from pathlib import Path


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
        print(dic_filename)
        doc = xmltodict.parse(fd.read())

    #############DataX##############
    dataX = []
    FPS= doc['data']['FPS']
    file_length = int(len(doc['data']['Frame']))
    ####
    # Note that the file 'rnext 239' and 'rnext 223' have only 4 data in total. Delete it from dataset.
    # The rest file has at least 16 data.
    ####
    down_sample_ratio = 1 / 3
    nr_timestep = int(len(doc['data']['Frame'])*down_sample_ratio)  # Downsample from 30fps to 5fps
    # print('nr_timestep is:', nr_timestep)
    # print('fps is:', doc['data']['FPS'])
    # print('file_length is:', file_length)
    # for idx in range(len(doc['data']['Frame'])): #Load all frames
    for idx in np.linspace(1, len(doc['data']['Frame'])-1, num=nr_timestep,
                           dtype=int):  # Remove the noise in the beginning and ending
        data_X = []
        #  The order must be consistent:
        #  left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist
        for j in [2, 3, 4, 5, 6, 7]:
            data_X.append(float(doc['data']['Frame'][idx]['Keypoint'][j]['X']))
            data_X.append(float(doc['data']['Frame'][idx]['Keypoint'][j]['Y']))
        dataX.append(data_X)
    dataX = np.vstack(dataX)

    #############DataY##############
    dataY = []
    if 'lprev' in dic_filename:
        dataY = [1, 0, 0]

    elif 'rnext' in dic_filename:
        dataY = [0, 1, 0]

    elif 'startstop' in dic_filename:
        dataY = [0, 0, 1]

    return dataX, dataY, FPS, file_length


def load_data_dic(file_dic, preprocessing, process_type):
    '''
    X is ndarray (nr_files, 25, 12)
    Y is ndarray (nr_files, 4)
    '''
    trainX = []
    trainY = []
    FPSs = []
    file_lengths = []
    # testX = []
    asd = os.listdir(file_dic)
    for filename in os.listdir(file_dic):
        file = Path(str(file_dic) + "/" + filename)
        if file.exists() and file.suffix == ".xml":
            dataX, dataY, FPS, file_length = load_data_file(str(file))
            trainX.append(dataX)
            trainY.append(dataY)
            FPSs.append(FPS)
            file_lengths.append(file_length)
    if preprocessing == True:
        trainX = preprocessNumpy(trainX, process_type=process_type)

    X = np.stack(trainX)
    Y = np.stack(trainY)
    return X, Y


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

    PICKLE_PATH_X = Path(PICKLE_FOLDER + 'X.npy')
    PICKLE_PATH_Y = Path(PICKLE_FOLDER + 'Y.npy')

    try:
        X = np.load(PICKLE_PATH_X)
        Y = np.load(PICKLE_PATH_Y)
        return X, Y
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

def xmlToNumpy(preprocessing = True, process_type = 'resample'):
    """
    Get X,Y values from xml files. If pickles exist first tries to read from
    pickles. If not read from .xml files.
    Important:
    The X, Y pickles changes depending on the "folders" variable.
    Delete pickles folder in (../../pickles) if "folders" variable changes.
    preprocessing (default: True): If preprocessing=True preprocess X with
                                   preprocessing_tools package before saving.
    process_type = 'resample' or 'truncate'
    """

    folders = ['LPrev', 'RNext', 'StartStop']

    loaded_pickle = pickleChecker()

    if type(loaded_pickle) != FileNotFoundError:
        X = loaded_pickle[0]
        Y = loaded_pickle[1]
    else:
        X, Y = [], []

        for folder in folders:
            dic = getDataPath()
            file_dic = Path(dic + folder + '/')

            if not file_dic.exists():
                print(file_dic, " does not exist")
            else:
                dataX, dataY = load_data_dic(file_dic, preprocessing=preprocessing, process_type=process_type)
                X.append(dataX)
                Y.append(dataY)

        if len(X) != 0 and len(Y) != 0:
            X = np.vstack(X)
            Y = np.vstack(Y)

            pickleSaver(X, 'X')
            pickleSaver(Y, 'Y')

    return X, Y
