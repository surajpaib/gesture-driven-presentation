
import os
import numpy as np
import pandas as pd
from pathlib import Path


"""
This file reads pickles of extracted coordinates from each .pkl test file,
converts and saves as numpy file.
The mapping of the hand coordinates:


           8   12  16  20
           |   |   |   |
           7   11  15  19
       4   |   |   |   |
       |   6   10  14  18
       3   |   |   |   |
       |   5---9---13--17
       2    \         /
        \    \       /
         1    \     /
          \    \   /
           ------0-

"""

def normalizeHandData(array):
    """
    Normalizes the coordinates of the hand dataset. The first two coordinates
    in the hand dataset are the x, y of wrist coordinates. This point is
    subtracted from other points. there are 21 x,y points in the hand pose.
    Each point is divided to x-distance between points 5 and 17.
    array: Array of hand pose. shape [1, frames count, 42]
    """
    repeat=21
    wrist_coord = array[:,:,:2]
    wrist_coord = np.tile(wrist_coord,(repeat))
    normalized_arr = array - wrist_coord
    dist = normalized_arr[:,:,10]-normalized_arr[:,:,34]
    normalized_arr = normalized_arr / dist.T

    return normalized_arr


def frameSampler(array, target_frame):
    """
    Random sample frames from the array to given target frame number:
    array: frame array with shape[1, frame_size, coordinates]
    coordinates are 12 for body pose and 42 for hand pose.
    target_frame: target frame number for the array.
    sampled_array: Shape: [1, target_frame, coordinates]
    """
    frame_size = array.shape[1]
    samples = np.linspace(0,frame_size, num=target_frame, endpoint=False)
    samples = samples//1
    samples = samples.astype(int)
    # Get the frames, which are in samples array:
    sampled_array = array[:,samples,:]

    return sampled_array


def handPickleReader(path):
    """
    Reads the pickle files in the folder and pad with zeros:
    path: folder of the pickle files.
    sampled_data_all: numpy array of the hand pose coordinates.
    Shape of output: [data count, frame_size, 42]
    The maximal count of frames is 142 in open hand dataset.
    The maximal count of frames is 112 in closed hand dataset.
    The hand dataset has 21 points each with an x and y coordinate.
    Therefore 21 x 2 = 42.
    """
    frame_size = 40
    sampled_data_all = np.empty([0,frame_size,42])
    file_list = sorted(os.listdir(path))
    for i in file_list:

        palm_data = pd.read_pickle(path + "/{}".format(i))
        palm_data = [data for data in palm_data if data is not None]
        palm_data = np.array(palm_data)
        palm_data = palm_data.reshape((1, palm_data.shape[0],-1))
        palm_data = normalizeHandData(palm_data)
        palm_data = frameSampler(palm_data, frame_size)
        # pad the data with zeros:
        # pad_zero = np.zeros([1,142,42])
        # pad_zero[0,:palm_data.shape[1], :palm_data.shape[2]] = palm_data
        sampled_data_all = np.append(sampled_data_all, palm_data, axis=0)

    return sampled_data_all


def pickleChecker():
    """
    Checks if x, y pickles exist. If Yes returns x, y.
    If not returns IOError.
    """
    PICKLE_FOLDER = "../../pickles/"
    PICKLE_PATH_X = Path(PICKLE_FOLDER + 'x_hand.npy')
    PICKLE_PATH_Y = Path(PICKLE_FOLDER + 'y_hand.npy')
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


def getAppendPickles(closed_palm_path, open_palm_path, balanced=True):
    """
    Get all pickles form hand dataset, comcate and save these as numpy arrays:
    closed_palm_path: folder of closed palm pickles.
    open_palm_path: folder of open palm pickles.
    balanced: bool. If true balances the dataset. closed and open dataset will
    have same array size.
    """
    closed_y = [1, 0]
    open_y = [0, 1]

    closed_x = handPickleReader(closed_palm_path)
    closed_y = np.tile(closed_y, (closed_x.shape[0],1))
    open_x = handPickleReader(open_palm_path)
    open_y = np.tile(open_y, (open_x.shape[0],1))

    if balanced:
        min_shape = min(closed_x.shape[0], open_x.shape[0])
        closed_x = closed_x[:min_shape]
        closed_y = closed_y[:min_shape]
        open_x = open_x[:min_shape]
        open_y = open_y[:min_shape]

    x = np.vstack((closed_x, open_x))
    y = np.vstack((closed_y, open_y))

    pickleSaver(x, 'x_hand')
    pickleSaver(y, 'y_hand')
    print("Hand dataset is saved to pickles folder...")
    return x,y


def pklToNumpy():
    """
    Get x,y values from pkl files. If x_hand and y_hand exist first tries to
    read from these. If not read from .pkl files.
    """
    CLOSED_PALM_PATH = "../../preprocessed_video_data/pkl_files/closed_palm"
    OPEN_PALM_PATH = "../../preprocessed_video_data/pkl_files/open_palm"

    loaded_pickle = pickleChecker()
    if type(loaded_pickle) != FileNotFoundError:
        x = loaded_pickle[0]
        y = loaded_pickle[1]
    else:
        x,y = getAppendPickles(CLOSED_PALM_PATH, OPEN_PALM_PATH, balanced=True)

    return x,y


























#
