
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from debugging_tools import *

def deleteNaN(array):
    """
    Delete NaN values from 3d array in axis=1. The function preserves the
    same size. To keep the shape size same NaN values are removed in place
    and then added to the end. Then All NaNs replaced with "0".
    array: 3d numpy array with NaN values.
    nan_removed: 3d numpy array NaNs removed in axis=1.
    """
    assert (len(array.shape)==3), "The input array must be 3d!"
    shape1 = array.shape[1]
    shape2 = array.shape[2]
    nan_removed = np.empty(shape=(0, shape1, shape2))
    for i in range(array.shape[0]):
        row = array[i]
        # Remove NaNs in place in axis=1:
        row = row[~np.isnan(row).any(axis=1)] # operator ~ means complementary
        # Create NaNs with same size to add to the end of array:
        deleted_row_count = array[i].shape[0] - row.shape[0]
        array_nan = np.empty((deleted_row_count, shape2))
        array_nan[:] = np.nan
        # Add same removed size to the end as NaNs:
        row = np.append(row, array_nan, axis=0)
        row = row.reshape([1, shape1, shape2])
        nan_removed = np.append(nan_removed, row, axis=0)
    # NaN values at the end are replaced with "0" to keep the size same:
    nan_removed = np.nan_to_num(nan_removed, copy=True, nan=0.0)
    return nan_removed


def kerasPadTruncate(array, max_len=120):
    """
    Sequences that are shorter than maxlen are padded with 0 value at the end.
    Sequences longer than maxlen are truncated so that they fit the desired length.
    array: A list of sequence
    """
    array = pad_sequences(array, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    return array

################## Main code #################

def preprocessNumpy(array):
    """
    Main function of preprocessing tools.
    array: Array to process. For deleteNaN() array must be 3d.
    """
    array = deleteNaN(array)

    return array
