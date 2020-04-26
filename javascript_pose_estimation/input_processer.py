
import numpy as np

def normalizeKeypoints(one_frame_array):
    """
    1. Subtract the coordinates of the average point from each keypoint. The
    average point is the middle of the shoulders.
    2. Normalize by dividing the euclidian shoulder distance to all keypoints.

    one_frame_array: Numpy array of interest points with shape (12,).
    one_frame_array must be one frame of the body pose.
    normalized_keypoints: Numpy array of normalized interest points with
    shape (12,).
    """
    # 1.
    left_shoulder_x = one_frame_array[0]
    left_shoulder_y = one_frame_array[1]
    right_shoulder_x = one_frame_array[6]
    right_shoulder_y = one_frame_array[7]

    av_point_x = (left_shoulder_x + right_shoulder_x)/2
    av_point_y = (left_shoulder_y + right_shoulder_y)/2

    repeat = int(len(one_frame_array)/2)
    to_subtract = np.tile(np.array([av_point_x, av_point_y]), repeat)
    subtracted_keypoints = one_frame_array - to_subtract
    # 2.
    shoulder_dist = np.sqrt((right_shoulder_x - left_shoulder_x)**2 +
                            (right_shoulder_y - left_shoulder_y)**2)
    normalized_keypoints = subtracted_keypoints / shoulder_dist

    return normalized_keypoints


def processInput(array):
    """
    The function is sending the array row by row to normalizeKeypoints().
    array: Numpy array of shape (1,120,12)
    normalized: Numpy array of shape (1,120,12)
    """
    array=array.squeeze()
    normalized = np.empty([120,12])
    for i in range(array.shape[0]):
        one_row = normalizeKeypoints(array[i])
        normalized[i] = one_row
    normalized = normalized.reshape([1,120,12])
    return normalized
