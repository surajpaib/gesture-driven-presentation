
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
    array: Numpy array of shape (120,12)
    normalized: Numpy array of shape (120,12)
    """
    normalized = np.zeros([70,12])
    # for i in range(array.shape[0]):
    for i in range(18):

        one_row = normalizeKeypoints(array[i])
        normalized[i] = one_row
    return normalized


def normalizeHandData(array):
    """
    Normalizes the coordinates of the hand dataset. The first two coordinates
    in the hand dataset are the x, y of wrist coordinates. This point is
    subtracted from other points. there are 21 x,y points in the hand pose.
    array: Array of hand pose. shape [1, frames count, 42]
    """
    repeat=21
    wrist_coord = array[:,:,:2]
    wrist_coord = np.tile(wrist_coord,(repeat))
    normalized_arr = array - wrist_coord

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
