
import requests
import json
import numpy as np


"""
Testing of tensorflow serving model:
"""

def testModelServer(array):
    """
    Client for the tensorflow serving model.

    Input:
    array: Numpy array of shape (1,120,12)
    120: frame size. If frame size of pose is smaller than 120, zeros can be
    added to the end of the array. 20 frames of coordinates and the rest as
    zeros should be ok for the prediction.
    12: x an y points of body pose. Important! The order must be consistent:

    [left shoulder x, left shoulder y, left elbow x, left elbow y,
     left wrist x, left wrist y, right shoulder x, right shoulder y,
     right elbow x, right elbow y, right wrist x, right wrist y]

    Before feeding the input "array" must be processed in the given execution
    order:

    1. Subtract the coordinates of the average point from each keypoint. The
    average point is the middle of the shoulders.

    average point x = (left shoulder x + right shoulder x)/2
    average point y = (left shoulder y + right shoulder y)/2
    subtracted keypoint x = keypoint x - average point x
    subtracted keypoint y = keypoint y - average point y

    2. Normalize by dividing the euclidian shoulder distance to all keypoints.

    shoulder distance = sqrt((right shoulder x - left shoulder x)**2 +
                             (right shoulder y - left shoulder y)**2)
    normalized keypoint x = subtracted keypoint x / shoulder distance
    normalized keypoint y = subtracted keypoint y / shoulder distance

    Output:
    predictions: list in list output. The order is:
    [[lprev, reset, rnext, startstop]]
    """
    data = json.dumps({"signature_name": "serving_default",
                       "instances": array.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
    'http://localhost:9000/v1/models/saved_model/versions/1:predict',
    data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    return predictions


# An array to test the serving model:
test_array = np.arange(1440).reshape([1,120,12])

model_prediction = testModelServer(test_array)
print(model_prediction)
