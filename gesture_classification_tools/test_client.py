
import requests
import json
import numpy as np


"""
Testing of keras serving model:
"""

def testModelServer(array):
    """
    Function to test the tensorflow serving model.
    array: Numpy array of shape (1,120,12)
    120: frame test_size If frame is size smaller zeros can be added to the
    end. 20 frames should be ok.
    12: x an y points of body pose. The order must be consistent:
    
    [left shoulder x, left shoulder y, left elbow x, left elbow y,
     left wrist x, left wrist y, right shoulder x, right shoulder y,
     right elbow x, right elbow y, right wrist x, right wrist y]

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
