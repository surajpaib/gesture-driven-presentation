
import requests
import json
import numpy as np

"""
Testing of keras serving model:
"""

def testModelServer():
    """
    Function to test the tensorflow serving model.
    """
    # An array to test the serving model:
    test_array = np.arange(1440).reshape([1,120,12])

    data = json.dumps({"signature_name": "serving_default", "instances": test_array.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:9000/v1/models/saved_model/versions/1:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    print(predictions)

    return None


testModelServer()
