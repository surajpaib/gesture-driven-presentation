
import requests
import json
import numpy as np

"""
Testing of keras serving model:
"""

test_array = np.arange(1440).reshape([1,120,12])
payload = {
  "instances": [{'pose': test_array}]
}
r = requests.post('http://localhost:9000/v1/models/LSTM_model:predict', json=payload)
json.loads(r.content)
