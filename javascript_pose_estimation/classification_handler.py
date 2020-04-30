import numpy as np
import requests
import json
from input_processer import processInput


INTEREST_PARTS = {"leftShoulder": 0, "leftElbow": 2, "leftWrist": 4, "rightShoulder": 6, "rightElbow":8, "rightWrist": 10}

class BodyClassificationHandler:
    """
    Class to handle body pose coordinates and output classification
    """
    def __init__(self, frames_per_call=20, minPoseConfidence=0.1):
        """
        frames_per_call : Set to number of frames to be collected in the array before sending to server
        minPoseConfidence: Minimum confidence of the pose to be considered
        """
        self.frames_per_call = frames_per_call
        self.minPoseConfidence = minPoseConfidence
        self.current_frame = 0

        # Intialize dummy classication input array to be sent to model server
        self.classification_input_array = np.zeros((120, 12))


    def checkforSendFrame(self):
        """
        Checks if the number of frames to be collected has been reached.
        Returns True if reached. If not, then False.
        """
        if self.current_frame == self.frames_per_call:
            self.current_frame = 0
            return True
        else:
            self.current_frame += 1
            return False


    def clearHistory(self):
        """
        Reset classification input array for new input to be sent
        """
        self.classification_input_array = np.zeros((120, 12))


    def sendFrametoServer(self, array):
        """
        API communication responsible for sending frame and gathering response from tensorflow model server
        """
        data = json.dumps({"signature_name": "serving_default",
            "instances": array.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(
        'http://localhost:9000/v1/models/saved_model/versions/1:predict',
        data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']

        print(predictions)

    def update(self, body_pose_dict, xmax=640, ymax=500):
        """
        body_pose_dict: Dictionary of values received from the socket server for body pose.
        xmax: Max width
        ymax: Max height

        This function loads the poses into classification input array.
        This array is sent to tensorflow model server if the frames_per_call frames have been reached.

        P.S. The array is discarded if any body pose keypoint that is mentioned in the INTEREST_POINTS is not detected.
        """

        print("-"*100)



        if body_pose_dict["score"] > self.minPoseConfidence:
            for part in body_pose_dict["keypoints"]:
                if part["part"] in INTEREST_PARTS:
                    if part["position"]["x"] < xmax and  part["position"]["y"] < ymax:
                        self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]] = float(part["position"]["x"])
                        self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]+1] = float(part["position"]["y"])

             

        # If required number of frames collected
        if self.checkforSendFrame():
            # If any of the required interest points are not detected, then the value will be zero for those in the array.
            # Consider only such arrays where all the required interests points are detected. 
            if not np.any(self.classification_input_array[0:self.frames_per_call]==0):
                print("Sending for server for classification: ")

                normalized_input_array = processInput(self.classification_input_array)
                input_array = np.expand_dims(normalized_input_array, axis=0)
                self.sendFrametoServer(input_array)

            self.clearHistory()
