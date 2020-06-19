import numpy as np
import requests
import json
from input_processer import processInput, normalizeHandData, frameSampler

BODY_CONFIDENCE_THRESHOLD = 0.9
HAND_CONFIDENCE_THRESHOLD = 0.9

INTEREST_PARTS = {"leftShoulder": 0, "leftElbow": 2, "leftWrist": 4, "rightShoulder": 6, "rightElbow":8, "rightWrist": 10}

class BodyClassificationHandler:
    """
    Class to handle body pose coordinates and output classification
    """
    def __init__(self, frames_per_call=30, minPoseConfidence=0.1, invert=False, flip=True):
        """
        frames_per_call : Set to number of frames to be collected in the array before sending to server
        minPoseConfidence: Minimum confidence of the pose to be considered
        """
        self.frames_per_call = frames_per_call
        self.minPoseConfidence = minPoseConfidence
        self.current_frame = 0
        self.invert = invert
        self.flip = flip
        # Intialize dummy classication input array to be sent to model server
        self.classification_input_array = np.zeros((70, 12))


    def checkforSendFrame(self):
        """
        Checks if the number of frames to be collected has been reached.
        Returns True if reached. If not, then False.
        """
        if self.current_frame == self.frames_per_call - 1:
            self.current_frame = 0
            return True
        else:
            self.current_frame += 1
            return False


    def clearHistory(self):
        """
        Reset classification input array for new input to be sent
        """
        #print("Starting Body Gesture Capture ...")
        self.classification_input_array = np.zeros((70, 12))


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
        predictions = predictions[0]
        max_prediction_value = max(predictions)
        max_prediction_index = predictions.index(max_prediction_value)

        BODY_GESTURES = ["NEXT SLIDE", "PREVIOUS SLIDE", "START/STOP"]
        if max_prediction_value >= BODY_CONFIDENCE_THRESHOLD:
            print("Body Gesture Prediction: " +
                  BODY_GESTURES[max_prediction_index] +
                  " [ " + str(max_prediction_value) + " ]")
        else:
            print(". . .")
        #print("Body Gesture Predictions: ", predictions)


    def update(self, body_pose_dict, xmax=640, ymax=500):
        """
        body_pose_dict: Dictionary of values received from the socket server for body pose.
        xmax: Max width
        ymax: Max height

        This function loads the poses into classification input array.
        This array is sent to tensorflow model server if the frames_per_call frames have been reached.

        P.S. The array is discarded if any body pose keypoint that is mentioned in the INTEREST_POINTS is not detected.
        """
        if body_pose_dict["score"] > self.minPoseConfidence:
            for part in body_pose_dict["keypoints"]:
                if part["part"] in INTEREST_PARTS:
                    if part["position"]["x"] < xmax and part["position"]["y"] < ymax:
                        if self.flip:
                            self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]] = xmax - float(part["position"]["x"])
                        else:
                            self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]] = float(part["position"]["x"])

                        if self.invert:
                            self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]+1] = ymax - float(part["position"]["y"])
                        else:
                            self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]+1] = float(part["position"]["y"])

        # If required number of frames collected
        if self.checkforSendFrame():
            # If any of the required interest points are not detected, then the value will be zero for those in the array.
            # Consider only such arrays where all the required interests points are detected.
            if not np.any(self.classification_input_array[0:self.frames_per_call]==0):
                #print("Sending for server for classification: ")

                normalized_input_array = processInput(self.classification_input_array)
                input_array = np.expand_dims(normalized_input_array, axis=0)
                self.sendFrametoServer(input_array)

            self.clearHistory()

### HAND CLASSIFICATION HANDLER

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


HAND_INTEREST_PARTS = { 'palmBase': [0],
                        'thumb': [1, 2, 3, 4],
                        'indexFinger': [5, 6, 7, 8],
                        'middleFinger': [9, 10, 11, 12],
                        'ringFinger': [13, 14, 15, 16],
                        'pinky': [17, 18, 19, 20], }


class HandClassificationHandler:
    """
    Class to handle body pose coordinates and output classification
    """
    def __init__(self, frames_per_call=10, minPoseConfidence=0.1, invert=False, flip=True):
        """
        frames_per_call : Set to number of frames to be collected in the array before sending to server
        minPoseConfidence: Minimum confidence of the pose to be considered
        """
        self.frames_per_call = frames_per_call
        self.minPoseConfidence = minPoseConfidence
        self.current_frame = 0
        self.invert = invert
        self.flip = flip
        # Intialize dummy classication input array to be sent to model server
        self.classification_input_array = np.zeros((self.frames_per_call, 42))


    def checkforSendFrame(self):
        """
        Checks if the number of frames to be collected has been reached.
        Returns True if reached. If not, then False.
        """
        if self.current_frame == self.frames_per_call - 1:
            self.current_frame = 0
            return True
        else:
            self.current_frame += 1
            return False


    def clearHistory(self):
        """
        Reset classification input array for new input to be sent
        """
        #print("Starting Hand Gesture Capture ...")
        self.classification_input_array = np.zeros((self.frames_per_call, 42))


    def sendFrametoServer(self, array):
        """
        API communication responsible for sending frame and gathering response from tensorflow model server
        """
        data = json.dumps({"signature_name": "serving_default",
            "instances": array.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(
        'http://localhost:9001/v1/models/saved_model/versions/1:predict',
        data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        predictions = predictions[0]

        max_prediction_value = max(predictions)
        max_prediction_index = predictions.index(max_prediction_value)
        HAND_GESTURES = ["ZOOM IN", "ZOOM OUT"]
        if max_prediction_value >= HAND_CONFIDENCE_THRESHOLD:
            print("Hand Gesture Prediction: " +
                  HAND_GESTURES[max_prediction_index] +
                  " [ " + str(max_prediction_value) + " ]")
        else:
            print(". . .")
        #print("Hand Gesture Predictions:", predictions)



    def update(self, body_pose_dict, xmax=640, ymax=500):
        """
        body_pose_dict: Dictionary of values received from the socket server for body pose.
        xmax: Max width
        ymax: Max height

        This function loads the poses into classification input array.
        This array is sent to tensorflow model server if the frames_per_call frames have been reached.

        P.S. The array is discarded if any body pose keypoint that is mentioned in the INTEREST_POINTS is not detected.
        """
        if body_pose_dict:
            if body_pose_dict[0]['handInViewConfidence'] > self.minPoseConfidence:
                hand_keypoints = body_pose_dict[0]['annotations']

                for finger in hand_keypoints:
                    perfinger_keypoints = np.array(hand_keypoints[finger])

                    x_index = [x*2 for x in HAND_INTEREST_PARTS[finger]]
                    y_index = [x+1 for x in x_index]

                    if self.flip:
                        self.classification_input_array[self.current_frame, x_index[:]] = xmax - perfinger_keypoints[:, 0]
                    else:
                        self.classification_input_array[self.current_frame, x_index[:]] = perfinger_keypoints[:, 0]

                    if self.invert:
                        self.classification_input_array[self.current_frame, y_index[:]] = ymax - perfinger_keypoints[:, 1]
                    else:
                        self.classification_input_array[self.current_frame, y_index[:]] = perfinger_keypoints[:, 1]


        # If required number of frames collected
        if self.checkforSendFrame():
            # If any of the required interest points are not detected, then the value will be zero for those in the array.
            # Consider only such arrays where all the required interests points are detected.
            #print("Sending for server for classification: ")
            input_array = np.expand_dims(self.classification_input_array, axis=0)
            #print(input_array)
            input_array = normalizeHandData(input_array)
            input_array = frameSampler(input_array, 40)

            self.sendFrametoServer(input_array)

            self.clearHistory()
