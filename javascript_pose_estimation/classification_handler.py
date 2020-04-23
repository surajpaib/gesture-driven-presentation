import numpy as np
import requests
import json

INTEREST_PARTS = {"leftShoulder": 0, "leftElbow": 2, "leftWrist": 4, "rightShoulder": 6, "rightElbow":8, "rightWrist": 10}

class BodyClassificationHandler:
    def __init__(self, frames_per_call=20, minPoseConfidence=0.1):
        self.frames_per_call = frames_per_call
        self.minPoseConfidence = minPoseConfidence
        self.current_frame = 0
        self.classification_input_array = np.zeros((120, 12))


    def checkforSendFrame(self):
        if self.current_frame == self.frames_per_call:
            self.current_frame = 0
            return True
        else:
            self.current_frame += 1
            return False


    def clearHistory(self):
        self.classification_input_array = np.zeros((120, 12))


    def sendFrametoServer(self, array):
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

        print("-"*100)



        if body_pose_dict["score"] > self.minPoseConfidence:
            for part in body_pose_dict["keypoints"]:
                if part["part"] in INTEREST_PARTS:
                    if part["position"]["x"] < xmax and  part["position"]["y"] < ymax:
                        self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]] = float(part["position"]["x"])
                        self.classification_input_array[self.current_frame, INTEREST_PARTS[part["part"]]+1] = float(part["position"]["y"])


        if self.checkforSendFrame():

            if not np.any(self.classification_input_array[0:self.frames_per_call]==0):
                print("Sending for server forr classification: ")

                input_array = np.expand_dims(self.classification_input_array, axis=0)
                self.sendFrametoServer(input_array)

            self.clearHistory()
