import math


X, Y = 0, 1


class Heuristic:

    def __init__(self, response_dict, min_body_confidence, min_hand_confidence):
        self.body_pose_dict = response_dict["body_pose"][0]
        self.hand_pose_dict = response_dict["handpose"]
        self.xmax = response_dict["image_width"]
        self.ymax = response_dict["image_height"]

        self.min_body_confidence = min_body_confidence
        self.min_hand_confidence = min_hand_confidence



    # for getting the (X, Y) of the desired body keypoint
    def get_body_keypoint(self, name):
        for part in self.body_pose_dict["keypoints"]:
            if part["part"] == name:
                if part["position"]["x"] < self.xmax and part["position"]["y"] < self.ymax:
                    point = (part["position"]["x"], part["position"]["y"])
                    return point
                else:
                    return None
        return None

    # for getting the (X, Y) of the desired hand keypoint
    # 0 is the base of the finger, 3 is its tip
    def get_hand_keypoint(self, name, pos=0):
        if self.hand_pose_dict[0]['handInViewConfidence'] > self.min_hand_confidence:
            hand_keypoints = self.hand_pose_dict[0]['annotations']
            point = hand_keypoints[name][pos]
            return (point[X], point[Y])
        else:
            return None


    def get_vector_angle(self, A, B):
        C = (B[X] - A[X], B[Y] - A[Y])
        return math.degrees(math.atan(C[X] / C[Y]))

    def calculate_distance(self, A, B):
        return math.sqrt((B[X] - A[X]) ** 2 + (B[Y] - A[Y]) ** 2)


    def heuristic_checks(self):

        body_ok, hand_ok = True, False
        left_hand_point, right_hand_point = None, None

        left_shoulder_point = self.get_body_keypoint("leftShoulder")
        right_shoulder_point = self.get_body_keypoint("rightShoulder")
        #left_wrist_point = self.get_body_keypoint("leftWrist")
        #right_wrist_point = self.get_body_keypoint("rightWrist")
        #    get_hand_keypoint('thumb')
        #    get_hand_keypoint('indexFinger')
        #    get_hand_keypoint('middleFinger')
        #    get_hand_keypoint('ringFinger')
        #    get_hand_keypoint('pinky')
        #    get_hand_keypoint('palmBase')

        palm_base_point = self.get_hand_keypoint('palmBase')
        middle_finger_point = self.get_hand_keypoint('middleFinger')
        index_finger_point = self.get_hand_keypoint('indexFinger')
        pinky_finger_point = self.get_hand_keypoint('pinky')

        # CHECK IF HAND IS IN CORRECT POSITION
        hand_orientation = self.get_vector_angle(palm_base_point, middle_finger_point)
        if palm_base_point[Y] > middle_finger_point[Y] and hand_orientation < 45 and hand_orientation > -45:
            #print("HAND ORIENTATION ok")
            hand_ok = True
        #else:
            #print("HAND ORIENTATION bad")
            #hand_ok = False

        # CHECK IF ONLY ONE HAND IS OVER THE SHOULDER
        # CHECK IF HAND IS RIGHT OR LEFT
        if hand_ok:


            index_pinky_Xdistance = index_finger_point[X] - pinky_finger_point[X]
            if index_pinky_Xdistance > 0:
                #print("DX UP")
                right_hand_point = palm_base_point
            else:
                #print("SN UP")
                left_hand_point = palm_base_point


            left_hand_UP, right_hand_UP = False, False
            if left_shoulder_point != None and left_hand_point != None:
                left_hand_up_measure = -(left_hand_point[Y] - left_shoulder_point[Y])  # inverted because is inverted y
                if left_hand_up_measure > 0:
                    #print("LEFT UP")
                    left_hand_UP = True

            if right_shoulder_point != None and right_hand_point != None:
                right_hand_up_measure = -(right_hand_point[Y] - right_shoulder_point[Y])  # inverted because is inverted y
                if right_hand_up_measure > 0:
                    #print("RIGHT UP")
                    right_hand_UP = True

            hand_ok = (left_hand_UP and not right_hand_UP) or (not left_hand_UP and right_hand_UP)  # XOR

        # CHECK IF THE HAND IS NEAR THE FACE
        if hand_ok:
            if right_hand_point:
                hand_point = right_hand_point
                eye_point = self.get_body_keypoint("rightEye")
            else:
                hand_point = left_hand_point
                eye_point = self.get_body_keypoint("leftEye")

            hand_eye_distance = self.calculate_distance(hand_point, eye_point)
            #print("dist: ", str(hand_eye_distance))

            palm_length = self.calculate_distance(index_finger_point, pinky_finger_point)

            if hand_eye_distance < 1.5 * palm_length or hand_eye_distance > 4 * palm_length:
                print("DISTANCE WRONG")
                hand_ok = False


        return body_ok, hand_ok