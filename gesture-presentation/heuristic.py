import math


X, Y = 0, 1
HAND_ORIENTATION_MARGIN = 40
MIN_CONSTANT_HAND = 1
MAX_CONSTANT_HAND = 2.5
HAND_THRESHOLD = 1.5

ARM_ORIENTATION_MARGIN = 45
FOREARM_ORIENTATION_DOWN_MARGIN = 30

class Heuristic:


    def __init__(self, response_dict, min_body_confidence, min_hand_confidence):
        #print(response_dict)
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

    def get_vector_angle2(self, A, B):
        C = (B[X] - A[X], B[Y] - A[Y])
        return math.degrees(math.atan2(C[Y], C[X]))


    def calculate_distance(self, A, B):
        return math.sqrt((B[X] - A[X]) ** 2 + (B[Y] - A[Y]) ** 2)


    def heuristic_checks(self):

        body_ok, hand_ok = False, False
        left_hand_point, right_hand_point = None, None

        # BODY:leftWrist
        left_shoulder_point = self.get_body_keypoint("leftShoulder")
        right_shoulder_point = self.get_body_keypoint("rightShoulder")
        left_elbow_point = self.get_body_keypoint("leftElbow")
        right_elbow_point = self.get_body_keypoint("rightElbow")
        left_wrist_point = self.get_body_keypoint("leftWrist")
        right_wrist_point = self.get_body_keypoint("rightWrist")

        #HAND:thumb, indexFinger, middleFinger, ringFinger, pinky, palmBase
        palm_base_point = self.get_hand_keypoint('palmBase')
        middle_finger_point = self.get_hand_keypoint('middleFinger')
        index_finger_point = self.get_hand_keypoint('indexFinger')
        pinky_finger_point = self.get_hand_keypoint('pinky')


########################################################################################################################
##########  HAND CHECK  ################################################################################################
########################################################################################################################


        # CHECK IF HAND IS IN CORRECT POSITION
        hand_orientation = self.get_vector_angle2(middle_finger_point, palm_base_point)
        if (90-HAND_ORIENTATION_MARGIN < hand_orientation < 90+HAND_ORIENTATION_MARGIN): #palm_base_point[Y] > middle_finger_point[Y] and
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

            palm_length = self.calculate_distance(index_finger_point, pinky_finger_point)
            hand_recognition_threshold = HAND_THRESHOLD * palm_length

            left_hand_UP, right_hand_UP = False, False
            if left_shoulder_point != None and left_hand_point != None:
                left_hand_up_measure = -(left_hand_point[Y] - left_shoulder_point[Y])  # inverted because is inverted y
                if left_hand_up_measure > hand_recognition_threshold:
                    #print("LEFT UP")
                    left_hand_UP = True

            if right_shoulder_point != None and right_hand_point != None:
                right_hand_up_measure = -(right_hand_point[Y] - right_shoulder_point[Y])  # inverted because is inverted y
                if right_hand_up_measure > hand_recognition_threshold:
                    #print("RIGHT UP")
                    right_hand_UP = True

            hand_ok = (left_hand_UP and not right_hand_UP) or (not left_hand_UP and right_hand_UP)  # XOR

        # CHECK IF THE HAND IS NEAR THE FACE
        ''' AVOIDED FOR NOW
        if hand_ok:
            if right_hand_point:
                hand_point = right_hand_point
                ear_point = self.get_body_keypoint("rightEye")
            else:
                hand_point = left_hand_point
                ear_point = self.get_body_keypoint("leftEye")

            hand_ear_distance = self.calculate_distance(hand_point, ear_point)
            #print("dist: ", str(hand_eye_distance))

            palm_length = self.calculate_distance(index_finger_point, pinky_finger_point)

            if hand_ear_distance < MIN_CONSTANT_HAND * palm_length or hand_ear_distance > MAX_CONSTANT_HAND * palm_length:
                #print("DISTANCE WRONG")
                hand_ok = False
        '''
########################################################################################################################
##########  BODY CHECK  ################################################################################################
########################################################################################################################

        left_arm_ok, right_arm_ok = False, False
        # CHECK IF SHOULDER-ELBOW JOINT IS "VERTICAL"
        if left_elbow_point and left_shoulder_point:
            left_arm_orientation = self.get_vector_angle2(left_elbow_point, left_shoulder_point)
            if (-90-ARM_ORIENTATION_MARGIN < left_arm_orientation < -90+ARM_ORIENTATION_MARGIN): #left_elbow_point[Y] > left_shoulder_point[Y] and
                #print("BODY ORIENTATION ok  ( " + str(left_arm_orientation) + " )")
                left_arm_ok = True
            #else:
            #   print("BODY ORIENTATION bad ( " + str(left_arm_orientation) + " )")

        if right_elbow_point and right_shoulder_point:
            right_arm_orientation = self.get_vector_angle2(right_elbow_point, right_shoulder_point)
            if (-90-ARM_ORIENTATION_MARGIN < right_arm_orientation < -90+ARM_ORIENTATION_MARGIN):
                #print("BODY ORIENTATION ok  ( " + str(left_arm_orientation) + " )")
                right_arm_ok = True
            #else:
            #    print("BODY ORIENTATION bad ( " + str(left_arm_orientation) + " )")

        left_forearm_ok, right_forearm_ok = False, False
        # CHECK IF ELBOW-WRIST JOINT IS "HORIZONTAL"
        if left_wrist_point and left_elbow_point:
            left_forearm_orientation = self.get_vector_angle2(left_wrist_point, left_elbow_point)
            if (-FOREARM_ORIENTATION_DOWN_MARGIN < left_forearm_orientation):
            #    print("BODY ORIENTATION ok  ( " + str(left_forearm_orientation) + " )")
                left_forearm_ok = True
            #else:
            #    print("BODY ORIENTATION bad ( " + str(left_forearm_orientation) + " )")

        if right_wrist_point and right_elbow_point:
            right_forearm_orientation = self.get_vector_angle2(right_wrist_point, right_elbow_point)
            if (right_forearm_orientation > 0 or right_forearm_orientation < -180+FOREARM_ORIENTATION_DOWN_MARGIN):
                #print("BODY ORIENTATION ok  ( " + str(right_forearm_orientation) + " )")
                right_forearm_ok = True
            #else:
            #    print("BODY ORIENTATION bad ( " + str(right_forearm_orientation) + " )")



        body_ok = (left_arm_ok and right_arm_ok) and (left_forearm_ok or right_forearm_ok)
        #return False, False
        return body_ok, hand_ok