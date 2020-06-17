

def heuristic_checks(response_dict, min_body_confidence, min_hand_confidence):
    X, Y = 0, 1
    body_ok, hand_ok = True, True

    xmax, ymax = response_dict["image_width"], response_dict["image_height"]
    body_pose_dict = response_dict["body_pose"][0]
    hand_pose_dict = response_dict["handpose"]

    def get_body_keypoint(name):
        for part in body_pose_dict["keypoints"]:
            if part["part"] == name:
                if part["position"]["x"] < xmax and part["position"]["y"] < ymax:
                    point = (part["position"]["x"], part["position"]["y"])
                    return point
                else:
                    return None
        return None

    left_shoulder_point = get_body_keypoint("leftShoulder")
    right_shoulder_point = get_body_keypoint("rightShoulder")
    left_wrist_point = get_body_keypoint("leftWrist")
    right_wrist_point = get_body_keypoint("rightWrist")

    # check if only one hand is over the shoulder (i used wrist point for now)
    left_hand_UP, right_hand_UP = False, False
    if left_shoulder_point != None and left_wrist_point != None:
        left_hand_up_measure = -(left_wrist_point[Y] - left_shoulder_point[Y])  # inverted because is inverted y
        if left_hand_up_measure > 0:
            #print("LEFT UP")
            left_hand_UP = True

    if right_shoulder_point != None and right_wrist_point != None:
        right_hand_up_measure = -(right_wrist_point[Y] - right_shoulder_point[Y])  # inverted because is inverted y
        if right_hand_up_measure > 0:
            #print("RIGHT UP")
            right_hand_UP = True

    hand_ok = (left_hand_UP and not right_hand_UP) or (not left_hand_UP and right_hand_UP)  # XOR

    return body_ok, hand_ok
