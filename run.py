
import os
import platform

"""
Run gesture-driven-presentation-2.0:
"""

running_os = platform.system()

if running_os == "Linux":

    WORKING_PATH = os.getcwd()
    WORKING_PATH = WORKING_PATH.replace(" ","\ ")
    POSE_SERVER_PATH=WORKING_PATH + "/gesture_classification_tools"
    # HAND_POSE_SERVER_PATH=WORKING_PATH + "/hand_gesture_classification_tools"

    os.system("tensorflow_model_server --model_base_path={} --rest_api_port=9000 --model_name=saved_model &".format(POSE_SERVER_PATH))
    # os.system("tensorflow_model_server --model_base_path={} --rest_api_port=9000 --model_name=saved_model &".format(HAND_POSE_SERVER_PATH))
    os.system("python3 javascript_pose_estimation/detector_api.py")
