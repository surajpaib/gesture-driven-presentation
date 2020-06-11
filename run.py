
import os
import platform

"""
Run gesture-driven-presentation-2.0:
"""


BODY_MODEL_PORT=9000
HAND_MODEL_PORT=9001


running_os = platform.system()

if running_os == "Linux":

    WORKING_PATH = os.getcwd()
    BODY_CLASSIFICATION_MODEL=WORKING_PATH + "/gesture_classification_tools"
    HAND_CLASSIFICATION_MODEL=WORKING_PATH + "/hand_gesture_classification_tools"

    os.system("tensorflow_model_server --model_base_path={} --rest_api_port={} --model_name=saved_model &".format(BODY_CLASSIFICATION_MODEL, BODY_MODEL_PORT))
    os.system("tensorflow_model_server --model_base_path={} --rest_api_port={} --model_name=saved_model &".format(HAND_SERVER_PATH, HAND_MODEL_PORT))

    os.system("python3 javascript_pose_estimation/detector_api.py")