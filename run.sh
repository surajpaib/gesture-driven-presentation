tensorflow_model_server --model_base_path=$PWD/gesture_classification_tools --rest_api_port=9000 --model_name=saved_model &
cd javascript_pose_estimation
python detector_api.py

