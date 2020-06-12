ls
# var=$(pwd)
# tzzzz=$("${var// /\\ }/gesture_classification_tools")

tensorflow_model_server --model_base_path= "/gesture_classification_tools" --rest_api_port=9000 --model_name=saved_model &

# function back_ground_process () {
# 	# Random number between 10 and 15
# 	tensorflow_model_server --model_base_path=/home/ob/MaastrichtUniversity/Research\ Project\ DSDM\ 2/gesture_driven_presentation/gesture-driven-presentation/gesture_classification_tools --rest_api_port=9000 --model_name=saved_model &
#
# 	python3 javascript_pose_estimation/detector_api.py
# }
#
# back_ground_process
