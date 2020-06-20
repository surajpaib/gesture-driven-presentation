## Content

/1
LSTM_truncate70_200units_next_prev_start.h5
export_to_protobuf.py
gesture_classification_tools.py
install_tf_serving.sh
preprocessing_tools.py
README.md
run_model_server.sh
test_client.py
xml_processing_tools.py

##### /1: 
Contains the tensorfow model (saved_model.pb) of keras model LSTM_truncate70_200units_next_prev_start.h5. The tensorflow model is needed for serving of keras model during production. This folder is created with export_to_protobuf.py script. "1" means the version number, It must be a number to be recognized as a version from tensorflow-model-server. If not tensorflow-model-server outputs an error.

##### LSTM_truncate70_200units_next_prev_start.h5:
The trained keras model

##### export_to_protobuf.py:
Script to export keras .h5 model to a protobuf .pb model. Creates /1 folder autoatically.

##### gesture_classification_tools.py:
Tools for the pose classification.

##### install_tf_serving.sh:
Bash script to install tensorflow model server.

##### preprocessing_tools.py:
Tools for the preprocessing of the dataset.

##### README.md:
Readme file for the gesture_classification_tools folder.

##### run_model_server.sh:
Bash script to run body classification model server.

##### test_client.py: 
A client to test the tensorflow serving model.

##### xml_processing_tools.py:
Tools for reading of the dataset.

<br><br>
### Folder structure:
For reading the datasets and creating the models the folder structure must be as following:

```
<folder_name>
└─── gesture-driven-presentation/
│    └─── dataset_manipulation/
│    └─── gesture_classification_tools/
│    └─── hand_gesture_classification_tools/
│    └─── ...
│
└─── pickles/
│    └─── X.npy
│    └─── x_hand.npy
│    └─── Y.npy
│    └─── y_hand.npy
│
└─── preprocessed_video_data/
     └─── pkl_files/
     |    └─── closed_palm/
     |    └─── open_palm/
     └─── xml_files/
          └─── LPrev/
          └─── ...
```

<br><br>
### How does the tensorflow serving model work?

#####Literature:
https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
https://blog.tensorflow.org/2018/08/training-and-serving-ml-models-with-tf-keras.html
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple?hl=uk

To run Tensorflow serving tensorflow-model-server must be installed.

To run the server enter in terminal:

```
tensorflow_model_server --model_base_path=<absolute_path_to_folder_which_contains_/1 folder> --rest_api_port=9000 --model_name=saved_model
```

To test the server run the test_client.py.

