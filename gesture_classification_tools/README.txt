# Content #

/1
LSTM_model.h5
export_to_protobuf.py
gesture_classification_tools.py
test_client.py

/1: 
Contains the tensorfow model (saved_model.pb) of keras model LSTM_model.h5. The tensorflow model is needed for serving of keras model during production. This folder is created with export_to_protobuf.py script. "1" means the version number, It must be a number to be recognized as a version from tensorflow-model-server. If not tensorflow-model-server outputs an error.

LSTM_model.h5:
The trained keras model

export_to_protobuf.py:
Script to export keras .h5 model to a protobuf .pb model. Creates /1 folder autoatically.

gesture_classification_tools.py:
Tools for the pose classification.

test_client:
A client to test the tensorflow serving model.

---------------------------------------------------------------------------

# How does the tensorflow serving model function? #

Literature:
https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
https://blog.tensorflow.org/2018/08/training-and-serving-ml-models-with-tf-keras.html
https://www.tensorflow.org/tfx/tutorials/serving/rest_simple?hl=uk

To run Tensorflow serving tensorflow-model-server must be installed.

To run the server enter in terminal:
tensorflow_model_server --model_base_path=<absolute_path_to_folder_which_contains_/1 folder> --rest_api_port=9000 --model_name=saved_model

To test the server run the test_client.py.

