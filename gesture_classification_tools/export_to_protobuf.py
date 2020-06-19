
import tensorflow as tf
from keras.models import load_model



def exportKerastoProtobuf(path_to_h5='LSTM_truncate70_200units_next_prev_start.h5',
                          export_path='serving_model'):
    """
    Export Keras h5 model to pb model (Protobuf: Protocol Buffer format).
    Protobuf model is needed for serving.
    path_to_h5: Path to Keras h5 model.
    export_path: Path to store Protobuf model.
    """
    model = tf.keras.models.load_model(path_to_h5)
    # Error if folder name is not "1"!
    model.save("1")


exportKerastoProtobuf()
