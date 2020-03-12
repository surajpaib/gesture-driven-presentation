import numpy as np
import tensorflow as tf
import cv2
from img_utils import *
import posenet
import time
from hand_tracker import HandTracker

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "./models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./models/hand_landmark.tflite"
ANCHORS_PATH = "./models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2


connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/posenet2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
input_shape = input_details[0]['shape']

for op in output_details:
    print(op)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

output_stride = 32
scale = 0.5354

start = time.time()
frame_count = 0
while cap.isOpened():
    _, img = cap.read()
    img = img[:, 80:560]

    input_image, display_image, output_scale = posenet.process_input(
                img, scale_factor=scale, output_stride=output_stride)

  
    interpreter.set_tensor(input_details[0]['index'], input_image)

    interpreter.invoke()

    heatmaps_result = interpreter.get_tensor(output_details[0]['index'])
    offsets_result = interpreter.get_tensor(output_details[1]['index'])
    displacement_fwd_result = interpreter.get_tensor(output_details[2]['index'])
    displacement_bwd_result = interpreter.get_tensor(output_details[3]['index'])
    
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.1)
    
    keypoint_coords *= output_scale
    
    # Hand Detector
    points, _ = detector(display_image)

    # Draw
    display_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.1, min_part_score=0.1)
    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(display_image, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(display_image, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

    cv2.imshow('out', display_image)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('Average FPS: ', frame_count / (time.time() - start))

