import numpy as np
import tensorflow as tf
import cv2
import posenet
import time

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
scale = 0.4015625

start = time.time()
frame_count = 0
while cap.isOpened():
    _, img = cap.read()
    padded_img = np.zeros((640, 640, 3),dtype=np.uint8)

    padded_img[80:560, :, :] = img
    img = padded_img
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
                min_pose_score=0.)
    
    keypoint_coords *= output_scale
    keypoint_coords = keypoint_coords[:, 5:11, :]
    keypoint_scores = keypoint_scores[:, 5:11]
    print(pose_scores.shape, keypoint_scores.shape)

    display_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0., min_part_score=0.)

    cv2.imshow('out', display_image)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('Average FPS: ', frame_count / (time.time() - start))

