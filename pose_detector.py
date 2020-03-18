import numpy as np
import tensorflow as tf
import cv2
import posenet
import time

class PoseNet:
    def __init__(self, model_path, scale_factor=1.0, output_stride=16):
        self.model = tf.lite.Interpreter(model_path)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.scale_factor = scale_factor
        self.output_stride = output_stride

    def detector(self, image):
        dims = image.shape[:-1]
        max_dim = np.max(dims)
        min_dim = np.min(dims)
        pad = (max_dim - min_dim)//2

        padded_img = np.zeros((max_dim, max_dim, 3),dtype=np.uint8)
        padded_img[pad:max_dim-pad, :, :] = image
        image =  cv2.resize(padded_img, (max_dim//2, max_dim//2))

        self.scale_factor = float(self.input_shape[1])/(max_dim//2)
        print(self.scale_factor)

        input_image, display_image, output_scale = posenet.process_input(
                image, scale_factor=self.scale_factor, output_stride=self.output_stride)

        self.model.set_tensor(self.input_details[0]['index'], input_image)

        self.model.invoke()

        heatmaps_result = self.model.get_tensor(self.output_details[0]['index'])
        offsets_result = self.model.get_tensor(self.output_details[1]['index'])
        displacement_fwd_result = self.model.get_tensor(self.output_details[2]['index'])
        displacement_bwd_result = self.model.get_tensor(self.output_details[3]['index'])

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=self.output_stride,
                    max_pose_detections=1,
                    min_pose_score=0.1)

        keypoint_coords *= output_scale
        keypoint_coords = keypoint_coords[:, 5:11, :]
        keypoint_scores = keypoint_scores[:, 5:11]       

        return image, posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.1, min_part_score=0.1)         