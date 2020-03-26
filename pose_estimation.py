import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import numpy as np
import json
from person import Person
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

PARTS = {
   
    0: 'LEFT_SHOULDER',
    1: 'RIGHT_SHOULDER',
    2: 'LEFT_ELBOW',
    3: 'RIGHT_ELBOW',
    4: 'LEFT_WRIST',
    5: 'RIGHT_WRIST'
  
}



def save_model_details(m):
    with open('model_details.json', 'w') as outfile:
        info = dict(list(enumerate(m.get_tensor_details())))
        s = json.dumps(str(info))
        outfile.write(s)


def draw(person, img, label_joints):
    radius = 2
    color = (0, 255, 0)  # BGR
    thickness = 1
    print(len(person.get_coords()))
    for index, p in enumerate(person.get_coords()):
        cv.circle(img, p, radius, color, thickness)
        if label_joints:
            cv.putText(img, PARTS[index], p, cv.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255))
    for p1, p2 in person.get_limbs():
        cv.line(img, p1, p2, color, thickness)
    return img



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_pose", type=bool, default=False, help="Set to True to enable hand pose estimation")
    parser.add_argument("--label_joints", type=bool, default=True,  help="Set to False to disable joint labels")

    args = parser.parse_args()

    if args.hand_pose:
        detector = HandTracker(
            PALM_MODEL_PATH,
            LANDMARK_MODEL_PATH,
            ANCHORS_PATH,
            box_shift=0.2,
            box_enlarge=1.3
        )

    model = tf.lite.Interpreter('models/posenet2.tflite')
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32

    cap = cv.VideoCapture(0)
    frame_count = 0
    start = time.time()

    while cap.isOpened():
        _, frame = cap.read()
        hand_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = np.copy(frame)
        print("original shape: ", img.shape)
        img2 = cv.resize(img, (257, 257), interpolation=cv.INTER_LINEAR)
        img = tf.reshape(tf.image.resize(img, [257, 257]), [1, 257, 257, 3])
        if floating_model:
            img = (np.float32(img) - 127.5) / 127.5
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()

        output_data = np.squeeze(
            model.get_tensor(
                output_details[0]['index']))  # o()
        offset_data = np.squeeze(model.get_tensor(output_details[1]['index']))
        p = Person(output_data, offset_data)

        if args.hand_pose:
            points, _ = detector(hand_image)



        pimg = draw(p, img2, args.label_joints)
    
        if args.hand_pose:
            if points is not None:
                for point in points:
                    x, y = point
                    cv.circle(pimg, (int(x * 257/hand_image.shape[1]), int(y*  257/hand_image.shape[0])), THICKNESS * 2, POINT_COLOR, THICKNESS)
                for connection in connections:
                    x0, y0 = points[connection[0]]
                    x1, y1 = points[connection[1]]
                    cv.line(pimg, (int(x0*  257/hand_image.shape[1]), int(y0*  257/hand_image.shape[0])), (int(x1* 257/hand_image.shape[1]), int(y1*  257/hand_image.shape[0])), CONNECTION_COLOR, THICKNESS)

        cv.imshow("Pose", pimg)

        cv.waitKey(1)
        frame_count += 1
        print('Average FPS: ', frame_count / (time.time() - start))

    cv.destroyAllWindows()