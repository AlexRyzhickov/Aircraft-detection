from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imutils
import time
import math

from configuration.configurator import get_cameras_configurations
from processing.processing import processingFrame

# paths = ["./data/10_08_12/941_075632_0_tl.avi", "./data/10_08_12/941_075632_1_tc.avi", "./data/10_08_12/941_075632_0_tr.avi"]
# paths = ["./data/28_01_14/675_100834_0_tl.avi", "./data/28_01_14/675_100834_0_tc.avi", "./data/28_01_14/675_100834_0_tr.avi"]
# camera_names = ["Left Camera", "Center Camera", "Right Camera"]
# paths = ["./data/synthetic_data/left.avi", "./data/synthetic_data/right.avi"]
# camera_names = ["Left Camera", "Right Camera"]

# paths = ["./data/unity/movie_016.mp4", "./data/unity/movie_017.mp4"]
# camera_names = ["Left Camera", "Right Camera"]

# paths = ["./data/unity_data_1/movie_021.mp4", "./data/unity_data_1/movie_022.mp4"]
paths = ["./data/unity_data_1/movie_022.mp4", "./data/unity_data_1/movie_023.mp4"]
#
# camera_names = ["Left Camera", "Right Camera"]

FRAMES_COUNT = len(paths)

np.set_printoptions(suppress=True)
my_file = open("some.txt", "w")

start_time = time.time()

import camera.camera as cm
import camera.fundamental_matrix as fm

# c1 = cm.Camera(1024, 640, 40e-3, 22e-6, -58, 16, -13, 5, 10)
# c2 = cm.Camera(1024, 640, 40e-3, 22e-6, -58, 16, 27, 5, -10)


# c1 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, 0, 18, 0, 0, 0)
# c1 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, -16, 18, -15, 0, 0)
# c2 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, -16, 18, 15, 0, 0)

paths, camera_names, cameras = get_cameras_configurations("./data/unity_data_1/configurations_two_cameras.json")

c1 = cameras[0]
c2 = cameras[1]

# c1 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, -16, 18, -15, 5, -10)
# c2 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, -16, 18, 15, 5, 10)

# print(c1.p_matrix)
# print(c2.p_matrix)

# c1.p_matrix = np.array([[9.84807753e-01, 0.00000000e+00, -1.73648178e-01, 5.48614234e+01],
#                         [1.51344359e-02, 9.96194698e-01, 8.58316512e-02, -1.39455064e+01],
#                         [1.72987394e-01, -8.71557427e-02, 9.81060262e-01, 2.41815441e+01]])
#
# c2.p_matrix = np.array([[9.84807753e-01, 0.00000000e+00, 1.73648178e-01, 5.24303489e+01],
#                         [-1.51344359e-02, 9.96194698e-01, 8.58316512e-02, -1.91343670e+01],
#                         [-1.72987394e-01, -8.71557427e-02, 9.81060262e-01, -3.51274040e+01]])

# c1.p_matrix = np.array([[9.84807753e-01, 0.00000000e+00, -1.73648178e-01, 2.73090648e+00],
#                         [1.51344359e-02, 9.96194698e-01, 8.58316512e-02, -1.07641317e+01],
#                         [1.72987394e-01, -8.71557427e-02, 9.81060262e-01, 6.05448232e+01]])
#
# c2.p_matrix = np.array([[9.84807753e-01, 0.00000000e+00, -1.73648178e-01, -3.66614036e+01],
#                         [1.51344359e-02, 9.96194698e-01, 8.58316512e-02, -1.13695092e+01],
#                         [1.72987394e-01, -8.71557427e-02, 9.81060262e-01, 5.36253275e+01]])

# print(c1.rotation_matrix)
# print(c2.rotation_matrix)


# c1 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, -16, 18, -15, 5, 10)
# c2 = cm.Camera(1920, 1080, 3.507403e-3, 0.00000375, -16, 18, 15, 5, -10)
#
# print(c1.intrinsic_matrix)
# print(c2.intrinsic_matrix)
#
# c1 = cm.Camera(1920, 1080, 941, 1, -16, 18, -15, 5, 10)
# c2 = cm.Camera(1920, 1080, 941, 1, -16, 18, 15, 5, -10)
#
# print(c1.intrinsic_matrix)
# print(c2.intrinsic_matrix)

# exit(0)

F = fm.create_fundamental_matrix(camera1=c1, camera2=c2)

backSub = [cv.createBackgroundSubtractorKNN(), cv.createBackgroundSubtractorKNN(), cv.createBackgroundSubtractorKNN()]

horizon_line_y = [0] * FRAMES_COUNT
horizon_line_lower_limit = [0] * FRAMES_COUNT
horizon_line_upper_limit = [0] * FRAMES_COUNT

captures = [cv.VideoCapture(path) for path in paths]

for capture in captures:
    if not capture.isOpened():
        print('Unable to open: ')
        exit(0)

isEnd = False
points = []

while True:
    frames = [capture.read()[1] for capture in captures]

    for frame in frames:
        if frame is None:
            isEnd = True
            break

    if isEnd:
        break

    frame_number = captures[0].get(cv.CAP_PROP_POS_FRAMES)

    # if frame_number < 1380:
    #     continue

    for i, frame in enumerate(frames):
        pts = processingFrame(frame, backSub[i], i, frame_number, horizon_line_y, horizon_line_lower_limit,
                              horizon_line_upper_limit)
        points.append(pts)

    points_size = len(points)
    for i in range(points_size):
        camera_points = points[i]
    camera_points1 = points[0]
    camera_points2 = points[1]

    if camera_points1 is not None and camera_points2 is not None:
        for point1 in camera_points1:
            for point2 in camera_points2:
                dst = float(np.dot(np.dot(point1, F), point2.transpose()))
                if abs(round(dst, 1)) == 0.0:
                    point_frame_1 = np.array([[point1[0], point1[1]]])
                    point_frame_2 = np.array([[point2[0], point2[1]]])

                    # print(point_frame_1.shape)
                    # print(point_frame_2.shape)

                    # point_frame_1 = np.array([[float(point1[0]), float(point1[1])]])
                    # point_frame_2 = np.array([[float(point2[0]), float(point2[1])]])

                    p = cv.triangulatePoints(c1.p_matrix, c2.p_matrix, point_frame_1.T, point_frame_2.T)
                    # however, homgeneous point is returned
                    p /= p[3]
                    new_p = p.T[0]
                    print("frame ", frame_number, new_p)
                    # print(p.T)

                    # print(c1.p_matrix)
                    # print(c2.p_matrix)

                    term = str(p.T) + " Frame: " + str(int(frame_number)) + " Angle: " + str(
                        (math.atan((p[1] - 16) / (p[0])) * 180 / math.pi)) + " Distance " + \
                           str(math.sqrt(math.pow(-new_p[0] - (-58), 2) + math.pow(-new_p[1] - (16), 2) + math.pow(
                               new_p[2] - (-13), 2))) + "\n"
                    my_file.write(term)

                    # print(c1.p_matrix)
                    # print(c2.p_matrix)
                    # print(point_frame_1)
                    # print(point_frame_2)

                    cv.circle(frames[0], (int(point1[0]), int(point1[1])), 4, (255, 0, 0), -1)
                    cv.circle(frames[1], (int(point2[0]), int(point2[1])), 4, (255, 0, 0), -1)

    points.clear()

    for i, frame in enumerate(frames):
        cv.imshow(camera_names[i], frame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

for capture in captures:
    capture.release()
cv.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
