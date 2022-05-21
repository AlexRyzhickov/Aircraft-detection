from __future__ import print_function
import cv2 as cv
import numpy as np
import time
import math
from configuration.configurator import get_cameras_configurations
from processing.processing import processingFrame, processing_points_on_image
import camera.fundamental_matrix as fm

np.set_printoptions(suppress=True)
my_file = open("some.txt", "w")

start_time = time.time()

paths, camera_names, cameras = get_cameras_configurations("./data/unity_data_1/configurations_two_cameras.json")
FRAMES_COUNT = len(paths)

# c1 = cameras[0]
# c2 = cameras[1]
# c3 = cameras[2]

# F = fm.create_fundamental_matrix(camera1=c1, camera2=c2)
# F2 = fm.create_fundamental_matrix(camera1=c1, camera2=c3)

# f_matrix_list = [F, F2]
# p_matrix_list = [c1.p_matrix, c2.p_matrix, c3.p_matrix]
# p_matrix_list = [c1.p_matrix, c2.p_matrix]

f_matrix_list = []
p_matrix_list = []

for i in range(1, len(cameras)):
    f_matrix_list.append(fm.create_fundamental_matrix(camera1=cameras[0], camera2=cameras[i]))

for camera in cameras:
    p_matrix_list.append(camera.p_matrix)

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

    # if frame_number < 553:
    #     continue

    for i, frame in enumerate(frames):
        pts = processingFrame(frame, backSub[i], i, frame_number, horizon_line_y, horizon_line_lower_limit,
                              horizon_line_upper_limit)
        points.append(pts)

    # points_size = len(points)
    # for i in range(points_size):
    #     camera_points = points[i]
    # camera_points1 = points[0]
    # camera_points2 = points[1]
    # camera_points3 = points[2]
    #
    # if camera_points1 is not None and camera_points3 is not None:
    #     for point1 in camera_points1:
    #         for point2 in camera_points3:
    #             dst = float(np.dot(np.dot(point1, F), point2.transpose()))
    #             if abs(round(dst, 1)) == 0.0:
    #                 point_frame_1 = np.array([[point1[0], point1[1]]])
    #                 point_frame_2 = np.array([[point2[0], point2[1]]])
    #
    #                 # print(point_frame_1.shape)
    #                 # print(point_frame_2.shape)
    #
    #                 # point_frame_1 = np.array([[float(point1[0]), float(point1[1])]])
    #                 # point_frame_2 = np.array([[float(point2[0]), float(point2[1])]])
    #
    #                 p = cv.triangulatePoints(c1.p_matrix, c3.p_matrix, point_frame_1.T, point_frame_2.T)
    #                 # however, homgeneous point is returned
    #                 p /= p[3]
    #                 new_p = p.T[0]
    #                 print("frame ", frame_number, new_p)
    #                 # print(p.T)
    #
    #                 # print(c1.p_matrix)
    #                 # print(c2.p_matrix)
    #
    #                 # term = str(p.T) + " Frame: " + str(int(frame_number)) + " Angle: " + str(
    #                 #     (math.atan((p[1] - 16) / (p[0])) * 180 / math.pi)) + " Distance " + \
    #                 #        str(math.sqrt(math.pow(-new_p[0] - (-58), 2) + math.pow(-new_p[1] - (16), 2) + math.pow(
    #                 #            new_p[2] - (-13), 2))) + "\n"
    #                 # my_file.write(term)
    #
    #                 # print(c1.p_matrix)
    #                 # print(c2.p_matrix)
    #                 # print(point_frame_1)
    #                 # print(point_frame_2)
    #
    #                 cv.circle(frames[0], (int(point1[0]), int(point1[1])), 4, (255, 0, 0), -1)
    #                 cv.circle(frames[2], (int(point2[0]), int(point2[1])), 4, (255, 0, 0), -1)


    processing_points_on_image(points, 0, f_matrix_list, p_matrix_list, frame_number, frames)

    points.clear()

    for i, frame in enumerate(frames):
        cv.imshow(camera_names[i], frame)

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

for capture in captures:
    capture.release()
cv.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
