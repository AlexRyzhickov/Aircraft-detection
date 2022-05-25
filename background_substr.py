from __future__ import print_function
import cv2 as cv
import numpy as np
import time
import math
from configuration.configurator import get_cameras_configurations
from processing.processing import processingFrame, processing_points_on_image
from processing.movement import calculate_speed
import camera.fundamental_matrix as fm

np.set_printoptions(suppress=True)

file1 = open("some1.txt", "w")
file2 = open("some2.txt", "w")
file3 = open("some3.txt", "w")

start_time = time.time()

# paths, camera_names, cameras = get_cameras_configurations("./data/unity_data_1/configurations_two_cameras.json")
# paths, camera_names, cameras = get_cameras_configurations("./data/unity_data_2/configurations_two_camera.json")
paths, camera_names, cameras = get_cameras_configurations("./data/unity_data_2/configurations.json")
FRAMES_COUNT = len(paths)

f_matrix_list = []
p_matrix_list = []
backSub = []

for i in range(1, len(cameras)):
    f_matrix_list.append(fm.create_fundamental_matrix(camera1=cameras[0], camera2=cameras[i]))

for camera in cameras:
    p_matrix_list.append(camera.p_matrix)
    backSub.append(cv.createBackgroundSubtractorKNN())

horizon_line_y = [0] * FRAMES_COUNT
horizon_line_lower_limit = [0] * FRAMES_COUNT
horizon_line_upper_limit = [0] * FRAMES_COUNT

captures = [cv.VideoCapture(path) for path in paths]

for capture in captures:
    if not capture.isOpened():
        print('Unable to open: ')
        exit(0)

isEnd = False
isFistTimeMeasurement = True
points = []
extreme_points = []
contours_sizes = []
# coordinate_vector_prev = np.array([[0, 0, 0, 0]])
# last_frame = 0
vectors_buff = []
buff_size = 50
#
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
        pts, extreme_pts, cnts_sizes = processingFrame(frame, backSub[i], i, frame_number, horizon_line_y,
                                                       horizon_line_lower_limit, horizon_line_upper_limit)
        points.append(pts)
        extreme_points.append(extreme_pts)
        if i == 0:
            contours_sizes = cnts_sizes

    middle_point, right_point, left_point = processing_points_on_image(points, 0, f_matrix_list, p_matrix_list,
                                                                       frame_number, frames,
                                                                       extreme_points, contours_sizes)
    file1.write("frame: " + str(int(frame_number)) + " " + str(middle_point) + "\n")
    file2.write("frame: " + str(int(frame_number)) + " " + str(right_point) + "\n")
    file3.write("frame: " + str(int(frame_number)) + " " + str(left_point) + "\n")

    # if frame_number != 0 and middle_point is not None and frame_number % 25 == 0:
    #     if isFistTimeMeasurement:
    #         coordinate_vector_prev = middle_point
    #         last_frame = frame_number
    #         isFistTimeMeasurement = False
    #     else:
    #         calculate_speed(middle_point, coordinate_vector_prev, frame_number - last_frame, 1 / 50, frame_number)
    #         coordinate_vector_prev = middle_point
    #         last_frame = frame_number
    if len(vectors_buff) == buff_size:
        last_frame, vector_prev = vectors_buff[0]
        if vector_prev  is not None and middle_point is not None:
            calculate_speed(middle_point, vector_prev, frame_number - last_frame, 1 / buff_size, frame_number)

    if len(vectors_buff) < buff_size:
        vectors_buff.append((frame_number, middle_point))
    else:
        vectors_buff.pop(0)
        vectors_buff.append((frame_number, middle_point))

    points.clear()
    extreme_points.clear()
    contours_sizes.clear()

    for i, frame in enumerate(frames):
        cv.imshow(camera_names[i], frame)

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

for capture in captures:
    capture.release()
cv.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
