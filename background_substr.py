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
points = []
extreme_points = []
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
        pts, extreme_pts = processingFrame(frame, backSub[i], i, frame_number, horizon_line_y, horizon_line_lower_limit, horizon_line_upper_limit)
        points.append(pts)
        extreme_points.append(extreme_pts)

    processing_points_on_image(points, 0, f_matrix_list, p_matrix_list, frame_number, frames, extreme_points)

    points.clear()
    extreme_points.clear()

    for i, frame in enumerate(frames):
        cv.imshow(camera_names[i], frame)

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

for capture in captures:
    capture.release()
cv.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
