from __future__ import print_function
import cv2 as cv
import numpy as np
import time
from configuration.configurator import get_cameras_configurations, get_scene_configurations
from processing.processing import processingFrame, processing_points_on_image
from processing.calculating import calculate_speed, calculate_landing_point
from processing.support import printing_results, get_bg_method
import camera.fundamental_matrix as fm
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-camera_conf", "--camera_conf_path", help="Cameras configurations path")
parser.add_argument("-scene_conf", "--scene_conf_path", help="Scene configurations path")

np.set_printoptions(suppress=True)

start_time = time.time()

args = parser.parse_args()
paths, camera_names, cameras = get_cameras_configurations(args.camera_conf_path)
scene_conf = get_scene_configurations(args.scene_conf_path)
FRAMES_COUNT = len(paths)

f_matrix_list = []
p_matrix_list = []
backSub = []

for i in range(1, len(cameras)):
    f_matrix_list.append(fm.create_fundamental_matrix(camera1=cameras[0], camera2=cameras[i]))

for camera in cameras:
    p_matrix_list.append(camera.p_matrix)
    backSub.append(get_bg_method(scene_conf.bg_method))

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
vectors_buff = []
frame_numbers_buff = []
frame_duration = 1 / scene_conf.frames_per_second

while True:
    frames = [capture.read()[1] for capture in captures]
    for frame in frames:
        if frame is None:
            isEnd = True
            break
    if isEnd:
        break

    frame_number = captures[0].get(cv.CAP_PROP_POS_FRAMES)

    for i, frame in enumerate(frames):
        pts, extreme_pts, cnts_sizes = processingFrame(frame, backSub[i], i, frame_number, horizon_line_y,
                                                       horizon_line_lower_limit, horizon_line_upper_limit,
                                                       scene_conf.contour_min_size, scene_conf.contour_large_size, scene_conf.size_difference)
        points.append(pts)
        extreme_points.append(extreme_pts)
        if i == 0:
            contours_sizes = cnts_sizes

    middle_point, right_point, left_point = processing_points_on_image(points, 0, f_matrix_list, p_matrix_list,
                                                                       frame_number, frames,
                                                                       extreme_points, contours_sizes)

    if middle_point is not None:
        middle_point = middle_point[0][:3]
        middle_point = np.array([middle_point[2], middle_point[1], middle_point[0]])
        if middle_point[2] < scene_conf.close_range:
            middle_point = ((left_point + right_point) / 2)[0][:3]
            middle_point = np.array([middle_point[2], middle_point[1] - scene_conf.v_offset, middle_point[0] - scene_conf.h_offset])

    if len(vectors_buff) == scene_conf.buff_size:
        last_frame, vector_prev = frame_numbers_buff[0], vectors_buff[0]
        if middle_point is not None:
            frame_offset = frame_number - last_frame
            velocity = calculate_speed(middle_point, vector_prev, frame_offset, frame_duration)
            z, x = calculate_landing_point(vectors_buff, scene_conf.y_plane)
            printing_results(frame_number, middle_point, frame_offset, frame_duration, velocity, x, z)

    if len(vectors_buff) < scene_conf.buff_size:
        if middle_point is not None:
            vectors_buff.append(middle_point)
            frame_numbers_buff.append(frame_number)
    else:
        if middle_point is not None:
            vectors_buff.pop(0)
            frame_numbers_buff.pop(0)
            vectors_buff.append(middle_point)
            frame_numbers_buff.append(frame_number)

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
