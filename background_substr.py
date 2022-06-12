from __future__ import print_function
import cv2 as cv
import numpy as np
import time
import math
from configuration.configurator import get_cameras_configurations
from processing.processing import processingFrame, processing_points_on_image
from processing.calculating import calculate_speed, calculate_landing_point
import camera.fundamental_matrix as fm

np.set_printoptions(suppress=True)
y_plane = 18
v_offset = 0.3791
h_offset = 10.8614
start_time = time.time()

paths, camera_names, cameras = get_cameras_configurations("./data/unity_data_2/configurations.json")
FRAMES_COUNT = len(paths)

f_matrix_list = []
p_matrix_list = []
backSub = []

for i in range(1, len(cameras)):
    f_matrix_list.append(fm.create_fundamental_matrix(camera1=cameras[0], camera2=cameras[i]))

for camera in cameras:
    p_matrix_list.append(camera.p_matrix)
    backSub.append(cv.bgsegm.createBackgroundSubtractorCNT())

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
buff_size = 50
frames_per_second = 50

start_times = []
after_capt_read_times = []
process_start_times = []
process_after_bg_times = []
process_end_times = []
before_calc_pos_times = []
end_times = []

while True:
    start_times.append(time.time())
    frames = [capture.read()[1] for capture in captures]
    for frame in frames:
        if frame is None:
            isEnd = True
            break
    if isEnd:
        break

    frame_number = captures[0].get(cv.CAP_PROP_POS_FRAMES)

    after_capt_read_times.append(time.time())
    for i, frame in enumerate(frames):
        pts, extreme_pts, cnts_sizes = processingFrame(frame, backSub[i], i, frame_number, horizon_line_y,
                                                       horizon_line_lower_limit, horizon_line_upper_limit,
                                                       process_start_times, process_after_bg_times, process_end_times)
        points.append(pts)
        extreme_points.append(extreme_pts)
        if i == 0:
            contours_sizes = cnts_sizes

    before_calc_pos_times.append(time.time())

    middle_point, right_point, left_point = processing_points_on_image(points, 0, f_matrix_list, p_matrix_list,
                                                                       frame_number, frames,
                                                                       extreme_points, contours_sizes)

    if middle_point is not None:
        middle_point = middle_point[0][:3]
        middle_point = np.array([middle_point[2], middle_point[1], middle_point[0]])
        print('_________________', middle_point)
        if middle_point[2] < 500:
            middle_point = ((left_point + right_point) / 2)[0][:3]
            middle_point = np.array([middle_point[2], middle_point[1] - v_offset, middle_point[0] - h_offset])
            print('_____________________________', middle_point)

    if len(vectors_buff) == buff_size:
        last_frame, vector_prev = frame_numbers_buff[0], vectors_buff[0]
        if middle_point is not None:
            calculate_speed(middle_point, vector_prev, frame_number - last_frame, 1 / frames_per_second, frame_number)
            z, x = calculate_landing_point(vectors_buff, y_plane)
            print("Coordinates x:", x, "z", z)

    if len(vectors_buff) < buff_size:
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

    end_times.append(time.time())

    # for i, frame in enumerate(frames):
    #     cv.imshow(camera_names[i], frame)
    #     if frame_number == 1490:
    #         cv.imwrite(f'./image{i}_frame{int(1560)}.jpg', frame)

    # keyboard = cv.waitKey(1)
    # if keyboard == 'q' or keyboard == 27:
    #     break

for capture in captures:
    capture.release()
cv.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))

bg_and_filtering_and_read_period = []
calculating_positions_period = []
read_frame_period = []
bg_and_filtering_period = []
bg_period = []
filtering_period = []

for item in zip(start_times, before_calc_pos_times, end_times, after_capt_read_times):
    a = item[1] - item[0]
    b = item[2] - item[1]
    c = item[2] - item[0]
    b1 = item[3] - item[0]
    b2 = item[1] - item[3]

    if a / c != 1 and b / c != 0:
        bg_and_filtering_and_read_period.append(a / c)
        calculating_positions_period.append(b / c)
        read_frame_period.append(b1 / c)
        bg_and_filtering_period.append(b2 / c)

print(len(start_times))
print(len(bg_and_filtering_and_read_period))
print(np.array(bg_and_filtering_and_read_period).mean(axis=0), "% - Bg, filtering and reading captures")
print(np.array(calculating_positions_period).mean(axis=0), "% - Calculating speed, position in space and landing coordinates")
print(np.array(read_frame_period).mean(axis=0), "% - Reading")
print(np.array(bg_and_filtering_period).mean(axis=0), "% - Bg, filtering")

for item in zip(process_start_times, process_after_bg_times, process_end_times):
    a = item[1] - item[0]
    process_start_times = item[2] - item[1]
    c = item[2] - item[0]

    bg_period.append(a / c)
    filtering_period.append(process_start_times / c)
print(np.array(bg_period).mean(axis=0), "% - Background subscription")
print(np.array(filtering_period).mean(axis=0), "% - Filtering")
