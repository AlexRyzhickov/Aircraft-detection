from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imutils
import time
import math

# paths = ["./data/10_08_12/941_075632_0_tl.avi", "./data/10_08_12/941_075632_1_tc.avi", "./data/10_08_12/941_075632_0_tr.avi"]
# paths = ["./data/28_01_14/675_100834_0_tl.avi", "./data/28_01_14/675_100834_0_tc.avi", "./data/28_01_14/675_100834_0_tr.avi"]
paths = ["./data/synthetic_data/left.avi", "./data/synthetic_data/right.avi"]
camera_names = ["Left Camera", "Center Camera", "Right Camera"]

FRAMES_COUNT = len(paths)

start_time = time.time()


class Camera():
    def __init__(self, frame_width, frame_height, f, size_px):
        self.intrinsic_matrix = np.array(
            [[f / size_px, 0, frame_width / 2], [0, f / size_px, frame_height / 2], [0, 0, 1]])
        self.rotation, self.translation = np.eye(3), np.zeros(3)


c1 = Camera(1024, 640, 40e-3, 22e-6)
c2 = Camera(1024, 640, 40e-3, 22e-6)

c1.rotation = np.array([[-math.sin(math.pi / 18), 0, math.cos(math.pi / 18)],
                        [math.sin(math.pi / 36) * math.cos(math.pi / 18), - math.cos(math.pi / 36),
                         math.sin(math.pi / 36) * math.sin(math.pi / 18)],
                        [math.cos(math.pi / 36) * math.cos(math.pi / 18), math.sin(math.pi / 36),
                         math.cos(math.pi / 36) * math.sin(math.pi / 18)]])
c1.translation = np.array([[-58], [16], [-13]])

c1.extrinsic_matrix = np.concatenate((c1.rotation, c1.translation), axis=1)
c1.p = np.dot(c1.intrinsic_matrix, c1.extrinsic_matrix)

c2.rotation = np.array([[math.sin(math.pi / 18), 0, math.cos(math.pi / 18)],
                        [math.sin(math.pi / 36) * math.cos(math.pi / 18), - math.cos(math.pi / 36),
                         -math.sin(math.pi / 36) * math.sin(math.pi / 18)],
                        [math.cos(math.pi / 36) * math.cos(math.pi / 18), math.sin(math.pi / 36),
                         -math.cos(math.pi / 36) * math.sin(math.pi / 18)]])
c2.translation = np.array([[-58], [16], [27]])

c2.extrinsic_matrix = np.concatenate((c2.rotation, c2.translation), axis=1)
c2.p = np.dot(c2.intrinsic_matrix, c2.extrinsic_matrix)


T = (c2.translation - c1.translation).reshape(3)
T_cross = np.array([[0, - T[2], T[1]], [T[2], 0, - T[0]], [-T[1], T[0], 0]])
E = np.dot(c1.rotation, np.dot(T_cross, np.transpose(c2.rotation)))
F = np.dot(np.linalg.inv(np.transpose(c1.intrinsic_matrix)), np.dot(E, np.linalg.inv(c2.intrinsic_matrix)))



backSub = [cv.createBackgroundSubtractorKNN(), cv.createBackgroundSubtractorKNN(), cv.createBackgroundSubtractorKNN()]

horizon_line_y = [0] * FRAMES_COUNT
horizon_line_lower_limit = [0] * FRAMES_COUNT
horizon_line_upper_limit = [0] * FRAMES_COUNT

captures = [cv.VideoCapture(path) for path in paths]

for capture in captures:
    if not capture.isOpened():
        print('Unable to open: ')
        exit(0)


def processingFrame(frame, backSubKNN, pos):
    fgMaskKNN = backSubKNN.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(captures[0].get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    frameKNN = cv.cvtColor(fgMaskKNN, cv.COLOR_GRAY2RGB)

    cv.rectangle(frameKNN, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameKNN, "KNN", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cnts = cv.findContours(fgMaskKNN, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centers = []

    for c in cnts:
        if c.size > 10:
            left = tuple(c[c[:, :, 0].argmin()][0])
            right = tuple(c[c[:, :, 0].argmax()][0])
            top = tuple(c[c[:, :, 1].argmin()][0])
            bottom = tuple(c[c[:, :, 1].argmax()][0])

            horizontal_size = abs(right[0] - left[0])
            vertical_size = abs(top[1] - bottom[1])

            if not horizontal_size / vertical_size > 4.5:
                M = cv.moments(c)
                m00 = M["m00"]
                if m00 != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if cY < horizon_line_lower_limit[pos] or cY > horizon_line_upper_limit[pos]:
                        cv.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        cv.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
                        centers.append([cX, cY])
            elif c.size > 50:
                if horizon_line_y[pos] == -1:
                    M = cv.moments(c)
                    m00 = M["m00"]
                    if m00 != 0:
                        cY = int(M["m01"] / M["m00"])
                        horizon_line_y[pos] = cY
                        horizon_line_lower_limit[pos] = horizon_line_y[pos] - 10
                        horizon_line_upper_limit[pos] = horizon_line_y[pos] + 10
                else:
                    M = cv.moments(c)
                    m00 = M["m00"]
                    if m00 != 0:
                        cY = int(M["m01"] / M["m00"])
                        horizon_line_y[pos] = (horizon_line_y[pos] + cY) / 2
                        horizon_line_lower_limit[pos] = horizon_line_y[pos] - 10
                        horizon_line_upper_limit[pos] = horizon_line_y[pos] + 10

    if len(centers) > 0:
        pts1 = np.array(centers)
        ones = np.ones((len(centers), 1))
        pts1_final = np.append(pts1, ones, axis=1)
        return pts1_final
    else:
        return None

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

    for i, frame in enumerate(frames):
        pts = processingFrame(frame, backSub[i], i)
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

                    p = cv.triangulatePoints(c1.p, c2.p, point_frame_1.T, point_frame_2.T)
                    # however, homgeneous point is returned
                    p /= p[3]
                    print(p.T)

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
