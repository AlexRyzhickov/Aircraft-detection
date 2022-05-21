import numpy as np
import cv2 as cv
import imutils


def processingFrame(frame,
                    backSubKNN,
                    pos,
                    frame_number: int,
                    horizon_line_y,
                    horizon_line_lower_limit,
                    horizon_line_upper_limit
                    ):
    fgMaskKNN = backSubKNN.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(frame_number), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

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

            if not horizontal_size / vertical_size > 6:
                m00, m01, m10, is_not_zero = get_moments(c)
                if is_not_zero:
                    cX = int(m10 / m00)
                    cY = int(m01 / m00)
                    if cY < horizon_line_lower_limit[pos] or cY > horizon_line_upper_limit[pos]:
                        cv.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        # cv.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
                        cv.circle(frame, (left[0], left[1]), 1, (0, 0, 255), -1)
                        cv.circle(frame, (right[0], right[1]), 1, (0, 0, 255), -1)
                        cv.circle(frame, (top[0], top[1]), 1, (0, 0, 255), -1)
                        cv.circle(frame, (bottom[0], bottom[1]), 1, (0, 0, 255), -1)

                        centers.append([cX, cY])
            elif c.size > 50:
                if horizon_line_y[pos] == -1:
                    m00, m01, _, is_not_zero = get_moments(c)
                    if is_not_zero:
                        cY = int(m01 / m00)
                        horizon_line_y[pos] = cY
                        horizon_line_lower_limit[pos] = horizon_line_y[pos] - 10
                        horizon_line_upper_limit[pos] = horizon_line_y[pos] + 10
                else:
                    m00, m01, _, is_not_zero = get_moments(c)
                    if is_not_zero:
                        cY = int(m01 / m00)
                        horizon_line_y[pos] = (horizon_line_y[pos] + cY) / 2
                        horizon_line_lower_limit[pos] = horizon_line_y[pos] - 10
                        horizon_line_upper_limit[pos] = horizon_line_y[pos] + 10

    if len(centers) > 0:
        pts1 = np.array(centers)
        ones = np.ones((len(centers), 1))
        pts1_final = np.append(pts1, ones, axis=1)
        return pts1_final
    else:
        return []


def get_moments(c):
    M = cv.moments(c)
    m00 = M["m00"]
    m01 = M["m01"]
    m10 = M["m10"]
    return m00, m01, m10, m00 != 0


class Pair:
    def __init__(self, point_position_on_main_camera, point_position_on_another_camera, dst, camera_number):
        self.point_position_on_main_camera = point_position_on_main_camera
        self.point_position_on_another_camera = point_position_on_another_camera
        self.dst = dst
        self.camera_number = camera_number


def increment_frequency(checker_dict, key):
    value = checker_dict.get(key)
    if value is None:
        checker_dict[key] = 1
    else:
        checker_dict[key] = value + 1


# def processing_points_on_image(points_on_main_camera, points_on_another_cameras, f_matrix_list, count_cameras):
#     pairs = {}
#     checker_dict = {}
#     for i, point_on_main_camera in enumerate(points_on_main_camera):
#         for camera_number, points in enumerate(points_on_another_cameras):
#             best_pair = Pair(0, 0, 0, 0)
#             min_dst = 1
#             for j, point in enumerate(points):
#                 dst = float(np.dot(np.dot(point_on_main_camera, f_matrix_list[camera_number]), point.transpose()))
#                 if abs(round(dst, 1)) == 0.0:
#                     if abs(dst) < min_dst:
#                         min_dst = dst
#                         best_pair = Pair(i, j, dst, camera_number)
#             if not min_dst == 1:
#                 key = (best_pair.point_position_on_another_camera, camera_number)
#                 pair = pairs.get(key)
#                 if pair is None:
#                     pairs[key] = best_pair
#                     increment_frequency(checker_dict=checker_dict, key=best_pair.point_position_on_main_camera)
#                 else:
#                     if pair.dst > best_pair.dst:
#                         pairs[key] = best_pair
#     values = pairs.values()

def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3 * n, 4 + n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3 * i:3 * i + 3, :4] = p
        M[3 * i:3 * i + 3, 4 + i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def processing_points_on_image(points, main_camera_position, f_matrix_list, p_matrix_list, frame_number, frames):
    main_camera_points = points[main_camera_position]

    if len(main_camera_points) == 0: return

    camera_points = []
    for i in range(len(points)):
        if i != main_camera_position:
            if len(points[i]) > 0:
                camera_points.append(points[i])
            else:
                return

    for main_point in main_camera_points:
        point_set = []
        for i, points2 in enumerate(camera_points):
            min_dst = 1
            best_point = np.array([[0, 0, 1]])
            for point in points2:
                dst = float(np.dot(np.dot(main_point, f_matrix_list[i]), point.transpose()))
                if abs(dst) < abs(min_dst):
                    min_dst = dst
                    best_point[0][0] = point[0]
                    best_point[0][1] = point[1]
            if not min_dst == 1:
                point_set.append(best_point)
        if len(point_set) == len(points) - 1:
            point_set.insert(0, np.array([[main_point[0], main_point[1], 1]]))
            p = triangulate_nviews(p_matrix_list, point_set)
            # print('________')
            print("frame ", frame_number, p)

            # point_frame_1 = np.array([[float(point_set[0][0][0]), float(point_set[0][0][0])]])
            # point_frame_2 = np.array([[float(point_set[1][0][0]), float(point_set[1][0][0])]])
            #
            # p = cv.triangulatePoints(p_matrix_list[0], p_matrix_list[1], point_frame_1.T, point_frame_2.T)
            # # however, homgeneous point is returned
            # p /= p[3]
            # new_p = p.T[0]
            # print("frame ", frame_number, new_p)
            # print('________')

            for j, frame in enumerate(frames):
                cv.circle(frame, (int(point_set[j][0][0]), int(point_set[j][0][1])), 4, (255, 0, 0), -1)

