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

            if not horizontal_size / vertical_size > 4.5:
                m00, m01, m10, is_not_zero = get_moments(c)
                if is_not_zero:
                    cX = int(m10 / m00)
                    cY = int(m01 / m00)
                    if cY < horizon_line_lower_limit[pos] or cY > horizon_line_upper_limit[pos]:
                        cv.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        cv.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
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
        return None


def get_moments(c):
    M = cv.moments(c)
    m00 = M["m00"]
    m01 = M["m01"]
    m10 = M["m10"]
    return m00, m01, m10, m00 != 0
