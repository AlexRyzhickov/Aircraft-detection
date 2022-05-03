from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imutils

backSubKNN_1 = cv.createBackgroundSubtractorKNN()
backSubKNN_2 = cv.createBackgroundSubtractorKNN()
backSubKNN_3 = cv.createBackgroundSubtractorKNN()

horizon_line_y = [0, 0, 0]
horizon_line_lower_limit = [0, 0, 0]
horizon_line_upper_limit = [0, 0, 0]

capture_1 = cv.VideoCapture("./data/10_08_12/941_075632_0_tl.avi")
capture_3 = cv.VideoCapture("./data/10_08_12/941_075632_1_tc.avi")
capture_2 = cv.VideoCapture("./data/10_08_12/941_075632_0_tr.avi")

if not capture_1.isOpened() or not capture_2.isOpened() or not capture_3.isOpened():
    print('Unable to open: ')
    exit(0)


def processingFrame(frame, backSubKNN, pos):
    fgMaskKNN = backSubKNN.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture_1.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    frameKNN = cv.cvtColor(fgMaskKNN, cv.COLOR_GRAY2RGB)

    cv.rectangle(frameKNN, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameKNN, "KNN", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    frame_with_cnt = frame.copy()
    cnts = cv.findContours(fgMaskKNN, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if c.size > 10:
            left = tuple(c[c[:, :, 0].argmin()][0])
            right = tuple(c[c[:, :, 0].argmax()][0])
            top = tuple(c[c[:, :, 1].argmin()][0])
            bottom = tuple(c[c[:, :, 1].argmax()][0])

            horizontal_size = abs(right[0] - left[0])
            vertical_size = abs(top[1] - bottom[1])

            if not horizontal_size / vertical_size > 3:
                M = cv.moments(c)
                m00 = M["m00"]
                if m00 != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if cY < horizon_line_y[pos] - 10 or cY > horizon_line_y[pos] + 10:
                        cv.drawContours(frame, [c], -1, (0, 255, 0), 1)
                        cv.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
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

    return frameKNN


while True:
    _, frame_1 = capture_1.read()
    _, frame_2 = capture_2.read()
    _, frame_3 = capture_3.read()

    if frame_1 is None or frame_2 is None or frame_3 is None:
        break

    frameKNN1 = processingFrame(frame_1, backSubKNN_1, 0)
    frameKNN2 = processingFrame(frame_2, backSubKNN_2, 1)
    frameKNN3 = processingFrame(frame_3, backSubKNN_3, 2)

    cv.imshow('Frame1', frame_1)
    cv.imshow('Frame2', frame_2)
    cv.imshow('Frame3', frame_3)

    # cv.imshow('FG Mask', frameKNN1)
    # cv.imshow('FG Mask2', frameKNN2)
    # cv.imshow('FG Mask3', frameKNN3)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
capture_1.release()
cv.destroyAllWindows()
