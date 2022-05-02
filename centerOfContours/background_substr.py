from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imutils

horizon_line_y = 0
horizon_line_lower_limit = 0
horizon_line_upper_limit = 0

backSubKNN = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture("../941_075632_0_tr.avi")

if not capture.isOpened():
    print('Unable to open: ')
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMaskKNN = backSubKNN.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    frameKNN = cv.cvtColor(fgMaskKNN, cv.COLOR_GRAY2RGB)
    # print(frameKNN[100][100])
    # produce_pixels_areas(fgMaskKNN)
    # thresh = cv.threshold(fgMaskKNN, 60, 255, cv.THRESH_BINARY)[1]

    frame_with_cnt = frame.copy()
    cnts = cv.findContours(fgMaskKNN, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
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
                    if cY < horizon_line_y - 10 or cY > horizon_line_y + 10:
                        cv.drawContours(frame_with_cnt, [c], -1, (0, 255, 0), 1)
                        cv.circle(frame_with_cnt, (cX, cY), 4, (255, 255, 255), -1)
            elif c.size > 50:
                if horizon_line_y == -1:
                    M = cv.moments(c)
                    m00 = M["m00"]
                    if m00 != 0:
                        cY = int(M["m01"] / M["m00"])
                        horizon_line_y = cY
                        horizon_line_lower_limit = horizon_line_y - 10
                        horizon_line_upper_limit = horizon_line_y + 10
                else:
                    M = cv.moments(c)
                    m00 = M["m00"]
                    if m00 != 0:
                        cY = int(M["m01"] / M["m00"])
                        horizon_line_y = (horizon_line_y + cY) / 2
                        horizon_line_lower_limit = horizon_line_y - 10
                        horizon_line_upper_limit = horizon_line_y + 10
                print(horizon_line_y)





    def concat_tile(im_list_2d):
        return cv.vconcat([cv.hconcat(im_list_h) for im_list_h in im_list_2d])


    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', frameKNN)
    cv.imshow("Image", frame_with_cnt)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
capture.release()
cv.destroyAllWindows()
