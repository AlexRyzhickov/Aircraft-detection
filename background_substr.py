from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imutils


backSubKNN_1 = cv.createBackgroundSubtractorKNN()
backSubKNN_2 = cv.createBackgroundSubtractorKNN()
backSubKNN_3 = cv.createBackgroundSubtractorKNN()


capture_1 = cv.VideoCapture("./data/10_08_12/941_075632_0_tl.avi")
capture_2 = cv.VideoCapture("./data/10_08_12/941_075632_0_tr.avi")
capture_3 = cv.VideoCapture("./data/10_08_12/941_075632_1_tc.avi")

if not capture_1.isOpened() or not capture_2.isOpened() or not capture_3.isOpened():
    print('Unable to open: ')
    exit(0)

def processingFrame(frame, backSubKNN):
    fgMaskKNN = backSubKNN.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture_1.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    frameKNN = cv.cvtColor(fgMaskKNN, cv.COLOR_GRAY2RGB)

    cv.rectangle(frameKNN, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameKNN, "KNN", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frameKNN

while True:
    _, frame_1 = capture_1.read()
    _, frame_2 = capture_2.read()
    _, frame_3 = capture_3.read()

    if frame_1 is None or frame_2 is None or frame_3 is None:
        break

    frameKNN1 = processingFrame(frame_1, backSubKNN_1)
    frameKNN2 = processingFrame(frame_2, backSubKNN_2)
    frameKNN3 = processingFrame(frame_3, backSubKNN_3)



    cv.imshow('Frame', frame_1)
    cv.imshow('FG Mask', frameKNN1)
    cv.imshow('FG Mask2', frameKNN2)
    cv.imshow('FG Mask3', frameKNN3)


    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
capture_1.release()
cv.destroyAllWindows()
