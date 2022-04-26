from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
# if args.algo == 'MOG2':
# backSub = cv.createBackgroundSubtractorMOG(history = 400, varThreshold = 50)
backSubMog = cv.bgsegm.createBackgroundSubtractorMOG(history = 400)
backSubKNN = cv.createBackgroundSubtractorKNN()
backSubCNT = cv.bgsegm.createBackgroundSubtractorCNT()

# backSub = cv.createBackgroundSubtractorKNN()
# backSub = cv.bgsegm.createBackgroundSubtractorCNT()
# backSub = cv.bgsegm.createBackgroundSubtractorLSBP()
# backSub = cv.bgsegm.createBackgroundSubtractorGSOC()

# else:
#     backSub = cv.createBackgroundSubtractorKNN()
## [create]

## [capture]
# capture = cv.VideoCapture("../941_075632_1_tc.avi")
capture = cv.VideoCapture("../941_075632_0_tr.avi")


width = int(capture.get(3)*2)
height = int(capture.get(4)*2)
size = (width, height)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('your_video.avi', fourcc, 25, size)


if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMaskMOG = backSubMog.apply(frame)
    fgMaskKNN = backSubKNN.apply(frame)
    fgMaskCNT = backSubCNT.apply(frame)

    ## [apply]

    # gr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # bl = cv.medianBlur(gr, 5)
    # fgMask = cv.Canny(gr, 10, 250)


    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    # vis = np.concatenate((frame, cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)), axis=1)

    frameMOG = cv.cvtColor(fgMaskMOG, cv.COLOR_GRAY2RGB)
    frameKNN = cv.cvtColor(fgMaskKNN, cv.COLOR_GRAY2RGB)
    frameCNT = cv.cvtColor(fgMaskCNT, cv.COLOR_GRAY2RGB)

    cv.rectangle(frameMOG, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameMOG, "MOG", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.rectangle(frameKNN, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameKNN, "KNN", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.rectangle(frameCNT, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameCNT, "CNT", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    def concat_tile(im_list_2d):
        return cv.vconcat([cv.hconcat(im_list_h) for im_list_h in im_list_2d])


    # im1_s = cv2.resize(im1, dsize=(0, 0), fx=0.5, fy=0.5)
    im_tile = concat_tile([[frame, frameMOG],[frameKNN, frameCNT]])

    # vis = cv.vconcat([[frame, frameMOG],[frameKNN, frameCNT]])
    out.write(im_tile)

    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMaskMOG)
    cv.imshow('ff', im_tile)
    ## [show]

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
capture.release()
out.release()
cv.destroyAllWindows()