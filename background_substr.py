from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imutils
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

backSubKNN = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture("./941_075632_0_tr.avi")

if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

from dataclasses import dataclass
from queue import Queue


@dataclass
class Pixel:
    x: int
    y: int


def calc_key_from_int_pair(x, y: int):
    return 10 * x + y


def df(queue, dict, value, x, y):
    if dict.get(calc_key_from_int_pair(x, y)) is not None:
        queue.put(Pixel(x, y))
        dict[value] = 1


def produce_pixels_areas(frame):
    width, height = np.shape(frame)
    dict = {}
    for i in range(0, width):
        for j in range(0, height):
            a = 2
            # dict[i * 10 + j] = 2
            # if frame[i][j] > 0 and dict.get(calc_key_from_int_pair(i, j)) is not None:
            #     pixel_area = [Pixel(i, j)]
                # dict[calc_key_from_int_pair(i, j)] = 1
                # queue = Queue()

                # if i > 0 and frame[i - 1][j] > 0:
                #     value = calc_key_from_int_pair(i - 1, j)
                #     df(queue, dict, value, i - 1, j)
                # if i < width and frame[i + 1][j] > 0:
                #     value = calc_key_from_int_pair(i + 1, j)
                #     df(queue, dict, value, i + 1, j)
                # if j > 0 and frame[i][j - 1] > 0:
                #     value = calc_key_from_int_pair(i, j - 1)
                #     df(queue, dict, value, i, j - 1)
                # if j < height and frame[i][j + 1] > 0:
                #     value = calc_key_from_int_pair(i, j + 1)
                #     df(queue, dict, value, i, j + 1)

                # if i > 0 and frame[i - 1][j] > 0 and dict.get(calc_key_from_int_pair(i - 1, j)) is not None:
                #     queue.put(Pixel(i - 1, j))
                #     dict[calc_key_from_int_pair(i - 1, j)] = 1
                # if i < width and frame[i + 1][j] > 0 and dict.get(Pixel(i + 1, j)) is not None:
                #     queue.put(Pixel(i + 1, j))
                #     dict[Pixel(i + 1, j)] = 1
                # if j > 0 and frame[i][j - 1] > 0 and dict.get(Pixel(i, j - 1)) is not None:
                #     queue.put(Pixel(i, j - 1))
                #     dict[Pixel(i, j - 1)] = 1
                # if j < height and frame[i][j + 1] > 0 and dict.get(Pixel(i, j + 1)) is not None:
                #     queue.put(Pixel(i, j + 1))
                #     dict[Pixel(i - 1, j)] = 1

                # while not queue.empty():
                #     pixel = queue.get()
                #     pixel_area.append(pixel)
                #
                #     x = pixel.x
                #     y = pixel.y
                #
                #     if x > 0 and frame[x - 1][y] > 0:
                #         value = calc_key_from_int_pair(x - 1, y)
                #         df(queue, dict, value, x - 1, y)
                #     if x < width and frame[x + 1][y] > 0:
                #         value = calc_key_from_int_pair(x + 1, y)
                #         df(queue, dict, value, x + 1, y)
                #     if y > 0 and frame[x][y - 1] > 0:
                #         value = calc_key_from_int_pair(x, y - 1)
                #         df(queue, dict, value, x, y - 1)
                #     if y < height and frame[x][y + 1] > 0:
                #         value = calc_key_from_int_pair(x, y + 1)
                #         df(queue, dict, value, x, y + 1)

                    # if x > 0 and frame[x - 1][y] > 0:
                    #     queue.put(Pixel(x - 1, y))
                    # if x < width and frame[x + 1][y] > 0:
                    #     queue.put(Pixel(x + 1, y))
                    # if y > 0 and frame[x][y - 1] > 0:
                    #     queue.put(Pixel(x, y - 1))
                    # if y < height and frame[x][y + 1] > 0:
                    #     queue.put(Pixel(x, y + 1))



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
    cnts = cv.findContours(fgMaskKNN.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv.moments(c)
        # m00 = M["00"]
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    cv.rectangle(frameKNN, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameKNN, "KNN", (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    def concat_tile(im_list_2d):
        return cv.vconcat([cv.hconcat(im_list_h) for im_list_h in im_list_2d])


    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', frameKNN)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
capture.release()
cv.destroyAllWindows()
