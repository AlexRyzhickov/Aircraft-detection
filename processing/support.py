from configuration.configurator import BgMethod
import cv2 as cv

history = 400

def printing_results(frame_number, position, frame_offset, frame_duration, velocity, x, z):
    print("______________________")
    print("Frame: ", frame_number)
    print("Position in space", position)
    print("Velocity calculate for last ", (frame_offset * frame_duration), " s")
    print("Velocity Vector: ", velocity)
    print("Landing coordinates x:", x, "z", z)
    print("______________________")

def get_bg_method(method: BgMethod):
    if method == BgMethod.KNN:
        return cv.createBackgroundSubtractorKNN()
    if method == BgMethod.CNT:
        return cv.bgsegm.createBackgroundSubtractorCNT()
    if method == BgMethod.MOG:
        return cv.bgsegm.createBackgroundSubtractorMOG(history = history)
    if method == BgMethod.MOG2:
        return cv.createBackgroundSubtractorMOG2()
