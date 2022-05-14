from __future__ import print_function
import cv2 as cv

paths = ["../data/synthetic_data/left.avi", "../data/synthetic_data/right.avi"]
captures = [cv.VideoCapture(path) for path in paths]

frame_number_for_shot = 486

for capture in captures:
    if not capture.isOpened():
        print('Unable to open: ')
        exit(0)

isEnd = False

while True:
    frames = [capture.read()[1] for capture in captures]

    for frame in frames:
        if frame is None:
            isEnd = True
            break

    if isEnd:
        break

    frame_number = captures[0].get(cv.CAP_PROP_POS_FRAMES)

    if frame_number == frame_number_for_shot:
        for i, frame in enumerate(frames):
            cv.imwrite(f'./frames/image{i}_frame{int(frame_number)}.jpg', frame)

    for i, frame in enumerate(frames):
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(frame_number), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0))
        cv.imshow(f'Camera{i}', frame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

for capture in captures:
    capture.release()
cv.destroyAllWindows()
