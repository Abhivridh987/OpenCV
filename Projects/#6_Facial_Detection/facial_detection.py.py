import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    width = int(frame.shape[0] * scale)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)
    if scale < 1:
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    elif scale > 1:
        return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)
    else:
        return frame


cap = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    ret, frame = cap.read()

    frame_copy = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(frame_copy, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

    cv.imshow('Facial Recognition', frame)
    key = cv.waitKey(20)
    if key == ord('q'):
        break

cv.destroyAllWindows()

