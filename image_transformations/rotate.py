import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale),int(h * scale))
    return cv.resize(frame, d, interpolation=cv.INTER_AREA)


def rotate (frame, angle, rotPoint=None):
    width = frame.shape[1]
    height = frame.shape[0]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(frame , rotMat, dimensions);

img = resize_image(cv.imread('../Images/Back-Side-Mehndi-.jpg'))
cv.imshow('Image', img)
angle=0

while True:
    rot_img = rotate(img, angle)
    cv.imshow('Image', rot_img)

    key = cv.waitKey(0) & 0xFF
    if key == ord('k'):
        angle = angle+1
        print('AC')
    elif key == ord('l'):
        angle = angle-1
        print('C')
    elif key == ord('p'):
        break
    else:
        continue
cv.waitKey(0)
cv.destroyAllWindows();