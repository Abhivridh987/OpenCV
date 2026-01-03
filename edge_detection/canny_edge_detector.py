import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rescale_image(frame, scale=2):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w*scale),int(h*scale))
    return cv.resize(frame,d,interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'))
cv.imshow('Image', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

a = 125
b = 175
while True:
    canny = cv.Canny(gray_blur, a,b)
    print(f'({a}, {b})')
    cv.imshow('Gray Blur', canny)
    pressed = cv.waitKey(0) & 0xff
    if pressed == ord('w'):
        a = a+1
    elif pressed == ord('s'):
        a = a - 1
    elif pressed == ord('a'):
        b = b-1
    elif pressed == ord('d'):
        b= b+1
    elif pressed== ord('p'):
        break
    else:
        continue
cv.destroyAllWindows()
