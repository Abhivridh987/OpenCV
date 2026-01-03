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

ret, thresh = cv.threshold(gray_blur, 125, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(img.shape[:2], dtype='uint8')

cv.drawContours(blank, contours, -1, (255,255,255), 1)
cv.imshow('Contours', blank)
cv.waitKey(0)
cv.destroyAllWindows()