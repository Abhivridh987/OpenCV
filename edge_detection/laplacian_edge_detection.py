import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rescale_image(frame, scale=2):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w*scale),int(h*scale))
    return cv.resize(frame,d,interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacian

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

contours, hierarchies = cv.findContours(lap, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(img.shape[:2], dtype='uint8')

cv.drawContours(blank, contours, -1, (255,255,255), 1)
cv.imshow('Contours', blank)
 
cv.waitKey(0)
cv.destroyAllWindows()

#Sobel

sobelx = cv.Sobel(gray, cv.CV_64F, 1,0)
sobely = cv.Sobel(gray, cv.CV_64F, 0,1)
sobel_x_y = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Sobel Combined', sobel_x_y)

cv.waitKey(0)
cv.destroyAllWindows()