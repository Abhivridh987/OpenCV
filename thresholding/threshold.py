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

# Simple Thresholding

ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY_INV)
cv.imshow('Thresh Inverse', thresh)

# Adaptive Thresholding

adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3 )
cv.imshow('Adaptive Thresh - Mean', adaptive_thresh)

adaptive_thresh_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11,3)
cv.imshow('Adaptive Thresh - Mean Inverse', adaptive_thresh_inv)

adaptive_thresh_gauss= cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3 )
cv.imshow('Adaptive Thresh - Gaussian', adaptive_thresh_gauss)

adaptive_thresh_gauss_inv = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11,3)
cv.imshow('Adaptive Thresh - Gaussian Inverse', adaptive_thresh_gauss_inv)


cv.waitKey(0)
cv.destroyAllWindows()