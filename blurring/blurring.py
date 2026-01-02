import cv2 as cv
import numpy as np

def rescale_image(frame, scale=2):
    width = frame.shape[1]
    height = frame.shape[0]
    dimensions = (int(width * scale), int(height * scale))
    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'), 2)
cv.imshow('Image', img)



# Normal Blur / Averaging

blur = cv.blur(img, (5,5))
cv.imshow('Blur', blur)


#Gaussian Blur

gauss_blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Gaussian Blur', gauss_blur)

#Median Blur
med_blur = cv.medianBlur(img, 5)
cv.imshow('Median Blur', med_blur)

#Bilateral Blur
bilateral_blur = cv.bilateralFilter(img, 9,75,75)
cv.imshow('Bilateral Blur', bilateral_blur)

cv.waitKey(0)
cv.destroyAllWindows()


