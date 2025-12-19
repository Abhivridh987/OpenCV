import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale),int(h * scale))
    if scale > 1:
        scale_mode = cv.INTER_CUBIC
    elif scale < 1:
        scale_mode = cv.INTER_AREA  
    return cv.resize(frame, d, interpolation=scale_mode)


img = cv.imread('../Images/Back-Side-Mehndi-.jpg')
cv.imshow('Original Image', img)

resized_img = resize_image(img, 0.5)
cv.imshow('Resized Image', resized_img)

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blurred Image', blur)

# edges = cv.Canny(resized_img, 100, 200)
# cv.imshow('Canny Edges', edges)

ret, thresh = cv.threshold(gray, 125,255,cv.THRESH_BINARY)
cv.imshow('Thresholded Image', thresh)

cv.waitKey(0)
cv.destroyAllWindows()


