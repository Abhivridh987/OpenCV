import cv2 as cv
import numpy as np

def rescale_image(frame, scale=2):
    width = frame.shape[1]
    height = frame.shape[0]
    dimensions = (int(width * scale), int(height * scale))
    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'), 2)
cv.imshow('Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b, g, r = cv.split(img)

cv.imshow('Gray Blue', b)
cv.imshow('Gray Green', g)
cv.imshow('Gray Red', r)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank ,g, blank])
red = cv.merge([blank, blank, r])

green_red = cv.merge([blank, g, r])


cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)
cv.imshow('Green Red', green_red)


cv.waitKey(0)
cv.destroyAllWindows()