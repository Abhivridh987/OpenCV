import cv2 as cv
import numpy as np

img = cv.imread('../Images/Back-Side-Mehndi-.jpg')


img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv.INTER_AREA)
cv.imshow('Image',img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

gray_blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Gray Blur', gray_blur)

canny = cv.Canny(gray_blur, 125,125)
cv.imshow('Canny', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found')

cv.waitKey(0)
cv.destroyAllWindows()