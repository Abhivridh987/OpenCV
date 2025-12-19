import cv2 as cv
import numpy as np

img = cv.imread('../Images/Back-Side-Mehndi-.jpg')
img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv.INTER_AREA)

blank = np.zeros(img.shape, dtype='uint8')

cv.imshow('Image',img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found');

cv.drawContours(blank, contours, -1, (255,255,255), 1)
cv.imshow('Contoured Image', blank)
cv.waitKey(0)
cv.destroyAllWindows()