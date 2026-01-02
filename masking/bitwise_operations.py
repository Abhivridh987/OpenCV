import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (50,50), (350,350), 255,-1)
circle = cv.circle(blank.copy(), (200,200), 190, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

cv.imshow('Rectangle & Circle', cv.bitwise_and(rectangle, circle))
cv.imshow('Rectangle | Circle', cv.bitwise_or(rectangle, circle))
cv.imshow('Rectangle ^ Circle', cv.bitwise_xor(rectangle, circle))
cv.imshow('! Rectangle ', cv.bitwise_not(rectangle))
cv.imshow('! Circle', cv.bitwise_not(circle))

cv.waitKey(0)
cv.destroyAllWindows()

blank = np.zeros((400,400,3), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (50,50), (350,350), (0,0,255),-1)
circle = cv.circle(blank.copy(), (200,200), 190, (0,255,0), -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)


cv.imshow('Rectangle & Circle', cv.bitwise_and(rectangle, circle))
cv.imshow('Rectangle | Circle', cv.bitwise_or(rectangle, circle))
cv.imshow('Rectangle ^ Circle', cv.bitwise_xor(rectangle, circle))
cv.imshow('! Rectangle ', cv.bitwise_not(rectangle))
cv.imshow('! Circle', cv.bitwise_not(circle))

cv.waitKey(0)
cv.destroyAllWindows()
