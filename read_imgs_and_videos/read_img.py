import cv2 as cv
    
img = cv.imread('../Images/Back-Side-Mehndi-.jpg')

cv.imshow('Output', img)

cv.waitKey(0)