import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')

width = blank.shape[1]
height = blank.shape[0]

width_bar = blank.shape[1] // 8
height_bar = blank.shape[0] // 8

center = (width//2, height//2);
square_up_pivot = (width//2 - 100, height//2 - 100);
square_down_pivot = (width//2 + 100, height//2 + 100);

white = (255,255,255)
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)

for i in range(1,8,1):
    blank[width_bar*i, :] = 255,255,255
    blank[:, height_bar*i] = 255,255,255


cv.imshow('CrossLines', blank);
cv.waitKey(3000)
cv.destroyAllWindows();

blank[:] = 0,0,0;
cv.circle(blank, center, 100, white, thickness=-1)
cv.rectangle(blank, square_up_pivot, square_down_pivot, white, thickness=1);
cv.imshow('Circle', blank);
cv.waitKey(0);
cv.destroyAllWindows();