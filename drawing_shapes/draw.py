import cv2 as cv
import numpy as np

def resize_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale);
    dimensions = (width,height);

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA);


blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('blank', blank);

#1. Paint the image a certain color
blank[:] = 0,255,0  # Paint the whole image green
cv.imshow('Green', blank);

#2. Paint certain regions
blank[:] = 0,0,0;
blank[200:300, 300:400] = 0,0,255
blank = resize_frame(blank,0.5);
cv.imshow('Red Box', blank)

#3. Paint a rectangle
blank[:] = 0,0,0;
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
cv.imshow('Rectangle', blank);

cv.rectangle(blank, (0,0),(100,100), (0,0,255), thickness=-1);
cv.imshow('Rectangle 2', blank);

#4. Paint a circle
blank[:] = 0,0,0;
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 50, (255,255,0), thickness=-1);
cv.imshow('Circle', blank);

#5. Paint a line
cv.line(blank, (0,0), (250,250), (255,255,0), thickness=5 )
cv.imshow('Line', blank);

#6. Put text
cv.putText(blank, 'Hello', (40,200), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness=2);
cv.imshow('Text ', blank);

cv.waitKey(3000);
cv.destroyAllWindows();