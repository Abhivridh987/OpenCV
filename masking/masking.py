import cv2 as cv
import numpy as np

def rescale_image(frame, scale=2):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w*scale),int(h*scale))
    return cv.resize(frame,d,interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'))

blank = np.zeros(img.shape[:-1], dtype='uint8')

x=blank.shape[1] // 2 
y=blank.shape[0] // 2 
rad=100


while True:
    circle = cv.circle(blank.copy(), (x,y), rad, 255, -1)

    masked = cv.bitwise_and(img, img, mask=circle)
    cv.imshow('Masked', masked)

    pressed = cv.waitKey(0) &  0xff
    if pressed == ord('a'):
        x = x-10
    elif pressed == ord('d'):
        x = x+10
    elif pressed == ord('w'):
        y = y-10
    elif pressed == ord('s'):
        y=y+10
    elif pressed == ord('+'):
        rad = rad+1
    elif pressed == ord('-'):
        rad = rad-1
    elif pressed == ord('p'):
        break
    else:
        continue

cv.destroyAllWindows()