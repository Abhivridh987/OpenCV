import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def rescale_image(frame, scale=2):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w*scale),int(h*scale))
    return cv.resize(frame,d,interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
blank = np.zeros(img.shape[:2], dtype='uint8')

x=img.shape[1] // 2
y= img.shape[0] // 2
rad = 100

while True:
    circle = cv.circle(blank.copy(), (x,y), rad, 255, -1)

    mask = cv.bitwise_and(gray, gray, mask=circle)
    cv.imshow('Masked Image', mask)

    gray_histogram = cv.calcHist([gray], [0], circle, [256], [0,256])   # img, index of channel , mask, no_of_bins, range of pixels
    gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])   # img, index of channel , mask, no_of_bins, range of pixels

    plt.figure()
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_histogram, color='b')
    plt.plot(gray_hist, color='g')
    plt.xlim([0,256])
    plt.show()

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
