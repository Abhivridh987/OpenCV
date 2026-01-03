import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def rescale_image(frame, scale=2):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w*scale),int(h*scale))
    return cv.resize(frame,d,interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'))

blank = np.zeros(img.shape[:2], dtype='uint8')

x=img.shape[1] // 2
y= img.shape[0] // 2
rad = 100

circle = cv.circle(blank.copy(), (x + 150,y - 70), rad, 255, -1)

mask = cv.bitwise_and(img, img, mask=circle)
cv.imshow('Masked Image', mask)

plt.figure()
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.xlim([0,256])

colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], circle, [256], [0,256])
    plt.plot(hist, color=col)
    
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()