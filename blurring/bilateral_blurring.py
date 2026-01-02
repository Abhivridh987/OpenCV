import cv2 as cv
import numpy as np

def rescale_image(frame, scale=2):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w*scale),int(h*scale))
    return cv.resize(frame,d,interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'))
cv.imshow('Image', img)
cv.moveWindow('Image', 400,20)

a1 = 5
a2 = 75
a3 = 75



while True:
    print(f'({a1}, {a2}, {a3}) \n(Diameter : {a1})  SigmaColor : {a2} SigmaSpace : {a3}')
    bilateral_blur = cv.bilateralFilter(img, a1, a2, a3)
    cv.imshow('Bilateral Blur', bilateral_blur)
    cv.moveWindow('Bilateral Blur', 400, 400)
    pressed = cv.waitKey(0) & 0xff
    if pressed == ord('w'):
        a1 = a1+1
    elif pressed == ord('s'):
        if a1 > 1 : a1 = a1-1
    elif pressed == ord('e'):
        a2 = a2+1
    elif pressed == ord('d'):
        a2 = a2-1
    elif pressed == ord('r'):
        a3 = a3+1
    elif pressed == ord('f'):
        a3 = a3-1
    elif pressed == ord('p'):
        break
    else :
        continue        

cv.destroyAllWindows()
