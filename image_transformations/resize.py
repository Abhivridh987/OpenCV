import cv2 as cv
import numpy as np

def resize_image(frame, scale):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale), int(h * scale))

    if scale > 1:
        scale_mode = cv.INTER_AREA
    else:
        scale_mode = cv.INTER_CUBIC
    return cv.resize(frame, d, interpolation=scale_mode);

img = cv.imread('../Images/Back-Side-Mehndi-.jpg')
cv.imshow('Image', img)
scale = 1

while True:
    resized_img = resize_image(img, scale);
    cv.imshow('Image', resized_img)

    key = cv.waitKey(0) & 0xFF
    if key == ord('+'):
        scale = scale+0.01
    elif key == ord('-'):
        scale = scale-0.01
    elif key == ord('p'):
        break
    else :
        continue

cv.destroyAllWindows();