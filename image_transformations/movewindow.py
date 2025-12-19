import cv2 as cv
import numpy as np

def resize_image(frame, scale):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale), int(h * scale))

    if scale > 1 :
        scale_mode = cv.INTER_CUBIC
    elif scale < 1 :
        scale_mode = cv.INTER_AREA
    return cv.resize(frame, d, interpolation=scale_mode)

def move_Window(frame_name, x, y):
    cv.moveWindow(frame_name, x, y)

img = resize_image(cv.imread('../Images/Back-Side-Mehndi-.jpg'), 0.5)
window_x = 0
window_y = 0

while True:
    cv.imshow('Image', img)
    move_Window('Image', window_x, window_y)
    key = cv.waitKey(0) & 0xFF

    if key == ord('8'):
        window_y = window_y - 10
    elif key == ord('6'):
        window_x = window_x + 10
    elif key == ord('4'):
        window_x = window_x - 10
    elif key == ord('2'):
        window_y = window_y + 10
    elif key == ord('p'):
        break
    else:
        continue
cv.destroyAllWindows()