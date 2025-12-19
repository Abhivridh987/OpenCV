import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale),int(h * scale))
    return cv.resize(frame, d, interpolation=cv.INTER_AREA)

def translate(frame, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (frame.shape[1], frame.shape[0])
    return cv.warpAffine(frame, transMat, dimensions)

x_trans = 0;
y_trans = 0;
img = resize_image(cv.imread('../Images/Back-Side-Mehndi-.jpg'))
cv.imshow('Image', img)

while True:
    trans_img = translate(img, x_trans, y_trans)
    cv.imshow('Image', trans_img)

    key = cv.waitKey(0) & 0xFF
    if key == ord('a'):
        print('left')
        x_trans = x_trans - 10
    elif key == ord('s'):
        print('down')
        y_trans = y_trans + 10
    elif key == ord('d'):
        print('right')
        x_trans = x_trans + 10
    elif key == ord('w') :
        print('up')
        y_trans = y_trans - 10
    elif key == ord('p'):
        break


cv.destroyAllWindows()

