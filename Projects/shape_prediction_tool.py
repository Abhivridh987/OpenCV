import cv2 as cv
import numpy as np


def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale),int(h * scale))
    if scale > 1:
        scale_mode = cv.INTER_CUBIC
    elif scale < 1:
        scale_mode = cv.INTER_AREA  
    return cv.resize(frame, d, interpolation=scale_mode)


img = cv.imread('../Images/octagon.png')

frame_copy = img.copy()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

canny = cv.Canny(blur, 100, 200)
# cv.imshow('Canny Edges', edges)

# ret, thresh = cv.threshold(blur, 0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
contours, hierarchies = cv.findContours(canny , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)
print(f'{len(contours)} contour(s) found')

max_area = 0
img_area = img.shape[0] * img.shape[1]
document_contour = None
for contour in contours:
    area = cv.contourArea(contour)
    if area > 0.9 * img_area:
        continue
    if area > 1000 :
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        if area > max_area:
            print(len(approx))
            document_contour  = approx
            max_area = area

if document_contour is None:
    print('No valid shape detected')
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit()

if len(document_contour) == 3:
    print('Triangle')
elif len(document_contour) == 4:
    x,y,w,h = cv.boundingRect(document_contour)
    aspect_ratio = w / float(h)
    if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
        print('Square')
    else:
        print('Rectangle')
elif len(document_contour) == 5:
    print('Pentagon')
elif len(document_contour) == 6:
    print('Hexagon')
elif len(document_contour) == 7:
    print('Heptagon')
elif len(document_contour) == 8:
    print('Octagon')
elif len(document_contour) > 8:
    print('Circle')

cv.drawContours(img, [document_contour], -1, (0,255,0), 3)
cv.imshow('Image', img)

cv.waitKey(0)
cv.destroyAllWindows()

