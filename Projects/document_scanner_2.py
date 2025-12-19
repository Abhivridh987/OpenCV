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


img = cv.imread('../Images/photo_6194762285033655125_y.jpg')
cv.imshow('Original Image', img)

resized_img = resize_image(img, 0.5)
cv.imshow('Resized Image', resized_img)

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blurred Image', blur)

# edges = cv.Canny(resized_img, 100, 200)
# cv.imshow('Canny Edges', edges)

ret, thresh = cv.threshold(blur, 0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('Thresholded Image', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)
print(f'{len(contours)} contour(s) found')

max_area = 0;
for contour in contours:
    area = cv.contourArea(contour)
    if area > 1000:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.015 * peri, True)
        if area > max_area and len(approx) == 4:
            document_contour  = approx
            max_area = area

cv.drawContours(resized_img, [document_contour], -1, (0,255,0), 3)
cv.imshow('Biggest Contour', resized_img)

cv.waitKey(0)
cv.destroyAllWindows()


