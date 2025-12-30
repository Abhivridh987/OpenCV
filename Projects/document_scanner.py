import cv2 as cv
import numpy as np
import os
# from imutils.perspective import four_point_transform


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl- bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32"
    )

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def ocr(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 128,255, cv.THRESH_BINARY)
    return thresh


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

frame_copy = resized_img.copy()

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

warped = four_point_transform(frame_copy, document_contour.reshape(4,2))
cv.imshow('Warped', warped)

scanned_image = ocr(warped)
cv.imshow('OCR', scanned_image)

pressed_key = cv.waitKey(0) & 0xff
count = 0

if pressed_key == ord('s'):
    os.makedirs("Scans", exist_ok=True)
    name = input("Enter the name of file : ")
    cv.imwrite(f'Scans/Scanned_Image_({name}).jpg', scanned_image)


cv.destroyAllWindows()


