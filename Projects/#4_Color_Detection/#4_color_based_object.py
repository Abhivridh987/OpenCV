import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(scale*w), int(scale*h))
    return cv.resize(frame, d,interpolation=cv.INTER_AREA)

# 1. Read Image
img = resize_image(cv.imread('../Images/book.jpg'), scale=0.9)

# 2. Convert BGR to HSV

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


#3. Set Bounds

lower = np.array([90, 50, 50])  # Light Blue
upper = np.array([140,255,255]) # Dark Blue

#4. Mask Generation

mask = cv.inRange(hsv, lower, upper)   # cv.inRange, cv.threshold, cv.adaptiveThreshold are used for mask generation

#5 . Blurring the mask for noise reduction

mask_blur = cv.GaussianBlur(mask, (5,5), 0)

#6. Mask Refinement

kernel = np.ones((5,5), np.uint8)
mask_clean = cv.morphologyEx(mask_blur, cv.MORPH_OPEN, kernel)
mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel)

#7. Result Object Isolation
result = cv.bitwise_and(img, img, mask=mask_clean)

#8. Mask Edge Detection

edges = cv.Canny(mask_clean, 50,100)
contours, hierarchies = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

max_area = 0
for contour in contours:
    area = cv.contourArea(contour)
    if area < 500:
        continue
    if area > max_area:
        x,y,w,h = cv.boundingRect(contour)
        detected = cv.rectangle(img.copy(), (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow('Image', img)
cv.imshow('HSV', hsv)
cv.imshow('Mask', mask)
cv.imshow('Mask Blurred', mask_blur)
cv.imshow('Mask Cleaned', mask_clean)
cv.imshow('Result', result)
cv.imshow('Edges', edges)
cv.imshow('Detected', detected)

cv.waitKey(0)
cv.destroyAllWindows()
