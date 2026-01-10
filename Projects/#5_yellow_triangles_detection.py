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

#1. Read Image

raw = cv.imread('../Images/shapes.png')

if raw is None:
    raise IOError('Image not found or invalid path')

img = resize_image(raw)


#2. Convert to HSV

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


#3. Define Color Ranges

lower_yellow = np.array([10, 40, 40])
upper_yellow = np.array([38, 255, 255])


#4. Generate Mask

mask = cv.inRange(hsv, lower_yellow,upper_yellow)

#5. Blurring the mask for noise reduction

mask_blur = cv.GaussianBlur(mask, (5,5), 0)

#6. Mask Refinement

kernel = np.ones((5,5), dtype='uint8')
mask_clean = cv.morphologyEx(mask_blur, cv.MORPH_OPEN, kernel)
mask_clean = cv.morphologyEx(mask_clean,  cv.MORPH_CLOSE, kernel)

#7.  Result Object Isolation

result = cv.bitwise_and(img, img,  mask=mask_clean)


#8. Detect Contours

result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

result_blur = cv.GaussianBlur(result_gray, (7,7), 0)

edges = cv.Canny(result_blur, 120,220)


edge_kernel = np.ones((5,5), dtype='uint8')
edges_closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, edge_kernel)

contours, hierarchies = cv.findContours(edges_closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

#9. Detect no of triangles

triangles_detected = img.copy()
approx_contours = []

for contour in contours:
    area = cv.contourArea(contour)
    
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 3:
        approx_contours.append(approx)

print(f'Total Triangles : {len(approx_contours)}')

#10. Remove the overlapping contours
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    union = w1*h1 + w2*h2 - intersection

    if union == 0:
        return 0

    return intersection / union

filtered_contours = []
boxes = []

for cnt in approx_contours:
    box = cv.boundingRect(cnt)
    keep = True

    for i, kept_box in enumerate(boxes):
        if iou(box, kept_box) > 0.4:
            # keep the larger contour
            if cv.contourArea(cnt) > cv.contourArea(filtered_contours[i]):
                filtered_contours[i] = cnt
                boxes[i] = box
            keep = False
            break

    if keep:
        filtered_contours.append(cnt)
        boxes.append(box)

cv.drawContours(triangles_detected,filtered_contours, -1, (0,255,0), 3)

print(f'Filtered Triangles : {len(filtered_contours)}')

cv.imshow('Image', img)
cv.imshow('HSV', hsv)
cv.imshow('Result', result)
cv.imshow('Result_gray', result_gray)
cv.imshow('Result_Blur', result_blur)
cv.imshow('Canny', edges)
cv.imshow('Edges Closed', edges_closing)
cv.imshow('Detected Triangles', triangles_detected)

cv.waitKey(0)
cv.destroyAllWindows()