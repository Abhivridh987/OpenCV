import cv2 as cv
import numpy as np


# def resize_image(frame, scale=0.5):
#     w = frame.shape[1]
#     h = frame.shape[0]
#     d = (int(w*scale), int(h*scale))
#     return cv.resize(frame, d, interpolation=cv.INTER_AREA)

# def order_points(pts):
#     rect = np.zeros((4,2), dtype='float32')
    
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]

#     return rect

# def get_center(pts):

#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect

#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     max_width = max(widthA, widthB)

#     if widthA > widthB :
#         (a,b) = tl
#         center_x = b + max_width//2
#     else:
#         (a,b) = bl
#         center_x = b + max_width//2

    
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     max_height = max(heightA, heightB)

#     if heightA > heightB :
#         (a,b) = tl
#         center_y = a + max_height//2
#     else:
#         (a,b) = tr
#         center_y = a + max_height//2
    
#     center = (center_x, center_y)
#     rad = (max_height // 2 + max_width // 2)//2
#     return (center, rad)

# ---------- Utility ----------
def resize_image(frame, scale=0.5):
    h, w = frame.shape[:2]
    return cv.resize(frame, (int(w * scale), int(h * scale)),
                     interpolation=cv.INTER_AREA)

# ---------- Center & Radius (robust for any shape) ----------
def get_center_and_radius(cnt):
    M = cv.moments(cnt)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    (_, _), radius = cv.minEnclosingCircle(cnt)
    return (cx, cy), int(radius)

img = resize_image(cv.imread('../Images/download.jfif'), scale=2)



cv.imshow('Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5), 0)
a = 38
b=277
canva = img.copy()
canny = cv.Canny(blur, a,b)
cv.imshow('Canny ', canny)
contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} contour(s) found')

h, w = img.shape[:2]
img_area = h*w

largest_contour = None
max_area = 0

for contour in contours:
    area = cv.contourArea(contour)
    if 1000 < area < 0.95*img_area:
        if area > max_area:
            max_area = area
            largest_contour = contour

if largest_contour is None:
    print("No valid object detected")
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit()

cv.drawContours(canva, [largest_contour], -1, (0,255,0), 1)
cv.imshow('Biggest Contour', canva)


(center, rad) = get_center_and_radius(largest_contour)
print(center)
print(rad)
center = (int(center[0]), int(center[1]))
rad = int(rad)
cv.circle(canva, center, rad, (0,255,0), thickness=2)
cv.circle(canva, center, 1, (0,255,0) ,thickness=5)

cv.imshow('Biggest Object', canva)



cv.waitKey(0)
cv.destroyAllWindows()