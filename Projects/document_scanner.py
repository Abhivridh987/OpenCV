import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale), int(h * scale))
    return cv.resize(frame, d, interpolation=cv.INTER_AREA)

def order_points(pts):
    pts = pts.reshape((4,2))
    rect = np.zeros((4,2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


doc = resize_image(cv.imread('../Images/notes.webp'))
cv.imshow('Original Document', doc)

blank = np.zeros(doc.shape, dtype='uint8')
blank[:] = 255,255,255

gray = cv.cvtColor(doc, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Doc', gray)

blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blurred Doc', blur)

canny = cv.Canny(blur, 50,150)  # Generally used for documnet scanning
cv.imshow('Canny Doc', canny)

# canny = cv.Canny(blur, 175, 200)
# canny = cv.dilate(canny, np.ones((5,5), np.uint8), iterations=1)
# cv.imshow('Canny Doc', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
# cv.RETR_EXTERNAL - generally used for 1. Document boundary detection  2. Biggest Object Detection   3. Object Detection 4. Shape Detection
# cv.RETR_LIST - generally used for all contours detection (here detects evrything like text , shades, document boundary, shadows, folds, etc)

print(f'{len(contours)} contour(s) found')

contours = sorted(contours, key=cv.contourArea, reverse=True)

for cnt in contours:
    peri = cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt, 0.02*peri, True)

    if(len(approx) == 4):
        doc_contours = approx
        break

    if len(approx) != 4:
        rect = cv.minAreaRect(cnt)
        doc_contours = np.int32(cv.boxPoints(rect))

print(doc_contours)
debug = doc.copy()
# draw contour
cv.drawContours(debug, [doc_contours], -1, (0, 255, 0), 2)

# draw red points on each corner
for p in doc_contours:
    x, y = p[0]
    cv.circle(debug, (x, y), 6, (0, 0, 255), -1)

cv.imshow("Document Contour with Points", debug)


ordered = order_points(doc_contours)

for (x, y) in ordered:
    print(x, y)
    cv.circle(debug, (int(x), int(y)), 3, (255,0,0), -1)

cv.imshow("Detected Document Contour", debug)





src_pts = order_points(doc_contours)

tl, tr, br ,bl = src_pts
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0])**2 + (tr[1] - br[1])**2))
heightB = np.linalg.norm(tl-bl)
maxHeight = max(int(heightA), int(heightB))

dst_pts = np.array([
    [0,0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype='float32'
)
print(src_pts)
print(dst_pts)
M = cv.getPerspectiveTransform(src_pts, dst_pts)
warped = cv.warpPerspective(gray, M, (maxWidth, maxHeight));
warped = cv.normalize(warped, None, 0, 255, cv.NORM_MINMAX)

cv.imshow('Warped Document', warped)

scanned = cv.adaptiveThreshold(
    warped,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    25,
    15
)

cv.imshow('Scanned Document', scanned)

cv.waitKey(0)
cv.destroyAllWindows()