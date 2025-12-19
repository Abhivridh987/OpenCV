import cv2 as cv

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale),int(h * scale))
    return cv.resize(frame, d, interpolation=cv.INTER_AREA)  #cv.INTER_LINEAR, cv.INTER_CUBIC for enlarging images maintaining quality


img = resize_image(cv.imread('../Images/Back-Side-Mehndi-.jpg'),0.5)
cv.imshow('Image',img);
canny = cv.Canny(img, 125,175)
cv.imshow('Canny', canny)

blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny_blur = cv.Canny(blur, 125,175)
cv.imshow('Canny Refined', canny_blur)

dilated = cv.dilate(canny_blur, (3,3), iterations=1)
cv.imshow('Dilated', dilated)

eroded = cv.erode(canny_blur, (3,3), iterations=3)
cv.imshow('Eroded', eroded)

cropped = img[50:200, 100:300]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
cv.destroyAllWindows()