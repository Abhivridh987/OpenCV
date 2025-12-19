import cv2 as cv

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(w * scale),int(h * scale))
    return cv.resize(frame, d, interpolation=cv.INTER_AREA)

img = resize_image(cv.imread('../Images/Back-Side-Mehndi-.jpg'), 0.5)

while True: 
    a = int(input('Enter a number : '))
    b = int(input('Enter next number : '))

    canny = cv.Canny(img, a,b);
    blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
    canny_improved = cv.Canny(blur, a,b)
    cv.imshow('Image', img)
    cv.imshow('Canny', canny)
    cv.imshow('Improved Canny', canny_improved)
    cv.waitKey(0)
    cv.destroyAllWindows()
