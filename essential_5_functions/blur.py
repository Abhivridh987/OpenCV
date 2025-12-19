import cv2 as cv

img = cv.imread('../Images/download.jfif');
cv.imshow('Image', img)
while True:
    cv.destroyAllWindows();
    a = int(input('Enter a number : '))
    b = int(input('Enter next number : '))
    blur_kernel = (a,b)
    blur = cv.GaussianBlur(img, blur_kernel, cv.BORDER_DEFAULT);
    cv.imshow('Image', img)
    cv.imshow('Blurred Image', blur)
    cv.waitKey(0);

