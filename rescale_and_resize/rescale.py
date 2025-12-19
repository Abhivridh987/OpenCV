import cv2 as cv

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

img = cv.imread('../Images/Back-Side-Mehndi-.jpg');
cv.imshow('Output', img)

rescaled_img = rescale_frame(img, scale=0.75)
cv.imshow('Rescaled Image', rescaled_img)

cv.waitKey(0)
cv.destroyAllWindows()