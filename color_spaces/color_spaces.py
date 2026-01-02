import cv2 as cv
import matplotlib.pyplot as plt

def rescale_image(frame, scale=2):
    width = frame.shape[1]
    height = frame.shape[0]
    dimensions = (int(width * scale), int(height * scale))
    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

img = rescale_image(cv.imread('../Images/download (2).jfif'), 2)
cv.imshow('Image', img)

# BGR TO HSV

hsv =  cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

#BGR TO LAB

lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

rgb_bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
cv.imshow('RGB --> BGR', rgb_bgr)


plt.imshow(rgb)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
