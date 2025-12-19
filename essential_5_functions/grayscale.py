import cv2 as cv

def resize_image(frame, scale=0.75):
    width = frame.shape[1]
    height = frame.shape[0];

    dimensions = (int(width * scale), int(height * scale));
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def color_to_grayscale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


img = resize_image(cv.imread('../Images/Back-Side-Mehndi-.jpg'), 0.5)
gray_img = color_to_grayscale(img)

curr_img = img
isGray = False

while True:
    cv.imshow('Image', curr_img)

    key = cv.waitKey(0) & 0xFF

    if key == ord('g') and isGray == False:
        isGray = True
        curr_img = gray_img
    elif key == ord('g') and isGray == True:
        isGray = False
        curr_img = img
    elif key == ord('p'):
        break

cv.destroyAllWindows();

