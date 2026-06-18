import os
import cv2 as cv
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class LicenseDetection():
    def __init__(self):
        pass
    
    def detect_license_plate(self, img_path):

        text = '\o'

        img = cv.imread(img_path)
        original = img.copy()

        #Gray Scale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Applying Bilateral Filter 
        filtered = cv.bilateralFilter(gray, 11,17,17)  # cv.bilateralFilter(img, neighbourhood pixels, color diff, influence)

        #Canny
        canny = cv.Canny(filtered, 30, 200)


        #Contours
        contours, ret = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #Sort Contours
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

        plate_contour = None

        #Loop through contours and find rectangle
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(approx) == 4:
                plate_contour = approx
                break

        if plate_contour is not None:
            # Get minimum area rotated rectangle
            rect = cv.minAreaRect(plate_contour)
            box = cv.boxPoints(rect)
            box = np.intp(box)

            # Draw tilted rectangle on original image
            cv.drawContours(original, [box], 0, (0, 255, 0), 3)

            # Get width, height and angle
            width = int(rect[1][0])
            height = int(rect[1][1])

            # Source and destination points for perspective transform
            src_pts = box.astype("float32")
            dst_pts = np.array([
                [0, height-1],
                [0, 0],
                [width-1, 0],
                [width-1, height-1]
            ], dtype="float32")

            # Straighten the tilted plate
            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            plate_image = cv.warpPerspective(original, M, (width, height))

            # Preprocess plate for pytesseract
            plate_gray = cv.cvtColor(plate_image, cv.COLOR_BGR2GRAY)
            plate_thresh = cv.threshold(plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


            plate_gray = cv.cvtColor(plate_image, cv.COLOR_BGR2GRAY)
            ret, plate_thresh = cv.threshold(plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            # Pass to pytesseract
            config = r'--oem 3 --psm 7'
            text = pytesseract.image_to_string(plate_thresh, config=config)
            text = text.strip()

        cv.imshow('Image', img)
        cv.imshow('Original', original)
        print(f'License Number : ', text)

        key = cv.waitKey(0) & 0xFF

        cv.destroyAllWindows()
        return


def resize_function(frame, scale=0.5):
    w = frame.shape[1]
    h  =frame.shape[0]
    d = (int(scale*w), int(scale*h))

    if scale > 1:
        scale_mode = cv.INTER_CUBIC
    elif scale < 1:
        scale_mode = cv.INTER_AREA
    else:
        return frame
    
    return cv.resize(frame, d, interpolation=scale_mode)



folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.jfif']
image_files = [f for f in os.listdir(folder_path)
               if os.path.splitext(f)[1].lower() in image_extensions]

detector = LicenseDetection()

for image in image_files:
    img_path = os.path.join(folder_path, image)
    detector.detect_license_plate(img_path)
    


