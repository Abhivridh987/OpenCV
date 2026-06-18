import os
import cv2 as cv
import numpy as np
# import pytesseract
import tensorflow as tf
from paddleocr import PaddleOCR

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

class LicenseDetection():
    def __init__(self, model_path):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input shape expected by model
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        print(f"Model loaded! Input shape: {self.input_shape}")

    def preprocess_image(self, img):
        # Resize image to model input size (640x640)
        resized = cv.resize(img, (self.input_width, self.input_height))
        # Normalize to 0-1
        normalized = resized / 255.0
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
        return input_data

    def run_inference(self, input_data):
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        # Run inference
        self.interpreter.invoke()
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def parse_detections(self, output_data, orig_width, orig_height, conf_threshold=0.7):
        # output_data shape: [1, 5, num_detections]
        # 5 = cx, cy, w, h, confidence
        predictions = output_data[0]  # shape: [5, num_detections]
        predictions = predictions.T   # shape: [num_detections, 5]

        boxes = []
        confidences = []

        for pred in predictions:
            cx, cy, w, h, conf = pred

            if conf < conf_threshold:
                continue

            # Convert from normalized to pixel coordinates
            x1 = int((cx - w / 2) * orig_width)
            y1 = int((cy - h / 2) * orig_height)
            x2 = int((cx + w / 2) * orig_width)
            y2 = int((cy + h / 2) * orig_height)

            # Clamp to image boundaries
            

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_width, x2)
            y2 = min(orig_height, y2)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(conf))

        if len(boxes) == 0:
            return [], []

        # Apply NMS
        nms_boxes = []

        for x1, y1, x2, y2 in boxes:

            nms_boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

        indices = cv.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=confidences,
            score_threshold=conf_threshold,
            nms_threshold=0.4
        )

        final_boxes = []
        final_confidences = []

        if len(indices) > 0:
            for idx in indices.flatten():
                final_boxes.append(boxes[idx])
                final_confidences.append(confidences[idx])
                
        return final_boxes, final_confidences

    def perspective_transform(self, img, x1, y1, x2, y2):
        # Crop plate region
        plate_region = img[y1:y2, x1:x2]

        if plate_region.size == 0:
            return None

        # Get contours for perspective transform
        gray = cv.cvtColor(plate_region, cv.COLOR_BGR2GRAY)
        filtered = cv.bilateralFilter(gray, 11, 17, 17)
        edges = cv.Canny(filtered, 30, 200)

        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
        if len(contours) == 0:
            return plate_region
        
        plate_contour = None
        

        #Loop through contours and find rectangle
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(approx) == 4 and cv.contourArea(contour) > 1000:
                plate_contour = approx
                break

        # Apply perspective transform only if 4 corners found
        if plate_contour is not None:
            rect = cv.minAreaRect(plate_contour)
            box = cv.boxPoints(rect)
            box = np.intp(box)

            width = int(rect[1][0])
            height = int(rect[1][1])

            if width == 0 or height == 0:
                return plate_region

            src_pts = box.astype("float32")
            dst_pts = np.array([
                [0, height - 1],
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1]
            ], dtype="float32")

            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv.warpPerspective(plate_region, M, (width, height))
            return warped

        # If 4 corners not found return cropped region as is
        return plate_region

    def extract_text(self, plate_image):
        # # Convert to grayscale
        # plate_gray = cv.cvtColor(plate_image, cv.COLOR_BGR2GRAY)
        # # Otsu thresholding
        # _, plate_thresh = cv.threshold(plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # # pytesseract OCR
        # config = r'--oem 3 --psm 7'
        # text = pytesseract.image_to_string(plate_thresh, config=config)
        # return text.strip()


        result = ocr.predict(plate_image)
        text = ""
        if result:
            for block in result:
                if "rec_texts" in block:
                    text += " ".join(block["rec_texts"]) + " "
        return text.strip()



    def detect_license_plate(self, img):

        original = img.copy()

        orig_height, orig_width = img.shape[:2]

        input_data = self.preprocess_image(img)

        output_data = self.run_inference(input_data)

        boxes, confidences = (
            self.parse_detections(
                output_data,
                orig_width,
                orig_height
            )
        )

        if len(boxes) == 0:
            print("No license plate detected")
            return original


        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]

            plate_image = self.perspective_transform(img,x1,y1,x2,y2)
            if plate_image is None:
                continue
            text = self.extract_text(plate_image)
            print(f"License Plate {i+1}: {text}")
            cv.rectangle(original,(x1, y1),(x2, y2),(0,255,0),2)

            cv.putText(original,text,(x1, y1-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            cv.putText(original,f"{conf:.2f}",(x1, y2+15),cv.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)

        return original

def resize_function(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(scale * w), int(scale * h))

    if scale > 1:
        scale_mode = cv.INTER_CUBIC
    elif scale < 1:
        scale_mode = cv.INTER_AREA
    else:
        return frame

    return cv.resize(frame, d, interpolation=scale_mode)


# Paths
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_float32.tflite')
# Initialize detector
detector = LicenseDetection(model_path=model_path)



cap = cv.VideoCapture(0)

while True:
    true, frame = cap.read()
    if not true:
        break
    result = detector.detect_license_plate(frame)
    cv.imshow('Result', result)

    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

    
    

cap.release()
cv.destroyAllWindows()