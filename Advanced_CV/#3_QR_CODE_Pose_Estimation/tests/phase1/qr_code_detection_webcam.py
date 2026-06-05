import cv2 as cv
import numpy as np

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

def plot_qr_boundary(img, points):
    points = points.astype(int)

    top_left = points[0][0]
    top_right = points[0][1]
    bottom_right = points[0][2]
    bottom_left = points[0][3]

    for i, point in enumerate(points[0]):
        x,y = point

        #Plot Points
        cv.circle(img, (x,y), 5, (0,0,255), -1)

        #label points
        cv.putText(img, str(i), (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)


    for i in range(4):
        pt1 = tuple(points[0][i])
        pt2 = tuple(points[0][(i + 1)%4])

        cv.line(img, pt1, pt2, (0,255,0), 2)


cap = cv.VideoCapture(1)

# Importing QR Code Detector
qr_detector = cv.QRCodeDetector()

while True:
    true, frame = cap.read()
    frame = cv.flip(frame, 1)
    # Detect and Decode
    data, points, _ = qr_detector.detectAndDecode(frame)

    #Decoded Data
    print("Decoded Data : ")
    print(data)

    # If QR CODE is Found
    if points is not None :
        plot_qr_boundary(frame, points)
    else:
        print("No QR CODE Points Found") 

    cv.imshow("QR Code Detector", frame)
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()