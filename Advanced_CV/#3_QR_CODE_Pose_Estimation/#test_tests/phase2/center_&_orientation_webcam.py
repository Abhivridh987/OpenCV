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

def plot_center(img, points, returns=False):
    center_x = int(np.mean(points[0][:,0]))
    center_y = int(np.mean(points[0][:,1]))
    center = tuple([center_x, center_y])

    if returns is True:
       return center
    
    cv.circle(img, center, 4, (255,0,0), -1)

def orientation2D(img, points, returns=False):
    center_x, center_y = plot_center(img, points, True)
    x0, y0 = points[0][0]
    x1, y1 = points[0][1]

    dx = x1 - x0
    dy = y1 - y0

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    if returns is True:
        return angle_deg
    
    cv.putText(img, f'2D Angle : {round(float(angle_deg), 2)}', (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    arrow_length = 100
    end_x = int(center_x + arrow_length * np.cos(angle_rad))
    end_y = int(center_y + arrow_length * np.sin(angle_rad))
    
    
    
    cv.arrowedLine(img, (center_x, center_y), (end_x, end_y), (255,0,255), 3)


cap = cv.VideoCapture(0)

# Importing QR Code Detector
qr_detector = cv.QRCodeDetector()

while True:
    true, frame = cap.read()
    # frame = cv.flip(frame, 1)
    # Detect and Decode
    data, points, _ = qr_detector.detectAndDecode(frame)

    #Decoded Data
    print("Decoded Data : ")
    print(data)

    # If QR CODE is Found
    if points is not None :
        plot_qr_boundary(frame, points)
        plot_center(frame, points)
        orientation2D(frame, points)
    else:
        print("No QR CODE Points Found") 

    cv.imshow("QR Code Detector", frame)
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()