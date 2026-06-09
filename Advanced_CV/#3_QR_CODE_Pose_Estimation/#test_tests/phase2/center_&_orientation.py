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

def plot_points_boundary(img, points):
    points = points.astype(int)
    print(points)

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

img = cv.imread('../../images/qr_1.jpg')

if img is None:
    print("Error Occured while loading image")
    exit()

img2 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
img3 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)


# Importing QR Code Detector
qr_detector = cv.QRCodeDetector()

# Detect and Decode
data, points, _ = qr_detector.detectAndDecode(img)
data2, points2, _ = qr_detector.detectAndDecode(img2)
data3, points3, _ = qr_detector.detectAndDecode(img3) 
#Decoded Data
print("Decoded Data 1: ")
print(data)

print("Decoded Data 2: ")
print(data2)

print("Decoded Data 3: ")
print(data3)


# If QR CODE is Found
if points is not None :
    plot_points_boundary(img, points)
    plot_center(img, points)
    orientation2D(img, points)
else:
   print("No QR CODE Points Found") 

if points2 is not None :
    plot_points_boundary(img2, points2)
    plot_center(img2, points2)
    orientation2D(img2, points2)
else:
   print("No QR CODE Points Found") 

if points3 is not None :
    plot_points_boundary(img3, points3)
    plot_center(img3, points3)
    orientation2D(img3, points3)
else:
   print("No QR CODE Points Found")   

cv.imshow("QR Code 1", img)
cv.imshow("QR Code 2", img2)
cv.imshow("QR Code 3", img3)

key = cv.waitKey(0)
cv.destroyAllWindows()
















