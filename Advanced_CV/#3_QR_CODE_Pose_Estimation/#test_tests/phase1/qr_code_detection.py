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
else:
   print("No QR CODE Points Found") 

if points2 is not None :
    plot_points_boundary(img2, points2)
else:
   print("No QR CODE Points Found") 

if points3 is not None :
    plot_points_boundary(img3, points3)
else:
   print("No QR CODE Points Found")   

cv.imshow("QR Code 1", img)
cv.imshow("QR Code 2", img2)
cv.imshow("QR Code 3", img3)

key = cv.waitKey(0)
cv.destroyAllWindows()
















