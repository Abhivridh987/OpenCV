import cv2 as cv

cap_lapCam = cv.VideoCapture(0)
cap_webCam = cv.VideoCapture(1)

while True:
    true_lapCam, lapCam_frame = cap_lapCam.read()
    true_webCam, webCam_frame= cap_webCam.read()


    cv.imshow("Lap Camera", cv.flip(lapCam_frame, 1))
    cv.imshow("Web Cam", cv.flip(webCam_frame, 1))
    
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cap_lapCam.release()
cap_webCam.release()
cv.destroyAllWindows()