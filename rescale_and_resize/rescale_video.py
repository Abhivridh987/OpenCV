import cv2 as cv

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    print(dimensions)
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

cap = cv.VideoCapture(0);

while True:
    isTrue, frame = cap.read()
    no_of_windows = 10
    for i in range(no_of_windows, 0,-1):
        rescaled_frame = rescale_frame(frame, scale=(1/no_of_windows) * i)
        cv.imshow(f'Rescaled Webcam {i}', rescaled_frame)  
        cv.moveWindow(f'Rescaled Webcam {i}', 100 + int(((frame.shape[1]/no_of_windows) * (no_of_windows - i))/2),100+ int(((frame.shape[0]/no_of_windows) * (no_of_windows - i))/2));
     
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
cap.release()
cv.destroyAllWindows()
