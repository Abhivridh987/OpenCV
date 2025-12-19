import cv2 as cv

def rescale_frame(frame,scale=0.75):
    width = int(frame.shape[1] * scale);
    height = int(frame.shape[0] * scale);
    dimensions = (width, height);
    return cv.resize(frame,dimensions, interpolation = cv.INTER_AREA);
def reset_frame(frame, width = 200, height = 200):
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA);

cap = cv.VideoCapture(0);
while True:
    isTrue, frame = cap.read();
    frame = rescale_frame(frame,0.4);

    for i in range(0,4):
        for j in range(0,6):
            cv.imshow(f'Webcam {i}{j}', frame);
            cv.moveWindow(f'Webcam {i}{j}', frame.shape[1]*j, frame.shape[0]*i);

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
cap.release();
cv.destroyAllWindows();
