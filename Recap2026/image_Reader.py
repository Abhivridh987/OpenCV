import cv2 as cv
import numpy as np

def resize_image(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(scale*w), int(scale*h))

    if scale > 1:
        return cv.resize(frame, d, interpolation=cv.INTER_CUBIC)
    elif scale < 1:
        return cv.resize(frame, d, interpolation=cv.INTER_AREA)
    else:
        return frame

def move_Window(frame_name, x, y):
    cv.moveWindow(frame_name, x, y)

cap = cv.VideoCapture(0)
true, frame = cap.read()

frame_scale = 0.5


init_x, init_y = 100, 100
blur_amount = 5

canny_upper_threshold = 170
canny_lower_threshold = 125

instructions = [
    "Press 'q' to Quit, 'o' to Save Image",
    "Press 'w', 'a', 's', 'd' to Move Window",
    "Press 'b', 'v' to Adjust Blur",
    "Press 'u', 'j' to Adjust Canny Upper Threshold",
    "Press 'o', 'l' to Adjust Canny Lower Threshold",
    "Press 'f', 'g' to Adjust Frame Scale"
]
instruction_page = np.zeros(frame.shape[:2], dtype='uint8')
for i, line in enumerate(instructions):
    cv.putText(instruction_page, line, (10, 30 + i*30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv.imshow('INSTRUCTIONS', instruction_page)

while True:
    true, frame = cap.read()
    resized_frame = resize_image(frame, frame_scale)
    cv.imshow("Resized Frame", resized_frame)
    move_Window("Resized Frame", init_x, init_y)

    # GrayScale Image
    gray = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray Scale", gray)
    #Blur Image
    blur = cv.GaussianBlur(gray, (blur_amount, blur_amount), cv.BORDER_DEFAULT)

    #Canny Edge Detection
    canny = cv.Canny(blur, canny_lower_threshold, canny_upper_threshold)
    cv.imshow("Canny Edges", canny)

    key = cv.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('o'):
        cv.imwrite('../Images/resized_webcam.jpg', resized_frame)
    elif key == ord('w'):
        init_y = init_y - 10
    elif key == ord('s'):
        init_y = init_y + 10
    elif key == ord('a'):
        init_x = init_x - 10
    elif key == ord('d'):
        init_x = init_x + 10
    elif key == ord('b'):
        blur_amount = blur_amount + 2
    elif key == ord('v'):
        if blur_amount > 0:
            blur_amount = blur_amount - 2
    elif key == ord('u'):
        canny_upper_threshold = canny_upper_threshold + 1
    elif key == ord('j'):
        if canny_upper_threshold > 0:
            canny_upper_threshold = canny_upper_threshold - 1
    elif key == ord('o'):
        canny_lower_threshold = canny_lower_threshold + 1
    elif key == ord('l'):
        if canny_lower_threshold > 0:
            canny_lower_threshold = canny_lower_threshold - 1
    elif key == ord('f'):
        frame_scale = frame_scale + 0.1
    elif key == ord('g'):
        if frame_scale > 0.1:
            frame_scale = frame_scale - 0.1
    print(f'Current Window Position: ({cv.getWindowImageRect("Resized Frame")[0]}, {cv.getWindowImageRect("Resized Frame")[1]})')
    print(f'Current Blur Amount: {blur_amount}')
    print(f'Current Canny : ( {canny_lower_threshold}, {canny_upper_threshold} )')
    print(f'Frame Scale: {frame_scale}')

cap.release()
cv.destroyAllWindows()