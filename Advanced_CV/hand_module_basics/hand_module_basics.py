import cv2 as cv
import mediapipe as mp
import time

def resize_function(frame, scale=0.5):
    w = frame.shape[1]
    h = frame.shape[0]
    d = (int(scale*w), int(scale*h))

    if scale > 1:
        scale_mode = cv.INTER_CUBIC
    elif scale < 1:
        scale_mode = cv.INTER_AREA
    else:
        return frame
    return cv.resize(frame, d, interpolation=scale_mode)


mpHands = mp.solutions.hands
hands = mpHands.Hands()     # mp.Hands(False, no_of_hands=2, detection_confidence=0.5, tracking_confidence=0.5  )
mpDraw = mp.solutions.drawing_utils

pTime = 0
thumb = None
index = None
wrist = None

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

max_length = 0

x_position = 60
y_position = 15

while True:
    true, frame = cap.read()
    resized_frame = resize_function(frame, 1)
    resized_frame = cv.flip(resized_frame, 1)
    imgRGB = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cv.putText(resized_frame, str(id), (int(lm.x*resized_frame.shape[1]), int(lm.y*resized_frame.shape[0])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                if id == 0:
                    wrist = (int(lm.x*resized_frame.shape[1]), int(lm.y*resized_frame.shape[0]))
                    cv.circle(resized_frame, wrist, 17, (255,0,255), cv.FILLED)
                if id == 4:
                    thumb = (int(lm.x*resized_frame.shape[1]), int(lm.y*resized_frame.shape[0]))
                    cv.circle(resized_frame, thumb, 17, (255,0,255), cv.FILLED)
                if id == 8:
                    index = (int(lm.x*resized_frame.shape[1]), int(lm.y*resized_frame.shape[0]))
                    cv.circle(resized_frame, index, 17, (255,0,255), cv.FILLED)
            mpDraw.draw_landmarks(resized_frame,
                                   handLms, 
                                   mpHands.HAND_CONNECTIONS,
                                   mpDraw.DrawingSpec(color=(0,0,255), thickness = 2, circle_radius = 2),
                                   mpDraw.DrawingSpec(color=(0,255,0), thickness = 2)
                                )
            
         

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(resized_frame, f'FPS : {round(fps, 2)}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)

    if thumb is None or index is None:
        frame_scale = 1
    else:
        length = ((index[0] - thumb[0])**2 + (index[1] - thumb[1])**2)**0.5
        if length > max_length:
            max_length = length
        frame_scale = length / max_length
    
    resized_frame = resize_function(resized_frame, frame_scale)
    
    if wrist is not None:
        x_position = 60 + int((806 * (wrist[0] / resized_frame.shape[1])))
        y_position = 15 + int((315 * (wrist[1] / resized_frame.shape[0])))
    else:
        x_position = 60
        y_position = 15

    
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    if key == ord('w'):
        y_position -= 1
    if key == ord('s'):
        y_position += 1
    if key == ord('a'):
        x_position -= 1
    if key == ord('d'):
        x_position += 1
    cv.imshow("Frame",resized_frame)
    cv.moveWindow("Frame", x_position, y_position)

cap.release()
cv.destroyAllWindows()