import cv2 as cv
import mediapipe as mp
import time

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

mpHands = mp.solutions.hands
hands = mpHands.Hands()     # mp.Hands(False, no_of_hands=2, detection_confidence=0.5, tracking_confidence=0.5  )
mpDraw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

finger_points = {
    "thumb":{},
    "index":{},
    "middle":{},
    "ring":{},
    "pinky":{}
}

while True:
    true, frame = cap.read()
    resized_frame = resize_function(frame, 1)

    resized_frame = cv.flip(resized_frame, 1)
    imgRGB = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
    
            # for marking the 21 points on the hand
            mpDraw.draw_landmarks(
                resized_frame,
                handLms, 
                mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0,0,255), thickness = 2, circle_radius = 2),
                mpDraw.DrawingSpec(color=(0,255,0), thickness = 2)
            )

            #Labelling the points on the hand
            for id, lm in enumerate(handLms.landmark):

                
                x= int(lm.x*resized_frame.shape[1])
                y= int(lm.y*resized_frame.shape[0])

                if id in [1,2,3,4]:
                    finger_points["thumb"][id] = (x,y)
                elif id in [5,6,7,8]:
                    finger_points["index"][id] = (x,y)
                elif id in [9,10,11,12]:
                    finger_points["middle"][id] = (x,y)
                elif id in [13,14,15,16]:
                    finger_points["ring"][id] = (x,y)
                elif id in [17,18,19,20]:
                    finger_points["pinky"][id] = (x,y)

                cv.putText(
                    resized_frame, 
                    str(id), 
                    (x, y),
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0,0,255),
                    2
                )
            
    cv.imshow("Frame", resized_frame)

    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
