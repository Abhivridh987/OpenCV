import cv2 as cv
import numpy as np
import mediapipe as mp



#Utility Funtions

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

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def closure_percentage(angle, max_angle = 160, min_angle =60):
    if angle > max_angle:
        return 0
    elif angle < min_angle:
        return 100
    else:
        return int((max_angle - angle) / (max_angle - min_angle) * 100)

mpHands = mp.solutions.hands
hands = mpHands.Hands()     # mp.Hands(False, no_of_hands=2, detection_confidence=0.5, tracking_confidence=0.5  )
mpDraw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
true, frame = cap.read()



previous_dots = []

while True:
    finger_points = {
    "thumb":{},
    "index":{},
    "middle":{},
    "ring":{},
    "pinky":{}
}
    true, frame = cap.read()
    inference_screen = np.zeros(frame.shape[:-1], dtype='uint8')
    resized_frame = resize_function(frame, 1)
    resized_frame = cv.flip(resized_frame, 1)

    for i, dot in enumerate(previous_dots):
        cv.circle(resized_frame, dot, 5, (255,255,0), -1)

        if i > 0:
            cv.line(resized_frame, previous_dots[i-1], dot, (255,255,0), 4) #cv.line(image, pt1, pt2, color, thickness)
        
    
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
            
    


    finger_angle_points = {
        "thumb":[2,3,4],
        "index":[5,6,8],
        "middle":[9,10,12],
        "ring":[13,14,16],
        "pinky":[17,18,20]
    }

    finger_percentages = {}

    for finger, points in finger_angle_points.items():
        if(all(id in finger_points[finger] for id in points)):
            angle = calculate_angle(
                finger_points[finger][points[0]],
                finger_points[finger][points[1]],
                finger_points[finger][points[2]]
            )
            
            if finger == "thumb":
                percentage = closure_percentage(angle, 170, 150)
            else:
                percentage = closure_percentage(angle)
            
            finger_percentages[finger] = percentage
            cv.putText(
                inference_screen, 
                f'{finger} : {percentage}% {" close" if percentage > 90 else " open"}', 
                (10, 40 + list(finger_angle_points.keys()).index(finger)*30), 
                cv.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255,255,255), 
                3
            )
            
    cv.imshow("Inference Screen", inference_screen)

    if 8 in finger_points["index"] and "thumb" in finger_percentages and finger_percentages["thumb"] > 90:
            cv.circle(resized_frame, finger_points["index"][8], 5, (255,255,0), -1)

            current_point = finger_points["index"][8]

            if len(previous_dots) == 0:
                previous_dots.append(current_point)

            else:
                last_point = previous_dots[-1]

                distance = np.sqrt(
                    (current_point[0] - last_point[0])**2 +
                    (current_point[1] - last_point[1])**2
                )

                if distance > 5:
                    previous_dots.append(current_point)

    cv.imshow("Frame", resized_frame)
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        previous_dots.clear()

cap.release()
cv.destroyAllWindows()
