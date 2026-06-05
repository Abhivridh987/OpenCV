import numpy as np
import cv2 as cv
import mediapipe as mp
from Utility import Utility

class HandDetection():
    def __init__(self, staticMode = False, no_of_hands = 2, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.staticMode = staticMode
        self.no_of_hands = no_of_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticMode,
            max_num_hands=self.no_of_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )        
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, resized_frame, draw=True, label=False):
        imgRGB = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                
                if draw == True :
                    # for marking the 21 points on the hand
                    self.mpDraw.draw_landmarks(
                        resized_frame,
                        handLms, 
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0,0,255), thickness = 2, circle_radius = 2),
                        self.mpDraw.DrawingSpec(color=(0,255,0), thickness = 2)
                    )

                if label == True:
                    #Labelling the points on the hand
                    for id, lm in enumerate(handLms.landmark):

                        
                        x= int(lm.x*resized_frame.shape[1])
                        y= int(lm.y*resized_frame.shape[0])

                        cv.putText(
                            resized_frame, 
                            str(id), 
                            (x, y),
                            cv.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0,0,255),
                            2
                        )
        return results
                    
    def extract_finger_points(self, resized_frame, draw=True, label=False):

        results = self.findHands(resized_frame, draw, label)

        finger_points = {
            "thumb":{},
            "index":{},
            "middle":{},
            "ring":{},
            "pinky":{}
        }
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
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

        return finger_points

    def closure_detection(self, inference_screen, finger_points):
        finger_angle_points = {
            "thumb":[2,3,4],
            "index":[5,6,8],
            "middle":[9,10,12],
            "ring":[13,14,16],
            "pinky":[17,18,20]
        }
        inference_screen[:] = 0
        for finger, points in finger_angle_points.items():
            
            if(all(id in finger_points[finger] for id in points)):
                angle = Utility.calculate_angle(
                    finger_points[finger][points[0]],
                    finger_points[finger][points[1]],
                    finger_points[finger][points[2]]
                )
                
                if finger == "thumb":
                    percentage = Utility.closure_percentage(angle, 170, 150)
                else:
                    percentage = Utility.closure_percentage(angle)
                
                
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
     
    def angle_extraction(self, inference_screen, finger_points):
        finger_angle_points = {
            "thumb":[2,3,4],
            "index":[5,6,8],
            "middle":[9,10,12],
            "ring":[13,14,16],
            "pinky":[17,18,20]
        }
        inference_screen[:] = 0
        angles = []
        for finger, points in finger_angle_points.items():
            
            if(all(id in finger_points[finger] for id in points)):
                angle = Utility.calculate_angle(
                    finger_points[finger][points[0]],
                    finger_points[finger][points[1]],
                    finger_points[finger][points[2]]
                )
                angles.append(int(angle))
        
        mess = ""
        for i, angle in enumerate(angles):
            mess+=str(angle)
            if i < len(angles) - 1:
                mess+=','
        return mess
     