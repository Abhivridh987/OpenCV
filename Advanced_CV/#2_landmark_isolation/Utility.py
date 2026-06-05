import cv2 as cv
import numpy as np
import mediapipe as mp

class Utility():
    @staticmethod
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

    @staticmethod
    def calculate_angle(a,b,c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - 180.0
        return angle

    @staticmethod
    def closure_percentage(angle, max_angle = 160, min_angle =60):
        if angle > max_angle:
            return 0
        elif angle < min_angle:
            return 100
        else:
            return int((max_angle - angle) / (max_angle - min_angle) * 100)
        