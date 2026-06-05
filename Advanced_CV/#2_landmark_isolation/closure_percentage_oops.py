import cv2 as cv
import numpy as np
import mediapipe as mp
import serial
import time
from Utility import Utility
from HandDetection import HandDetection


def main():

    cap = cv.VideoCapture(0)
    true, frame = cap.read()
    inference_screen = np.zeros(frame.shape[:-1], dtype='uint8')

    detector = HandDetection()

    #Initialising Serial Port
    ser = serial.Serial('COM10', 9600, timeout=1)
    time.sleep(2) # giving time for arduino to rest

    while True:
        true, frame = cap.read()
        
        resized_frame = Utility.resize_function(frame=frame, scale=1)
        
        resized_frame = cv.flip(resized_frame, 1)
        finger_points = detector.extract_finger_points(resized_frame)
        UART_MESSAGE = detector.angle_extraction(inference_screen=inference_screen, finger_points=finger_points)
        print(UART_MESSAGE)
        ser.write((UART_MESSAGE + '\n').encode('utf-8'))
        detector.closure_detection(inference_screen, finger_points)


        cv.imshow("Frame", resized_frame)
        key = cv.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    ser.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()