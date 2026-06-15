import cv2 as cv
import numpy as np
import math
import time
import serial
from pupil_apriltags import Detector


class AprilTagPlotter():
    def __init__(self, frame, detection):
        self.frame = frame
        self.points = detection.corners
        self.tag_id = detection.tag_id
        if self.tag_id == 409:
            self.isCar = True
        if self.tag_id == 408:
            self.isParking = True

    def get_tag_id(self, draw=False):

        if draw is True:
            x2, y2 = self.points[2]
            x2, y2 = int(x2), int(y2)
            cv.putText(self.frame, f'Tag Id : {self.tag_id}', (x2 + 10, y2 - 5), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR, THICKNESS)

        return self.tag_id
    
    def plot_april_boundary(self):
        points = self.points.astype(int)

        for i, point in enumerate(points):
            x,y = point

            #Plot Points
            cv.circle(self.frame, (x,y), 5, (0,0,255), -1)

            #label points
            cv.putText(self.frame, str(i), (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)


        for i in range(4):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1)%4])

            cv.line(self.frame, pt1, pt2, (0,255,0), 2)
    
    def get_center(self, draw=False, pointThickness=4, color=(255,0,0)):
        center_x = int(np.mean(self.points[:,0]))
        center_y = int(np.mean(self.points[:,1]))
        center = tuple([center_x, center_y])

        if draw is True:
            cv.circle(self.frame, center, pointThickness, color, -1)

        self.tag_cx = center_x
        self.tag_cy = center_y

        return center
            
    def orientation2D(self, draw=False):
        center_x, center_y = self.get_center()
        x0, y0 = self.points[0]
        x1, y1 = self.points[1]

        dx = x1 - x0
        dy = y1 - y0

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if draw is True:
            x2, y2 = self.points[2].astype(int)
            
            cv.putText(self.frame, f'2D Angle : {round(float(angle_deg), 2)}', (int(x2)+10,int(y2)+5), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR, THICKNESS)
            
            arrow_length = 100
            end_x = int(center_x + arrow_length * np.cos(angle_rad))
            end_y = int(center_y + arrow_length * np.sin(angle_rad))
            
            
            
            cv.arrowedLine(self.frame, (center_x, center_y), (end_x, end_y), (255,0,255), 3)

        return angle_deg
        
    def getBotBoundary(self,  draw=False, drawColor=(0,255,0), fill=False, alpha=0.4, fillColor=(0,255,0)):
    
        h, w = self.frame.shape[:2]
        fx = CAMERA_MATRIX[0, 0]
        fy = CAMERA_MATRIX[1, 1]
        height = self.tvec[2][0]       
        # Robot dimensions (mm) - adjust based on your actual robot
        robot_width = 150  # Width of robot (left to right)
        robot_length = 150  # Length of robot (front to back)
        
        # Get tag orientation (yaw angle in degrees)
        tag_angle_deg = self.orientation2D()
        tag_angle_rad = np.radians(tag_angle_deg)
        
        # Get tag center
        center_x, center_y = self.get_center()
        
        # Calculate pixel scaling factors
        pixel_per_mm_x = fx / height
        pixel_per_mm_y = fy / height
        
        # Convert robot dimensions to pixels
        robot_width_px = int(robot_width * pixel_per_mm_x)
        robot_length_px = int(robot_length * pixel_per_mm_y)
        
        # Define robot rectangle relative to tag center (in mm)
        # Assuming tag is at center of robot
        half_width = robot_width / 2
        half_length = robot_length / 2
        
        # Four corners of robot in tag coordinates (mm)
        robot_corners_mm = np.array([
            [-half_width, -half_length],  # Front-left (relative to tag)
            [ half_width, -half_length],  # Front-right
            [ half_width,  half_length],  # Back-right
            [-half_width,  half_length]   # Back-left
        ])
        
        # Rotate corners based on tag orientation
        rotation_matrix = np.array([
            [np.cos(tag_angle_rad), -np.sin(tag_angle_rad)],
            [np.sin(tag_angle_rad),  np.cos(tag_angle_rad)]
        ])
        
        rotated_corners = np.dot(robot_corners_mm, rotation_matrix.T)
        
        # Convert to pixels and add tag center
        pixel_corners = []
        for corner in rotated_corners:
            x_px = int(center_x + corner[0] * pixel_per_mm_x)
            y_px = int(center_y + corner[1] * pixel_per_mm_y)
            pixel_corners.append((x_px, y_px))
        
        if draw:
            # Draw robot boundary with dotted lines
            corners = pixel_corners
            draw_dotted_line(self.frame, corners[0], corners[1], drawColor, dot_size=5)
            draw_dotted_line(self.frame, corners[1], corners[2], drawColor, dot_size=5)
            draw_dotted_line(self.frame, corners[2], corners[3], drawColor, dot_size=5)
            draw_dotted_line(self.frame, corners[3], corners[0], drawColor, dot_size=5)
            
            # Draw front indicator (arrow or colored line)
            front_center = ((corners[1][0] + corners[2][0]) // 2, 
                        (corners[1][1] + corners[2][1]) // 2)
            front_direction = (
                front_center[0] + int(40 * np.cos(tag_angle_rad)),
                front_center[1] + int(40 * np.sin(tag_angle_rad))
            )
            self.front_center  = front_center
            cv.arrowedLine(self.frame, front_center, front_direction, (0, 255, 255), 3)
        
        if fill == True:
            # Debug print
              
            overlay = self.frame.copy()
            pts = np.array(pixel_corners, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.fillPoly(overlay, [pts], fillColor)
            cv.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)
        
        # Return corners for further use
        return pixel_corners
    
    def extract_tvec_rvec(self):
        image_points = self.points.astype(np.float32)
        success_pnp, rvec, tvec = cv.solvePnP(
                OBJECT_POINTS,
                image_points,
                CAMERA_MATRIX,
                DIST_COEFFS
        )
        self.success_pnp = success_pnp
        self.tvec = tvec
        self.rvec = rvec
        return (success_pnp, rvec, tvec)

    def extract_position(self, draw=False):
        if draw == True and self.success_pnp == True:
            x2, y2 = self.points[2]
            x2 = int(x2)
            y2 = int(y2)
            cv.putText(
                self.frame, 
                f'X : {float(self.tvec[0][0]):.2f} mm',
                (x2+10,y2+15),
                cv.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COLOR,
                THICKNESS
            )

            cv.putText(
                self.frame,
                f'Y : {float(self.tvec[1][0]):.2f} mm',
                (x2+10, y2+25),
                cv.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COLOR,
                THICKNESS
            )

            cv.putText(
                self.frame,
                f'Z : {float(self.tvec[2][0]):.2f} mm',
                (x2+10, y2+35),
                cv.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COLOR,
                THICKNESS
            )
        
        self.x = self.tvec[0][0]
        self.y = self.tvec[1][0]
        self.z = self.tvec[2][0]

        return (self.tvec[0][0], self.tvec[1][0], self.tvec[2][0])

    def extract_roll_pitch_yaw(self, draw=False):
        # Convert Rotation Vector -> Rotation Matrix
        rotation_matrix, _ = cv.Rodrigues(self.rvec)

        # Calculate sy
        sy = math.sqrt(
            rotation_matrix[0,0]**2 +
            rotation_matrix[1,0]**2
        )

        singular = sy < 1e-6

        if not singular:

            roll = math.atan2(
                rotation_matrix[2,1],
                rotation_matrix[2,2]
            )

            pitch = math.atan2(
                -rotation_matrix[2,0],
                sy
            )

            yaw = math.atan2(
                rotation_matrix[1,0],
                rotation_matrix[0,0]
            )

        else:

            roll = math.atan2(
                -rotation_matrix[1,2],
                rotation_matrix[1,1]
            )

            pitch = math.atan2(
                -rotation_matrix[2,0],
                sy
            )

            yaw = 0

        # Convert radians -> degrees
        self.roll = np.degrees(roll)
        self.pitch = np.degrees(pitch)
        self.yaw = np.degrees(yaw)

        if draw is True:
            x2,y2 = self.points[2].astype(int)
            cv.putText(
                self.frame,
                f'Roll : {self.roll:.2f}',
                (x2+10, y2+45),
                cv.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COLOR,
                THICKNESS
            )
            cv.putText(
                self.frame,
                f'Pitch : {self.pitch:.2f}',
                (x2+10,y2+55),
                cv.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COLOR,
                THICKNESS
            )
            cv.putText(
                self.frame,
                f'Yaw : {self.yaw:.2f}',
                (x2+10,y2+65),
                cv.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COLOR,
                THICKNESS
            )
        return (self.roll, self.pitch, self.yaw)

    def plotTag(self):
        self.plot_april_boundary()
        self.get_center(draw=True)
        self.get_tag_id(draw=True)
        self.orientation2D(draw=True)

class RobotController():
    def __init__(self, frame, Car, Parking):
        self.Car = Car
        self.Parking = Parking
        self.frame = frame
    
    def drawPath(self, draw=True):
        if draw:
            car_cx , car_cy = self.Car.tag_cx, self.Car.tag_cy
            parking_cx, parking_cy = self.Parking.tag_cx, self.Parking.tag_cy
            cv.line(self.frame, (car_cx, car_cy), (parking_cx, parking_cy), (0,255,0), 2)

    
    def Car_Parking_Angle(self):
        car_center = (self.Car.tag_cx, self.Car.tag_cy)
        parking_center = (self.Parking.tag_cx, self.Parking.tag_cy)
        car_front_center = self.Car.front_center

        # Angle of car's heading direction
        car_heading_angle = angle_between(car_center, car_front_center)

        # Angle from car to parking spot (target direction)
        target_angle = angle_between(car_center, parking_center)

        # Angle the car needs to turn (steering correction)
        angle_to_turn = target_angle - car_heading_angle

        # Normalize to [-180, 180]
        angle_to_turn = (angle_to_turn + 180) % 360 - 180

        print('Angle to Turn : ', angle_to_turn)
        self.angle_to_turn = angle_to_turn

        return angle_to_turn

    def alignRobot(self, ser, start, outerAngle=20, innerAngle=10):
        if start:
            if (self.angle_to_turn > outerAngle or self.angle_to_turn < -outerAngle):
                if self.angle_to_turn < -innerAngle:
                    UART_MESSAGE = 'R 30'
                elif self.angle_to_turn > innerAngle:
                    UART_MESSAGE = 'L 30'
                print(UART_MESSAGE)
            else:
                UART_MESSAGE = 'S 100'
                print(UART_MESSAGE)
            
            ser.write((UART_MESSAGE + '\n').encode('utf-8'))
        else:
            UART_MESSAGE = 'S 100'
            print(UART_MESSAGE)
            ser.write((UART_MESSAGE + '\n').encode('utf-8'))

    def moveRobot(self, ser, start):
        car_center = (self.Car.tag_cx, self.Car.tag_cy)
        parking_center = (self.Parking.tag_cx, self.Parking.tag_cy)
        distance = distance_between(car_center, parking_center)
        if start:
            if distance > ((20 * CAMERA_MATRIX[0,0]) / self.Car.z):
                UART_MESSAGE = 'F 40'
                print(UART_MESSAGE)
                ser.write((UART_MESSAGE + '\n').encode('utf-8'))
            else:
                UART_MESSAGE = 'S 250'
                print(UART_MESSAGE)
                ser.write((UART_MESSAGE + '\n').encode('utf-8'))

def fps(frame, prev_time, show=False):
    curr_time = time.time()
    deno = curr_time - prev_time
    frames_per_second = 1 / deno

    if show is True:
        cv.putText(frame, f'FPS : {round(float(frames_per_second), 2)}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return curr_time, frames_per_second

def draw_dotted_line(img, pt1, pt2, color, dot_size=3, spacing=5):

    # Calculate line length and direction
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return
    
    # Normalize direction vector
    dx = dx / length
    dy = dy / length
    
    # Draw dots along the line
    distance = 0
    while distance < length:
        x = int(x1 + dx * distance)
        y = int(y1 + dy * distance)
        cv.circle(img, (x, y), dot_size // 2, color, -1)
        distance += dot_size + spacing

def angle_between(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def distance_between(p1, p2):
    distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    return int(distance)

FONT_SIZE = 0.3
THICKNESS = 1
COLOR = (0,0,0)

#TAG_SIZE
TAG_SIZE = 40

# PATHS
CAMERA_MATRIX_PATH = '../calibration/camera_matrix.npy'
DIST_COEFFS_PATH = '../calibration/dist_coeffs.npy'

# Load Calibration File
CAMERA_MATRIX = np.load(CAMERA_MATRIX_PATH)
DIST_COEFFS = np.load(DIST_COEFFS_PATH)

#Real World Corners
HALF_TAG = TAG_SIZE / 2

OBJECT_POINTS = np.array(
    [
        [-HALF_TAG, -HALF_TAG, 0],
        [ HALF_TAG, -HALF_TAG, 0],
        [ HALF_TAG,  HALF_TAG, 0],
        [-HALF_TAG,  HALF_TAG, 0]
    ],
    dtype=np.float32
)

#Setting Resolution
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

#Importing April Tag Detection
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1
)

prev_time = 0
waitDelay = 1

drawPath = False
startRotation = False
startTranslation = False
outerAngle = 25
innerAngle = 10

previousCarParkingAngle = 0

#Initialising Serial Port
ser = serial.Serial('COM9', 9600, timeout=1)
time.sleep(2) # giving time for esp 32 to rest

while True:
    Car = None
    Parking = None
    ret, frame = cap.read()
    if not ret:
        print("Camera Error")
        exit()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        tag_id = detection.tag_id
        points = detection.corners

        if points is not None:
            Plotter = AprilTagPlotter(frame, detection)
            Plotter.plotTag()

            #Extracting 6DoF
            success_pnp, rvec, tvec = Plotter.extract_tvec_rvec()

            if success_pnp :
                
                if tag_id == 409:
                    Car = Plotter
                    Plotter.extract_position(draw=True)
                    Plotter.extract_roll_pitch_yaw(draw=True)
                    Plotter.getBotBoundary(draw=True,drawColor=(0,0,255), fill=True, alpha=0.3,fillColor=(0,0,255))
                
                elif tag_id == 408:
                    Parking = Plotter
                    Plotter.extract_position(draw=True)
                    Plotter.extract_roll_pitch_yaw(draw=True)
                    Plotter.getBotBoundary(draw=True,drawColor=(0,255,0), fill=True, alpha=0.3,fillColor=(0,255,0))
    
    if Car is not None and Parking is not None:
        controller = RobotController(frame, Car, Parking)
        controller.drawPath(drawPath)
        controller.Car_Parking_Angle()
        print('Starting Alignment')
        controller.alignRobot(ser=ser, start=startRotation, outerAngle=outerAngle, innerAngle=innerAngle)

        curr_carParkingAngle = controller.Car_Parking_Angle()
        if curr_carParkingAngle < 25 and curr_carParkingAngle > -25:
            print('Starting Translation')
            controller.moveRobot(ser=ser, start=startTranslation)
        
        

    #FPS And Wait Delay
    curr_time, frames_per_second = fps(frame, prev_time, True)
    prev_time = curr_time
    cv.putText(frame, f'Wait Delay : {waitDelay}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.putText(frame, f'Draw Path : {drawPath}', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.putText(frame, f'Start Rotation : {startRotation}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.putText(frame, f'Start Translation: {startTranslation}', (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.putText(frame, f'Outer Angle : {outerAngle}', (10, 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.putText(frame, f'Inner Angle : {innerAngle}', (10, 160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    cv.imshow("Frame", frame)

    key = cv.waitKey(waitDelay) & 0xFF
    if key == ord('q'):
        break
    if key == ord('d'):
        if drawPath == True:
            drawPath = False
        elif drawPath == False:
            drawPath = True
    if key == ord('r'):
        if startRotation == True:
            startRotation = False
        elif startRotation == False:
            startRotation = True
    if key == ord('t'):
        if startTranslation == True:
            startTranslation= False
        elif startTranslation == False:
            startTranslation = True

ser.close()
cap.release()
cv.destroyAllWindows()
