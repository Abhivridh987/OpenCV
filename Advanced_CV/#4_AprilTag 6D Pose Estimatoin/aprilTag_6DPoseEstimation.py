import cv2 as cv
import numpy as np
import math
import time
from pupil_apriltags import Detector

FONT_SIZE = 0.3
THICKNESS = 1
COLOR = (0,0,0)

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

def plot_qr_boundary(img, points):
    points = points.astype(int)

    top_left = points[0]
    top_right = points[1]
    bottom_right = points[2]
    bottom_left = points[3]

    for i, point in enumerate(points):
        x,y = point

        #Plot Points
        cv.circle(img, (x,y), 5, (0,0,255), -1)

        #label points
        cv.putText(img, str(i), (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)


    for i in range(4):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1)%4])

        cv.line(img, pt1, pt2, (0,255,0), 2)

def plot_center(img, points, returns=False):
    center_x = int(np.mean(points[:,0]))
    center_y = int(np.mean(points[:,1]))
    center = tuple([center_x, center_y])

    if returns is True:
       return center
    
    cv.circle(img, center, 4, (255,0,0), -1)

def orientation2D(img, points, returns=False):
    center_x, center_y = plot_center(img, points, True)
    x0, y0 = points[0]
    x1, y1 = points[1]

    dx = x1 - x0
    dy = y1 - y0

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    if returns is True:
        return angle_deg
    
    x2, y2 = points[2].astype(int)
    cv.putText(img, f'2D Angle : {round(float(angle_deg), 2)}', (int(x2)+10,int(y2)+5), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR, THICKNESS)
    
    arrow_length = 100
    end_x = int(center_x + arrow_length * np.cos(angle_rad))
    end_y = int(center_y + arrow_length * np.sin(angle_rad))
    
    
    
    cv.arrowedLine(img, (center_x, center_y), (end_x, end_y), (255,0,255), 3)

def extract_tvec_rvec(points, object_points, camera_matrix, dist_coeffs):
    image_points = points.astype(np.float32)
    success_pnp, rvec, tvec = cv.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )
    return (success_pnp, rvec, tvec)

def extract_position(frame, tvec, draw=False, points=None):
    if draw == True:
        x2, y2 = points[2]
        x2 = int(x2)
        y2 = int(y2)
        cv.putText(
            frame, 
            f'X : {float(tvec[0][0]):.2f} mm',
            (x2+10,y2+15),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            COLOR,
            THICKNESS
        )

        cv.putText(
            frame,
            f'Y : {float(tvec[1][0]):.2f} mm',
            (x2+10, y2+25),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            COLOR,
            THICKNESS
        )

        cv.putText(
            frame,
            f'Z : {float(tvec[2][0]):.2f} mm',
            (x2+10, y2+35),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            COLOR,
            THICKNESS
        )

    return (tvec[0][0], tvec[1][0], tvec[2][0])

def extract_roll_pitch_yaw(frame, rvec, draw=False, points=None):
    # Convert Rotation Vector -> Rotation Matrix
    rotation_matrix, _ = cv.Rodrigues(rvec)

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
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    if draw is True:
        x2,y2 = points[2].astype(int)
        cv.putText(
            frame,
            f'Roll : {roll:.2f}',
            (x2+10, y2+45),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            COLOR,
            THICKNESS
        )
        cv.putText(
            frame,
            f'Pitch : {pitch:.2f}',
            (x2+10,y2+55),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            COLOR,
            THICKNESS
        )
        cv.putText(
            frame,
            f'Yaw : {yaw:.2f}',
            (x2+10,y2+65),
            cv.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            COLOR,
            THICKNESS
        )
    return (roll, pitch, yaw)

def fps(frame, prev_time, show=False):
    curr_time = time.time()
    deno = curr_time - prev_time
    frames_per_second = 1 / deno

    if show is True:
        cv.putText(frame, f'FPS : {round(float(frames_per_second), 2)}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return curr_time, frames_per_second

TAG_SIZE = 50

# Paths

CAMERA_MATRIX_PATH = 'calibration/camera_matrix.npy'
DIST_COEFFS_PATH = 'calibration/dist_coeffs.npy'

# Load Calibration File
camera_matrix = np.load(CAMERA_MATRIX_PATH)
dist_coeffs = np.load(DIST_COEFFS_PATH)

#Real World Corners
object_points = np.array(
    [
        [0, 0, 0],
        [TAG_SIZE, 0, 0],
        [TAG_SIZE, TAG_SIZE, 0],
        [0, TAG_SIZE, 0]
    ],
    dtype=np.float32
)

cap = cv.VideoCapture(0)

#Importing April Tag Detector
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1
)

prev_time = 0

while True:
    true, frame = cap.read()
    # frame = cv.flip(frame, 1)

    #GrayScale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    detections = detector.detect(gray)

    for detection in detections:

        tag_id = detection.tag_id
        points = detection.corners
        # If April Tag is Found
        if points is not None :
            x1,y1 = points[2]
            x1, y1 = int(x1), int(y1)
            cv.putText(frame, f'Tag ID: {tag_id}', (x1 + 10, y1 - 5),cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, COLOR, THICKNESS)
            plot_qr_boundary(frame, points)
            plot_center(frame, points)
            orientation2D(frame, points)

            success_pnp, rvec, tvec = extract_tvec_rvec(points, object_points, camera_matrix, dist_coeffs)
            if success_pnp :
                cv.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    30
                )

                x, y, z = extract_position(frame,tvec,True, points)
                roll, pitch, yaw = extract_roll_pitch_yaw(frame, rvec, True, points)

    curr_time, frames_per_second = fps(frame, prev_time, True)
    prev_time = curr_time

    cv.imshow("QR Code Detector", frame)
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()