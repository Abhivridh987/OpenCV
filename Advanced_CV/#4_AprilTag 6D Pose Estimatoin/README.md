# Real-Time AprilTag 6DoF Pose Estimation and 3D Visualization

## Overview

This project implements a complete real-time AprilTag-based 6 Degrees of Freedom (6DoF) pose estimation system using OpenCV and AprilTags.

The system detects AprilTags from a webcam feed, estimates their 3D position and orientation relative to the camera, and visualizes the pose using coordinate axes.

The project was developed to understand the mathematics and implementation behind camera calibration, pose estimation, rotation matrices, Rodrigues transformation, and 3D computer vision.

---

## Features

* Camera calibration using checkerboard images
* Automatic extraction of camera intrinsic parameters
* Lens distortion estimation and correction
* Real-time AprilTag detection
* Multi-tag support
* 2D orientation estimation
* 3D pose estimation using solvePnP
* Position estimation (X, Y, Z)
* Rotation estimation (Roll, Pitch, Yaw)
* 3D coordinate frame visualization
* Real-time FPS monitoring

---

## Technologies Used

* Python
* OpenCV
* NumPy
* pupil_apriltags

---

## Project Workflow

### Phase 1: AprilTag Detection

Detect AprilTags from webcam frames and extract corner coordinates.

### Phase 2: Camera Calibration

Capture checkerboard images from multiple viewpoints and estimate:

* Camera Matrix
* Distortion Coefficients

using OpenCV's calibration pipeline.

### Phase 3: Pose Estimation

Use solvePnP to estimate:

* Rotation Vector (rvec)
* Translation Vector (tvec)

from detected tag corners.

### Phase 4: Orientation Extraction

Convert the rotation vector into a rotation matrix using Rodrigues transformation.

Extract:

* Roll
* Pitch
* Yaw

from the rotation matrix.

### Phase 5: 3D Visualization

Visualize the local coordinate frame of each AprilTag using:

cv.drawFrameAxes()

---

## Camera Calibration

### Checkerboard Configuration

* Checkerboard Size: 9 × 6 inner corners
* Square Size: 27 mm

### Calibration Outputs

* camera_matrix.npy
* dist_coeffs.npy

Example calibration result:

Camera Matrix:

[[973.99,   0.00, 620.38],
[  0.00, 978.37, 324.43],
[  0.00,   0.00,   1.00]]

Mean Reprojection Error:

0.286 pixels

---

## Pose Information

For each detected AprilTag:

### Position

* X (mm)
* Y (mm)
* Z (mm)

### Orientation

* Roll (degrees)
* Pitch (degrees)
* Yaw (degrees)

### Additional Information

* Tag ID
* 2D Orientation Angle
* Coordinate Axes

---

## Project Structure

```text
AprilTag_6DPoseEstimation/
│
├── calibration/
│   ├── calibration.py
│   ├── images/
│   ├── camera_matrix.npy
│   └── dist_coeffs.npy
│
├── aprilTag_6DPoseEstimation.py
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AprilTag_6DPoseEstimation.git
```

Install dependencies:

```bash
pip install opencv-python
pip install numpy
pip install pupil-apriltags
```

---

## Running the Project

Run the pose estimation script:

```bash
python aprilTag_6DPoseEstimation.py
```

Press:

```text
q
```

to exit.

---

## Mathematical Concepts Used

### Camera Calibration

Maps 3D world coordinates to 2D image coordinates using intrinsic camera parameters.

### Perspective-n-Point (PnP)

Estimates camera pose from:

* Known 3D points
* Corresponding 2D image points

### Rodrigues Transformation

Converts rotation vectors into rotation matrices.

### Euler Angles

Extracts:

* Roll
* Pitch
* Yaw

from the rotation matrix.

---

## Applications

* Robotics
* Autonomous Navigation
* Drone Localization
* Augmented Reality (AR)
* Industrial Automation
* Camera Tracking
* Marker-Based Pose Estimation

---

## Future Improvements

* Kalman Filter Pose Smoothing
* AprilTag Grid Mapping
* Multi-Camera Pose Estimation
* Marker-Based SLAM
* 3D Object Overlay
* AR Applications

---

## Author

Abhivridh

B.Tech Computer Science and Engineering

College of Engineering Trivandrum (CET)

---

## License

This project is released under the MIT License.
