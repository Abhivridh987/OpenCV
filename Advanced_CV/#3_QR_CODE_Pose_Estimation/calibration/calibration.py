import cv2 as cv
import numpy as np
import glob
import os

# Checker Board Configuraration

CHECKERBOARD = (9,6)
SQUARE_SIZE = 27

# Storage Lists

object_points = []
image_points = []

# Real World Coordinates

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1,2)

objp*=SQUARE_SIZE

# Corner Refinement
criteria = (
    cv.TERM_CRITERIA_EPS + 
    cv.TERM_CRITERIA_MAX_ITER, 
    30, 0.001
)

#Load Images

images = glob.glob("images/*.jpg")
successful_images = 0
failed_images = 0

if len(images) == 0:
    print("Wrong path or no images exist")
    exit()

image_size = None

for image_path in images:

    img = cv.imread(image_path)

    if img is None:
        print(f'Could not Load Image from Image Path : {image_path}')
        continue
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]

    
    ret, corners = cv.findChessboardCorners(
        gray, 
        CHECKERBOARD,
        cv.CALIB_CB_ADAPTIVE_THRESH +
        cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        successful_images+=1
        refined_corners = cv.cornerSubPix(
            gray, 
            corners,
            (11,11),
            (-1,-1),
            criteria
        )

        object_points.append(objp)
        image_points.append(refined_corners)
        cv.drawChessboardCorners(
            img, 
            CHECKERBOARD,
            refined_corners,
            ret
        )

        cv.imshow("Detected Corners", img)

        cv.waitKey(200)
        print(f'SUCCESS : {image_path}')
    else:
        failed_images+=1
        print(f'FAILED : {image_path}')

cv.destroyAllWindows()

#Caliberation

print(f"\nSuccessful Images : {successful_images}")
print(f"Failed Images     : {failed_images}")

if successful_images == 0:
    print("No Valid checkerboard pattern detected")
    exit()

if len(object_points) == 0:
    print("No valid checkerboard patterns found")
    exit()

print(f"Object Point Sets : {len(object_points)}")
print(f"Image Point Sets  : {len(image_points)}")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None, 
    None
)

mean_error = 0

for i in range(len(object_points)):
    projected_points, _ = cv.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv.norm(
        image_points[i],
        projected_points,
        cv.NORM_L2
    ) / len(projected_points)

    mean_error += error

mean_error /= len(object_points)

print(f"Mean Reprojection Error : {mean_error}")

# Save Results

np.save('../camera_matrix.npy', camera_matrix)

np.save('../dist_coeffs.npy', dist_coeffs)

# Results

print("Calibration Complete")

print("\nCamera Matrix:\n")
print(camera_matrix)

print("\nDistortion Coefficients:\n")
print(dist_coeffs)

print(f"\nReprojection Error: {ret}")

print("\nSaved:")
print("camera_matrix.npy")
print("dist_coeffs.npy")
