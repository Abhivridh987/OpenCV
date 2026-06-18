import os
import cv2 as cv

frames_folder = "frames"

files = sorted(
    os.listdir(frames_folder),
    key=lambda x:
    int(
        x.split("_")[1]
        .split(".")[0]
    )
)

first = cv.imread(
    os.path.join(
        frames_folder,
        files[0]
    )
)

h, w = first.shape[:2]

writer = cv.VideoWriter(
    "road_output.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h)
)

frame_num=0
for file in files:

    frame = cv.imread(
        os.path.join(
            frames_folder,
            file
        )
    )
    
    writer.write(frame)
    print(f"Frame {frame_num} written ")
    frame_num+=1

writer.release()

print("Video saved")