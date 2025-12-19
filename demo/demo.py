import cv2
import numpy as np

# BLUE color range (HSV)
print("Starting object tracking...")
lower = np.array([100, 150, 50])
upper = np.array([140, 255, 255])

cap = cv2.VideoCapture(0)

object_id = 0   # ID counter
objects = {}    # stores ID: (x, y)

def assign_id(x, y, objects, threshold=40):
    global object_id

    # Try to match with existing objects
    for oid, (ox, oy) in objects.items():
        if abs(x - ox) < threshold and abs(y - oy) < threshold:
            objects[oid] = (x, y)
            return oid
    
    # If no match found → new object
    object_id += 1
    objects[object_id] = (x, y)
    return object_id


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            x, y = int(x), int(y)

            # Assign ID to this object
            oid = assign_id(x, y, objects)

            # Draw circle + center
            cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Display ID
            cv2.putText(frame, f"ID {oid}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
