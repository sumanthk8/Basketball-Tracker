import cv2
import numpy as np
from object_detection import ObjectDetection

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("videos/Steph Curry 3 point contest.mp4")

while True:
    _, frame = cap.read()

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(24)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()