import cv2
import numpy as np
import imutils
from collections import deque

cap = cv2.VideoCapture("../videos/Steph Curry 3 point contest.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2()

while True:

    _, frame = cap.read()

    mask = object_detector.apply(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500 and area < 5000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(24)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
