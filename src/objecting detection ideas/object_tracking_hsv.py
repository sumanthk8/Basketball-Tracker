import cv2
import numpy as np
import imutils
from collections import deque

cap = cv2.VideoCapture("../videos/Steph Curry 3 point contest.mp4")

cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);


_, frame = cap.read()

while True:

    bballLower = (0, 75, 65)
    bballUpper = (15, 255, 255)

    if frame is None:
        break

    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bballLower, bballUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)



        if area > 250:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Blurred", blurred)
    # create trackbars for high,low H,S,V

    key = cv2.waitKey(24)
    if key == 27:
        break
    if key == 13:
        print(bballLower)
        print(bballUpper)
        _, frame = cap.read()


cap.release()
cv2.destroyAllWindows()
