import cv2
import numpy as np
import imutils
from collections import deque
import math
from src.kalman_filter.kalman_filter import KF

cap = cv2.VideoCapture("../../videos/jan 6th 2022/freestyle1.MOV")


_, frame = cap.read()

kfx = None
prevX = -1
kfy = None
prevY = -1

new = True

while True:

    if not new:
        key = cv2.waitKey(1)
        if key == 27:
            break
        # if key == 13:
        _, frame = cap.read()
        new = True
        continue

    bballLower = (0, 0, 0)
    bballUpper = (16, 182, 189)

    if frame is None:
        break

    frame = imutils.resize(frame, width=800)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bballLower, bballUpper)
    mask = cv2.erode(mask, None, iterations=1)
    # mask = cv2.dilate(mask, None, iterations=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bestContour = None;
    bestMatchFactor = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 25:

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            minCircArea = math.pi * radius * radius

            matchFactor = area / (minCircArea)

            if matchFactor > bestMatchFactor and matchFactor > .5:
                bestContour = cnt
                bestMatchFactor = matchFactor

            # if prevX < 0:
            #     prevX = x
            #     prevY = y
            #
            # if kf is None and prevX >= 0:
            #     kf = new KF(x, (x-prevX)/(1.0/60))


    if bestContour is not None:
        print(bestMatchFactor)
        (x, y), radius = cv2.minEnclosingCircle(bestContour)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0))

    cv2.imshow("Frame", frame)
    cv2.imshow("Blurred", mask)
    # create trackbars for high,low H,S,V

    new = False

cap.release()
cv2.destroyAllWindows()
