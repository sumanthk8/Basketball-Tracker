import cv2
import numpy as np
import imutils
from collections import deque
import math

cap = cv2.VideoCapture("../../videos/Steph Curry 3 point contest.mp4")

cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550, 10);


_, frame = cap.read()

once = False

while True:

    bballLower = (5, 80, 60)
    bballUpper = (16, 255, 150)

    if frame is None:
        break

    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bballLower, bballUpper)
    mask = cv2.erode(mask, None, iterations=1)
    # mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bestContour = None;
    bestMatchFactor = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 250:

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            minCircArea = math.pi * radius * radius
            # ((rx, ry), (width, height), angle) = cv2.minAreaRect(cnt)
            # minRectArea = width*height
            # print(minRectArea)
            # print(minCircArea)

            if (y > 275):
                continue

            # if minCircArea > minRectArea:
            #     continue

            matchFactor = area / (minCircArea)

            if matchFactor > bestMatchFactor and matchFactor > .5:
                bestContour = cnt
                bestMatchFactor = matchFactor

    if bestContour is not None:
        if not once:
            (x, y), radius = cv2.minEnclosingCircle(bestContour)
            print(bestMatchFactor)
            print(radius*radius*math.pi)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0))
            once = True


    cv2.imshow("Frame", frame)
    cv2.imshow("Blurred", mask)
    # create trackbars for high,low H,S,V

    key = cv2.waitKey(24)
    if key == 27:
        break
    # if key == 13:
    _, frame = cap.read()
    once = False



cap.release()
cv2.destroyAllWindows()
