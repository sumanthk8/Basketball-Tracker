import cv2
import numpy as np
import imutils
from collections import deque
import math
from src.kalman_filter.kalman_filter import KF

cap = cv2.VideoCapture("../../videos/jan 6th 2022/freestyle1.MOV")


_, frame = cap.read()

framesWithoutDetection = 0

kfx = None
prevX = -1
kfy = None
prevY = -1

distanceThreshold = 20

new = True

while True:

    if not new:
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == 13:
            _, frame = cap.read()
            new = True
        continue

    bballLower = (0, 0, 0)
    bballUpper = (16, 186, 196)

    if frame is None:
        break

    frame = imutils.resize(frame, width=800)
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bballLower, bballUpper)
    mask = cv2.erode(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bestContour = None;
    bestMatchFactor = 0

    if kfx is not None:
        kfx.predict(1.0 / 59.95)
        xBounds = kfx.twoSidedConfidenceInterval()

        kfy.predict(1.0 / 59.95)
        yBounds = kfy.twoSidedConfidenceInterval()

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 25:

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            minCircArea = math.pi * radius * radius

            matchFactor = area / (minCircArea)

            # predict with kalman filter

            if kfx is not None:
                if (x < xBounds[0] - distanceThreshold) or (x > xBounds[1] + distanceThreshold):
                    continue

                if (y < yBounds[0] - distanceThreshold) or (y > yBounds[1] + distanceThreshold):
                    continue

            if matchFactor > bestMatchFactor and matchFactor > 0.5:
                bestContour = cnt
                bestMatchFactor = matchFactor

    if bestContour is not None:
        (x, y), radius = cv2.minEnclosingCircle(bestContour)

        if kfx is not None:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0))
            cv2.rectangle(frame, (int(xBounds[0]-distanceThreshold), int(yBounds[0]-distanceThreshold)),
                                    (int(xBounds[1]+distanceThreshold), int(yBounds[1]+distanceThreshold)), (0, 0, 255))
            cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0))

            print((int(xBounds[0]-distanceThreshold), int(yBounds[0]-distanceThreshold)),
                    (int(xBounds[1]+distanceThreshold), int(yBounds[1]+distanceThreshold)))

            kfx.update(int(x), .0001)
            kfy.update(int(y), .0001)
        elif prevX < 0:
            prevX = int(x)
            prevY = int(y)
        else:
            kfx = KF(int(x), (x - prevX) / (1.0 / 59.94), 10000)
            kfy = KF(int(y), (y - prevY) / (1.0 / 59.94), 10000)

    cv2.imshow("Frame", frame)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Blurred", mask)
    # create trackbars for high,low H,S,V

    new = False

cap.release()
cv2.destroyAllWindows()
