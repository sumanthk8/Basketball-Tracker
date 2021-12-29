import cv2
import numpy as np
import imutils
from collections import deque



cap = cv2.VideoCapture("../videos/Steph Curry 3 point contest.mp4")

_, frame = cap.read()

newFrame = True

while True:

    key = cv2.waitKey(24)
    if key == 27:
        break
    if key == 13:
        _, frame = cap.read()
        newFrame = True

    if not newFrame:
        continue

    if frame is None:
        break

    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 500)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(out, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(out, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        cv2.imshow("output", np.hstack([frame, out]))

    newFrame = False


cap.release()
cv2.destroyAllWindows()
