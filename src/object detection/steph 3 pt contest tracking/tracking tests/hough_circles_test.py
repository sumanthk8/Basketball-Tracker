import cv2
import numpy as np
import imutils
from collections import deque

img = cv2.imread("../images/VY_Canis_Majoris.png")
out = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 500)

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
    cv2.imshow("output", np.hstack([img, out]))
    cv2.waitKey(0)
