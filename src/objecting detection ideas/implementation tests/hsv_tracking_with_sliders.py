import cv2
import numpy as np
import imutils
from collections import deque


# trackbar callback fucntion to update HSV value
def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    # assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')
    V_low = cv2.getTrackbarPos('low V', 'controls')
    V_high = cv2.getTrackbarPos('high V', 'controls')
    print(H_low, S_low, V_low)
    print(H_high, S_high, V_high)

#global variable
H_low = 4
H_high = 18
S_low= 45
S_high = 255
V_low= 60
V_high = 150


cap = cv2.VideoCapture("../../videos/Steph Curry 3 point contest.mp4")


cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);

cv2.createTrackbar('low H', 'controls', 0, 179, callback)
cv2.createTrackbar('high H', 'controls', 179, 179, callback)

cv2.createTrackbar('low S', 'controls', 0, 255, callback)
cv2.createTrackbar('high S', 'controls', 255, 255, callback)

cv2.createTrackbar('low V', 'controls', 0, 255, callback)
cv2.createTrackbar('high V', 'controls', 255, 255, callback)

cv2.setTrackbarPos('low H', 'controls', H_low)
cv2.setTrackbarPos('low S', 'controls', S_low)
cv2.setTrackbarPos('low V', 'controls', V_low)
cv2.setTrackbarPos('high H', 'controls', H_high)
cv2.setTrackbarPos('high S', 'controls', S_high)
cv2.setTrackbarPos('high V', 'controls', V_high)

_, frame = cap.read()

while True:

    bballLower = (H_low, S_low, V_low)
    bballUpper = (H_high, S_high, V_high)

    # bballLower = (4, 45, 60)
    # bballUpper = (18, 255, 150)

    if frame is None:
        break

    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bballLower, bballUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    # create trackbars for high,low H,S,V

    key = cv2.waitKey(24)
    if key == 27:
        break
    if key == 13:
        _, frame = cap.read()


cap.release()
cv2.destroyAllWindows()
