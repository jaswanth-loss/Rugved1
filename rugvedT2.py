import cv2 as cv
import numpy as np
from collections import deque
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])
vid = cv.VideoCapture(r'C:\Users\kishg\Downloads\Ball_Tracking.mp4')
points = deque(maxlen=64)
while True:
    isTrue, frame = vid.read()
    if not isTrue:   # when video ends
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_green, upper_green)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    center = None
    if len(contours) > 0:
        # Find the largest contour (most likely the ball)
        c = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        if radius > 10:
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv.circle(frame, center, 5, (0, 0, 255), -1)
    points.appendleft(center)
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv.line(frame, points[i - 1], points[i], (0, 0, 255), 2)
    cv.imshow('Video', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()
