import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# x is raw distance & y is distance in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coeff = np.polyfit(x, y, 2)

# game variables
cx, cy = 250, 250
color = (255, 0, 255)

# loop
while True:
    success, img = cap.read()
    hand = detector.findHands(img, draw=False)

    if hand:
        lmList = hand[0]['lmList']
        x, y, w, h = hand[0]['bbox']
        x1, y1 = lmList[5]
        x2, y2 = lmList[17]

        distance = int(math.sqrt(abs(y2-y1)**2 + abs(x2-x1)**2))
        A, B, C = coeff[0], coeff[1], coeff[2]

        distance_in_cm = A*distance**2 + B*distance + C

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 5)
        cvzone.putTextRect(img, f'{int(distance_in_cm)} cm', (x+10, y+10))

    # draw button
    cv2.circle(img, (cx, cy), 20, color, cv2.FILLED)
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), cv2.FILLED)
    cv2.circle(img, (cx, cy), 15, (255, 255, 255), 2)
    cv2.circle(img, (cx, cy), 20, (0, 0, 0), 3)

    cv2.imshow("Camera", img)
    cv2.waitKey(1)
